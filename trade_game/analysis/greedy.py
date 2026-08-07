"""基于当前公开行情的贪心经营基准。

该策略只读取游戏已经公开的报价、路线和财务状态，不预知后续的价格随机过程。
它用于衡量当前经济规则下，持续融资、扩充运力并优先套现高收益货物能够达到的水平。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from decimal import Decimal
from statistics import median

from trade_game.core import (
    Borrow,
    Buy,
    BuyTruck,
    Command,
    GameEndReason,
    GameMode,
    GameSession,
    NextDay,
    Product,
    ProductCategory,
    RouteNotFound,
    Sell,
    TransportMode,
    Travel,
    available_credit,
    cargo_quantity,
    create_game_session,
    estimate_travel,
    free_capacity,
    money,
    purchase_cooldown_remaining,
    purchase_unit_price,
    quote_sale,
    remote_sale_distance_premium,
    sale_unit_price,
    settlement_assets,
)


@dataclass(frozen=True, slots=True)
class GreedyConfig:
    """贪心基准的经营偏好与扩张阈值。"""

    vehicle_min_fill_fraction: Decimal = Decimal("0.50")
    vehicle_min_utilization_fraction: Decimal = Decimal("0.80")
    vehicle_payback_multiplier: Decimal = Decimal("1.00")


@dataclass(frozen=True, slots=True)
class TradePlan:
    """一次从当前城市出发、可在期限内完成的采购和套现计划。"""

    product_id: str
    origin: str
    destination: str
    entry_mode: TransportMode | None
    mode: TransportMode
    quantity: int
    purchase_unit_price: Decimal
    expected_sale_unit_price: Decimal
    expected_net_gain: Decimal
    duration_days: int
    entry_days: int

    @property
    def score(self) -> Decimal:
        """以占用每一天能够增加的终局现金作为候选计划排序依据。"""

        return self.expected_net_gain / max(1, self.duration_days)


@dataclass(frozen=True, slots=True)
class LiquidationPlan:
    """将当前货物在某地一次性变现的行动方案。"""

    destination: str
    mode: TransportMode | None
    expected_cash: Decimal
    incremental_gain: Decimal
    duration_days: int

    @property
    def score(self) -> Decimal:
        """以相对当前位置立刻出售的单位时间增益排序。"""

        return self.incremental_gain / max(1, self.duration_days)


@dataclass(frozen=True, slots=True)
class GreedyStep:
    """贪心策略的一步经营记录。"""

    day: int
    location: str
    command: Command
    assets_after: Decimal


@dataclass(frozen=True, slots=True)
class GreedyEpisode:
    """一局贪心经营的终局结果和完整命令轨迹。"""

    seed: int
    final_assets: Decimal
    elapsed_days: int
    end_reason: GameEndReason
    truck_count: int
    command_counts: tuple[tuple[str, int], ...]
    trace: tuple[GreedyStep, ...]


@dataclass(frozen=True, slots=True)
class GreedyEvaluation:
    """多个随机种子上的贪心策略汇总结果。"""

    episodes: tuple[GreedyEpisode, ...]
    mean_final_assets: Decimal
    median_final_assets: Decimal
    minimum_final_assets: Decimal
    maximum_final_assets: Decimal
    bankruptcy_rate: float


class GreedyPolicy:
    """以套现效率为主、主动使用授信并扩张运力的无学习策略。"""

    def __init__(self, config: GreedyConfig | None = None) -> None:
        self.config = config or GreedyConfig()

    def choose(self, session: GameSession) -> Command:
        """根据当前可见市场信息选择下一条核心命令。"""

        state = session.state
        if state.player.cargo_lots:
            return self._choose_cargo_command(session)

        cash = state.player.cash
        credit = available_credit(session.catalog, session.rules, state)
        leveraged_plan = _best_trade_plan(session, spending_power=cash + credit)

        if credit > 0 and _should_borrow(session, leveraged_plan=leveraged_plan):
            return Borrow(amount=credit)

        plan = _best_trade_plan(session, spending_power=state.player.cash)
        if _should_buy_truck(session, plan, self.config):
            return BuyTruck(quantity=1)

        if plan is None:
            return NextDay()
        if plan.origin == state.player.location:
            if purchase_cooldown_remaining(state, plan.origin, plan.product_id) > 0:
                return NextDay()
            return Buy(product_id=plan.product_id, quantity=plan.quantity)
        if plan.entry_mode is None:
            raise RuntimeError("异地产地计划缺少运输方式")
        return Travel(destination=plan.origin, mode=plan.entry_mode)

    def _choose_cargo_command(self, session: GameSession) -> Command:
        plan = _best_liquidation_plan(session)
        if plan is None:
            raise RuntimeError("持有货物时未能生成可执行的套现计划")
        if plan.destination != session.state.player.location:
            if plan.mode is None:
                raise RuntimeError("异地套现计划缺少运输方式")
            travel = _travel_estimate(session, plan.destination, mode=plan.mode)
            if travel is None:
                raise RuntimeError("异地套现计划的路线不可执行")
            travel_cost, _travel_days, _travel_mode, _travel_distance = travel
            if session.state.player.cash < travel_cost:
                credit = available_credit(session.catalog, session.rules, session.state)
                if credit > 0 and session.catalog.city(session.state.player.location).has_bank:
                    return Borrow(amount=credit)
                return _best_sale_command(session)
            return Travel(destination=plan.destination, mode=plan.mode)
        return _best_sale_command(session)


def play_greedy(
    *,
    seed: int,
    config: GreedyConfig | None = None,
    capture_trace: bool = True,
) -> GreedyEpisode:
    """在挑战模式下完整执行一局贪心经营。"""

    session = create_game_session(seed=seed, mode=GameMode.CHALLENGE)
    policy = GreedyPolicy(config)
    initial_day = session.state.day
    trace: list[GreedyStep] = []
    command_counts: Counter[str] = Counter()
    while session.state.outcome is None:
        day = session.state.day
        location = session.state.player.location
        command = policy.choose(session)
        result = session.dispatch(command)
        if not result.accepted:
            raise RuntimeError(f"贪心策略生成了被拒绝的命令：{command!r}")
        command_counts[type(command).__name__] += 1
        if capture_trace:
            trace.append(
                GreedyStep(
                    day=day,
                    location=location,
                    command=command,
                    assets_after=settlement_assets(session.catalog, session.rules, session.state),
                )
            )
    outcome = session.state.outcome
    if outcome is None:
        raise RuntimeError("挑战模式未形成终局结果")
    return GreedyEpisode(
        seed=seed,
        final_assets=outcome.final_assets,
        elapsed_days=session.state.day - initial_day,
        end_reason=outcome.reason,
        truck_count=session.state.player.truck_count,
        command_counts=tuple(sorted(command_counts.items())),
        trace=tuple(trace),
    )


def evaluate_greedy(
    *,
    seeds: tuple[int, ...],
    config: GreedyConfig | None = None,
    capture_trace: bool = False,
) -> GreedyEvaluation:
    """在给定种子集上运行贪心策略并汇总终局资产。"""

    episodes = tuple(play_greedy(seed=seed, config=config, capture_trace=capture_trace) for seed in seeds)
    if not episodes:
        raise ValueError("评估种子不能为空")
    assets = tuple(episode.final_assets for episode in episodes)
    return GreedyEvaluation(
        episodes=episodes,
        mean_final_assets=money(sum(assets, start=Decimal("0")) / len(assets)),
        median_final_assets=money(Decimal(str(median(assets)))),
        minimum_final_assets=min(assets),
        maximum_final_assets=max(assets),
        bankruptcy_rate=sum(
            episode.end_reason is GameEndReason.BANKRUPTCY for episode in episodes
        )
        / len(episodes),
    )


def _best_trade_plan(
    session: GameSession,
    *,
    spending_power: Decimal,
) -> TradePlan | None:
    """枚举在期限内可完成的采购、运输和销售组合。"""

    state = session.state
    catalog = session.catalog
    rules = session.rules
    capacity = free_capacity(state.player.cargo_lots, state.player.truck_total_capacity)
    if capacity == 0 or spending_power <= 0:
        return None
    candidates: list[TradePlan] = []
    for product in catalog.products.values():
        for origin in product.origins:
            entry = _travel_estimate(session, origin)
            if entry is None:
                continue
            entry_cost, entry_days, entry_mode, _entry_distance = entry
            cooldown_wait = max(
                0,
                purchase_cooldown_remaining(state, origin, product.id) - entry_days,
            )
            purchase_price = purchase_unit_price(catalog, rules, state, product.id, origin)
            for destination in catalog.cities:
                if destination == origin:
                    continue
                for mode in TransportMode:
                    delivery = _travel_estimate(session, destination, origin=origin, mode=mode)
                    if delivery is None:
                        continue
                    delivery_cost, delivery_days, _delivery_mode, delivery_distance = delivery
                    budget = max(Decimal("0"), spending_power - entry_cost - delivery_cost)
                    quantity = min(capacity, int(budget / purchase_price))
                    if quantity == 0:
                        continue
                    duration = entry_days + cooldown_wait + 1 + delivery_days + 1
                    if state.day + duration > rules.limits.challenge_max_days:
                        continue
                    retained_quantity = _expected_retained_quantity(
                        product,
                        delivery_days,
                        delivery_distance,
                    )
                    retained_quantity = int(round(quantity * retained_quantity))
                    if retained_quantity == 0:
                        continue
                    sale_price = sale_unit_price(
                        catalog,
                        rules,
                        state,
                        product.id,
                        destination,
                        origin_city=origin,
                        remote_distance_premium=remote_sale_distance_premium(
                            catalog,
                            rules,
                            origin,
                            destination,
                        ),
                    )
                    expected_sale = sale_price * retained_quantity
                    expected_gain = money(
                        expected_sale - purchase_price * quantity - entry_cost - delivery_cost
                    )
                    if expected_gain <= 0:
                        continue
                    candidates.append(
                        TradePlan(
                            product_id=product.id,
                            origin=origin,
                            destination=destination,
                            entry_mode=entry_mode,
                            mode=mode,
                            quantity=quantity,
                            purchase_unit_price=purchase_price,
                            expected_sale_unit_price=sale_price,
                            expected_net_gain=expected_gain,
                            duration_days=duration,
                            entry_days=entry_days,
                        )
                    )
    return max(candidates, key=lambda plan: (plan.score, plan.expected_net_gain), default=None)


def _best_liquidation_plan(session: GameSession) -> LiquidationPlan | None:
    """选择当前库存的最快高价值套现地；优先整车带货后再统一出售。"""

    state = session.state
    catalog = session.catalog
    rules = session.rules
    current_cash = sum(
        (
            quote_sale(
                catalog,
                rules,
                state,
                product_id,
                cargo_quantity(state.player.cargo_lots, product_id),
            ).total
            for product_id in {lot.product_id for lot in state.player.cargo_lots}
        ),
        start=Decimal("0"),
    )
    candidates: list[LiquidationPlan] = []
    for destination in catalog.cities:
        if destination == state.player.location:
            candidates.append(
                LiquidationPlan(
                    destination=destination,
                    mode=None,
                    expected_cash=current_cash,
                    incremental_gain=Decimal("0"),
                    duration_days=1,
                )
            )
            continue
        for mode in TransportMode:
            travel = _travel_estimate(session, destination, mode=mode)
            if travel is None:
                continue
            travel_cost, travel_days, _travel_mode, travel_distance = travel
            expected_cash = Decimal("0")
            for lot in state.player.cargo_lots:
                product = catalog.product(lot.product_id)
                retained_quantity = int(
                    round(
                        lot.quantity
                        * _expected_retained_quantity(product, travel_days, travel_distance)
                    )
                )
                if retained_quantity == 0:
                    continue
                expected_cash += sale_unit_price(
                    catalog,
                    rules,
                    state,
                    lot.product_id,
                    destination,
                    origin_city=lot.origin_city,
                    remote_distance_premium=remote_sale_distance_premium(
                        catalog,
                        rules,
                        lot.origin_city,
                        destination,
                    ),
                ) * retained_quantity
            expected_cash = money(expected_cash - travel_cost)
            if expected_cash > 0:
                candidates.append(
                    LiquidationPlan(
                        destination=destination,
                        mode=mode,
                        expected_cash=expected_cash,
                        incremental_gain=expected_cash - current_cash,
                        duration_days=travel_days + 1,
                    )
                )
    return max(candidates, key=lambda plan: (plan.score, plan.expected_cash), default=None)


def _best_sale_command(session: GameSession) -> Sell:
    """在当前城市出售能立即带来最多现金的一类库存。"""

    product_id = max(
        {lot.product_id for lot in session.state.player.cargo_lots},
        key=lambda candidate: quote_sale(
            session.catalog,
            session.rules,
            session.state,
            candidate,
            cargo_quantity(session.state.player.cargo_lots, candidate),
        ).total,
    )
    return Sell(
        product_id=product_id,
        quantity=cargo_quantity(session.state.player.cargo_lots, product_id),
    )


def _should_borrow(
    session: GameSession,
    *,
    leveraged_plan: TradePlan | None,
) -> bool:
    """只有授信能扩大下一次交易或触发有收益的扩张时才新增债务。"""

    if not session.catalog.city(session.state.player.location).has_bank:
        return False
    credit = available_credit(session.catalog, session.rules, session.state)
    if credit == 0:
        return False
    # 授信会在后续销售后继续扩大，用满可用额度才能为新增运力积累营运资金。
    return leveraged_plan is not None


def _should_buy_truck(
    session: GameSession,
    plan: TradePlan | None,
    config: GreedyConfig,
) -> bool:
    """新增一辆货车须能在剩余时间内覆盖其额外人工成本。"""

    if plan is None:
        return False
    state = session.state
    rules = session.rules
    if state.day + plan.duration_days > rules.limits.challenge_max_days:
        return False
    required_quantity = int(
        state.player.truck_total_capacity * config.vehicle_min_utilization_fraction
    )
    if plan.quantity < required_quantity:
        return False
    if state.player.cash < _vehicle_purchase_threshold(session, plan, config):
        return False
    remaining_days = rules.limits.challenge_max_days - state.day
    labor_cost = rules.vehicles.daily_labor_cost_per_extra_truck * remaining_days
    per_unit_gain = plan.expected_net_gain / plan.quantity
    marginal_gain = per_unit_gain * rules.vehicles.capacity_per_vehicle
    return marginal_gain >= money(labor_cost * config.vehicle_payback_multiplier)


def _vehicle_purchase_threshold(
    session: GameSession,
    plan: TradePlan,
    config: GreedyConfig,
) -> Decimal:
    """保留新增运力最低装载所需的营运现金，避免购车后无法经营。"""

    vehicle_price = session.rules.vehicles.purchase_price
    fill_cost = (
        plan.purchase_unit_price
        * session.rules.vehicles.capacity_per_vehicle
        * config.vehicle_min_fill_fraction
    )
    return money(vehicle_price + fill_cost)


def _travel_estimate(
    session: GameSession,
    destination: str,
    *,
    origin: str | None = None,
    mode: TransportMode | None = None,
) -> tuple[Decimal, int, TransportMode | None, int] | None:
    """返回确定性路线估算；不可达路线不参与贪心候选。"""

    if origin is None:
        origin = session.state.player.location
    if destination == origin:
        return Decimal("0"), 0, None, 0
    if origin == session.state.player.location:
        state = session.state
    else:
        from dataclasses import replace

        state = replace(
            session.state,
            player=replace(session.state.player, location=origin),
        )
    modes = (mode,) if mode is not None else tuple(TransportMode)
    candidates = []
    for candidate_mode in modes:
        try:
            estimate = estimate_travel(
                session.catalog,
                session.rules,
                state,
                Travel(destination=destination, mode=candidate_mode),
            )
        except RouteNotFound:
            continue
        candidates.append(
            (estimate.standard_cost, estimate.standard_days, candidate_mode, estimate.distance_km)
        )
    return min(candidates, key=lambda candidate: (candidate[1], candidate[0]), default=None)


def _expected_retained_quantity(product: Product, days: int, distance_km: int) -> float:
    """按运输规则的期望损耗估算保留比例，不消耗游戏随机数。"""

    if product.category is ProductCategory.ELECTRONICS:
        multiplier = 1.0 + distance_km / 2000
    elif product.category is ProductCategory.PERISHABLE:
        multiplier = 1.0 + days / 5 + float(product.perishable_aging_strength) * (days / 10) ** 2
    else:
        multiplier = 1.0 + distance_km / 6000
    loss_rate = min(1.0, float(product.transport_loss_rate) * multiplier)
    return 1.0 - loss_rate


__all__ = [
    "GreedyConfig",
    "GreedyEpisode",
    "GreedyEvaluation",
    "GreedyPolicy",
    "GreedyStep",
    "evaluate_greedy",
    "play_greedy",
]
