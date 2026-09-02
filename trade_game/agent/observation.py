"""将游戏公开状态整理为无实体 ID 的事实矩阵。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal
from math import log, log1p

from trade_game.core import (
    GameSession,
    Product,
    RouteNotFound,
    Travel,
    TravelEstimate,
    available_credit,
    estimate_travel,
    free_capacity,
    purchase_cooldown_remaining,
    reference_sale_price_history,
    shortest_distance_any,
    total_debt,
)

from .actions import ActionVocabulary


GLOBAL_FEATURE_NAMES = (
    "day_progress",
    "cash_signed_log",
    "debt_log",
    "available_credit_log",
    "truck_count_log",
    "truck_total_capacity_log",
    "free_capacity_fraction",
    "truck_durability_fraction",
)
CITY_FEATURE_NAMES = ("has_bank",)
PRODUCT_FEATURE_NAMES = (
    "base_purchase_price_log",
    "profit_margin_rate",
    "is_perishable",
    "shelf_life_fraction",
    "perishable_aging_strength_fraction",
    "transport_loss_rate",
)
MARKET_STATIC_FEATURE_NAMES = (
    "is_purchase_origin",
    "purchase_cooldown_fraction",
)
ROUTE_FEATURE_NAMES = (
    "standard_fare_log",
    "standard_travel_time_fraction",
    "express_fare_log",
    "express_travel_time_fraction",
    "truck_durability_loss_fraction",
    "economic_distance_fraction",
)
CARGO_LOT_FEATURE_NAMES = (
    "quantity_fraction",
    "remaining_shelf_life_fraction",
    "fifo_rank_fraction",
)


@dataclass(frozen=True, slots=True)
class ObservationConfig:
    """市场历史采样配置。

    空元组表示读取规则配置允许保存的全部历史点。训练配置可显式传入
    任意非负偏移量，例如 ``(6, 4, 2, 0)``；实现本身不依赖具体天数。
    """

    market_history_offsets: tuple[int, ...] = ()

    def resolve_market_history_offsets(self, history_days: int) -> tuple[int, ...]:
        if history_days <= 0:
            raise ValueError("市场历史长度必须为正")
        offsets = self.market_history_offsets or tuple(range(history_days - 1, -1, -1))
        if any(offset < 0 or offset >= history_days for offset in offsets):
            raise ValueError("市场历史采样偏移超出规则保存范围")
        if len(set(offsets)) != len(offsets):
            raise ValueError("市场历史采样偏移不能重复")
        return offsets


@dataclass(frozen=True, slots=True)
class GlobalObservation:
    """与城市和商品轴无关的经营事实。"""

    features: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class AgentObservation:
    """供学习层张量化的完整事实快照。

    城市和商品仅作为矩阵轴及动作解码顺序。没有城市、商品、区域、类别或
    运输方式 ID 被作为神经网络特征输入。
    """

    market_history_offsets: tuple[int, ...]
    global_state: GlobalObservation
    current_city_flags: tuple[float, ...]
    city_features: tuple[tuple[float, ...], ...]
    product_features: tuple[tuple[float, ...], ...]
    market_features: tuple[tuple[tuple[float, ...], ...], ...]
    route_available: tuple[tuple[tuple[bool, ...], ...], ...]
    route_features: tuple[tuple[tuple[tuple[float, ...], ...], ...], ...]
    cargo_lot_table: tuple[
        tuple[tuple[tuple[tuple[float, ...], ...], ...], ...], ...
    ]


def market_feature_names(history_offsets: tuple[int, ...]) -> tuple[str, ...]:
    """返回与当前历史采样配置严格对齐的市场矩阵字段名。"""

    return tuple(f"sale_price_offset_{offset}_log" for offset in history_offsets) + (
        *MARKET_STATIC_FEATURE_NAMES,
    )


def build_observation(
    session: GameSession,
    vocabulary: ActionVocabulary,
    *,
    config: ObservationConfig | None = None,
) -> AgentObservation:
    """从会话构造一次只含公开事实的矩阵观测。"""

    catalog = session.catalog
    rules = session.rules
    state = session.state
    observation_config = config or ObservationConfig()
    history_offsets = observation_config.resolve_market_history_offsets(
        rules.market.price_history_days
    )
    money_scale = max(rules.initial.initial_cash, Decimal("1"))
    day_span = max(1, rules.limits.challenge_max_days - rules.initial.initial_day)
    shelf_life_scale = max(
        1,
        max(
            (product.perishable_shelf_life_days or 0 for product in catalog.products.values()),
            default=0,
        ),
    )
    aging_strength_scale = max(
        Decimal("1"),
        max(
            (product.perishable_aging_strength or Decimal("0") for product in catalog.products.values()),
            default=Decimal("0"),
        ),
    )

    player = state.player
    elapsed_days = max(0, state.day - rules.initial.initial_day)
    global_state = GlobalObservation(
        features=(
            elapsed_days / day_span,
            _signed_money_log(player.cash, money_scale),
            _money_log(total_debt(state.loans), money_scale),
            _money_log(available_credit(catalog, rules, state), money_scale),
            log1p(player.truck_count),
            log1p(player.truck_total_capacity / rules.initial.initial_truck_capacity),
            free_capacity(player.cargo_lots, player.truck_total_capacity)
            / player.truck_total_capacity,
            float(player.truck_durability / rules.initial.initial_truck_durability),
        )
    )
    current_city_flags = tuple(
        float(city_name == player.location) for city_name in vocabulary.city_names
    )
    city_features = tuple(
        (float(catalog.city(city_name).has_bank),) for city_name in vocabulary.city_names
    )
    product_features = tuple(
        _product_features(
            catalog.product(product_id),
            money_scale=money_scale,
            shelf_life_scale=shelf_life_scale,
            aging_strength_scale=aging_strength_scale,
        )
        for product_id in vocabulary.product_ids
    )
    market_features = _market_features(session, vocabulary, history_offsets)
    route_available, route_features = _route_features(session, vocabulary, money_scale)
    cargo_lot_table = _cargo_lot_table(session, vocabulary)
    return AgentObservation(
        market_history_offsets=history_offsets,
        global_state=global_state,
        current_city_flags=current_city_flags,
        city_features=city_features,
        product_features=product_features,
        market_features=market_features,
        route_available=route_available,
        route_features=route_features,
        cargo_lot_table=cargo_lot_table,
    )


def _product_features(
    product: Product,
    *,
    money_scale: Decimal,
    shelf_life_scale: int,
    aging_strength_scale: Decimal,
) -> tuple[float, ...]:
    shelf_life = product.perishable_shelf_life_days
    aging_strength = product.perishable_aging_strength
    return (
        log(float(product.base_purchase_price / money_scale)),
        float(product.profit_margin_rate),
        float(shelf_life is not None),
        0.0 if shelf_life is None else shelf_life / shelf_life_scale,
        0.0 if aging_strength is None else float(aging_strength / aging_strength_scale),
        float(product.transport_loss_rate),
    )


def _market_features(
    session: GameSession,
    vocabulary: ActionVocabulary,
    history_offsets: tuple[int, ...],
) -> tuple[tuple[tuple[float, ...], ...], ...]:
    catalog = session.catalog
    rules = session.rules
    state = session.state
    rows: list[tuple[tuple[float, ...], ...]] = []
    cooldown_scale = max(1, rules.market.purchase_cooldown_days)
    for city_name in vocabulary.city_names:
        row: list[tuple[float, ...]] = []
        for product_id in vocabulary.product_ids:
            product = catalog.product(product_id)
            prices = reference_sale_price_history(catalog, rules, state, product_id, city_name)
            history = _sample_market_history(prices, product.base_purchase_price, history_offsets)
            row.append(
                (
                    *history,
                    float(city_name in product.origins),
                    purchase_cooldown_remaining(state, city_name, product_id) / cooldown_scale,
                )
            )
        rows.append(tuple(row))
    return tuple(rows)


def _route_features(
    session: GameSession,
    vocabulary: ActionVocabulary,
    money_scale: Decimal,
) -> tuple[
    tuple[tuple[tuple[bool, ...], ...], ...],
    tuple[tuple[tuple[tuple[float, ...], ...], ...], ...],
]:
    """一次构造所有城市对的公开运输报价，不预计算商品收益。"""

    state = session.state
    economic_distances = tuple(
        tuple(
            shortest_distance_any(session.catalog, origin_city, destination_city)
            for destination_city in vocabulary.city_names
        )
        for origin_city in vocabulary.city_names
    )
    economic_distance_scale = max(1, max(max(row) for row in economic_distances))
    estimates: list[list[list[TravelEstimate | None]]] = []
    day_scale = 1
    for origin_city in vocabulary.city_names:
        origin_state = replace(state, player=replace(state.player, location=origin_city))
        destinations: list[list[TravelEstimate | None]] = []
        for destination_city in vocabulary.city_names:
            modes: list[TravelEstimate | None] = []
            for mode in vocabulary.transport_modes:
                try:
                    estimate = estimate_travel(
                        session.catalog,
                        session.rules,
                        origin_state,
                        Travel(destination=destination_city, mode=mode),
                    )
                except RouteNotFound:
                    modes.append(None)
                    continue
                day_scale = max(day_scale, estimate.standard_days, estimate.fast_days)
                modes.append(estimate)
            destinations.append(modes)
        estimates.append(destinations)

    durability_scale = max(Decimal("1"), session.rules.initial.initial_truck_durability)
    available_rows: list[tuple[tuple[bool, ...], ...]] = []
    feature_rows: list[tuple[tuple[tuple[float, ...], ...], ...]] = []
    for origin_index, destinations in enumerate(estimates):
        available_destinations: list[tuple[bool, ...]] = []
        feature_destinations: list[tuple[tuple[float, ...], ...]] = []
        for destination_index, modes in enumerate(destinations):
            available_modes: list[bool] = []
            feature_modes: list[tuple[float, ...]] = []
            for estimate in modes:
                if estimate is None:
                    available_modes.append(False)
                    feature_modes.append((0.0,) * len(ROUTE_FEATURE_NAMES))
                    continue
                available_modes.append(True)
                feature_modes.append(
                    (
                        _money_log(estimate.standard_cost, money_scale),
                        estimate.standard_days / day_scale,
                        _money_log(estimate.fast_cost, money_scale),
                        estimate.fast_days / day_scale,
                        float(estimate.truck_durability_loss / durability_scale),
                        economic_distances[origin_index][destination_index]
                        / economic_distance_scale,
                    )
                )
            available_destinations.append(tuple(available_modes))
            feature_destinations.append(tuple(feature_modes))
        available_rows.append(tuple(available_destinations))
        feature_rows.append(tuple(feature_destinations))
    return tuple(available_rows), tuple(feature_rows)


def _cargo_lot_table(
    session: GameSession,
    vocabulary: ActionVocabulary,
) -> tuple[tuple[tuple[tuple[tuple[float, ...], ...], ...], ...], ...]:
    """按商品、真实产地和批次保存库存；FIFO 次序属于真实状态。"""

    player = session.state.player
    product_lots = {
        product_id: tuple(lot for lot in player.cargo_lots if lot.product_id == product_id)
        for product_id in vocabulary.product_ids
    }
    table: list[list[list[tuple[float, ...]]]] = [
        [[] for _ in vocabulary.city_names] for _ in vocabulary.product_ids
    ]
    for product_index, product_id in enumerate(vocabulary.product_ids):
        product = session.catalog.product(product_id)
        lots = product_lots[product_id]
        rank_scale = max(1, len(lots) - 1)
        for fifo_rank, lot in enumerate(lots):
            origin_index = vocabulary.city_index(lot.origin_city)
            shelf_life_fraction = (
                0.0
                if lot.shelf_life_remaining_days is None
                else lot.shelf_life_remaining_days / product.perishable_shelf_life_days
            )
            table[product_index][origin_index].append(
                (
                    lot.quantity / player.truck_total_capacity,
                    shelf_life_fraction,
                    fifo_rank / rank_scale,
                )
            )
    return tuple(
        tuple(tuple(tuple(lots) for lots in origins) for origins in products)
        for products in table
    )


def _sample_market_history(
    prices: tuple[Decimal, ...],
    base_price: Decimal,
    offsets: tuple[int, ...],
) -> tuple[float, ...]:
    values: list[float] = []
    for offset in offsets:
        price = prices[-offset - 1] if len(prices) > offset else base_price
        values.append(log(float(price / base_price)))
    return tuple(values)


def _money_log(value: Decimal, scale: Decimal) -> float:
    return log1p(float(value / scale))


def _signed_money_log(value: Decimal, scale: Decimal) -> float:
    normalized = float(value / scale)
    return (1.0 if normalized >= 0 else -1.0) * log1p(abs(normalized))


__all__ = [
    "AgentObservation",
    "CARGO_LOT_FEATURE_NAMES",
    "CITY_FEATURE_NAMES",
    "GLOBAL_FEATURE_NAMES",
    "GlobalObservation",
    "MARKET_STATIC_FEATURE_NAMES",
    "ObservationConfig",
    "PRODUCT_FEATURE_NAMES",
    "ROUTE_FEATURE_NAMES",
    "build_observation",
    "market_feature_names",
]
