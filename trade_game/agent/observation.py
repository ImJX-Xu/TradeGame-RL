"""将游戏会话编码为不泄露市场内部状态的结构化智能体观测。"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from math import log, log1p

from trade_game.core import (
    GameSession,
    Product,
    ProductCategory,
    RouteNotFound,
    Travel,
    available_credit,
    daily_interest_charge,
    daily_labor_cost,
    estimate_travel,
    free_capacity,
    purchase_cooldown_remaining,
    reference_sale_price_history,
    remote_sale_distance_premium,
    sale_unit_price,
    total_debt,
)

from .actions import ActionVocabulary


MARKET_HISTORY_OFFSETS = (6, 4, 2, 0)
GLOBAL_FEATURE_NAMES = (
    "day_progress",
    "remaining_day_fraction",
    "cash_signed_log",
    "debt_log",
    "credit_log",
    "truck_count_log",
    "capacity_log",
    "free_capacity_fraction",
    "truck_durability_fraction",
    "daily_interest_log",
    "daily_labor_log",
)
CITY_FEATURE_NAMES = (
    "is_current_city",
    "has_bank",
    "has_port",
    "is_high_consumption",
)
PRODUCT_FEATURE_NAMES = (
    "base_price_log",
    "profit_margin_rate",
    "shelf_life_fraction",
    "has_shelf_life",
    "transport_loss_rate",
)
ROUTE_FEATURE_NAMES = (
    "distance_log",
    "standard_cost_log",
    "standard_days_fraction",
    "fast_cost_log",
    "fast_days_fraction",
    "truck_durability_loss_fraction",
)
CARGO_FEATURE_NAMES = (
    "quantity_fraction",
    "age_fraction",
    "shelf_life_fraction",
    "has_shelf_life",
    "fifo_fraction",
    "sale_price_log",
)
PRODUCT_CATEGORY_NAMES = tuple(category.value for category in ProductCategory)


@dataclass(frozen=True, slots=True)
class GlobalObservation:
    """与具体城市、商品无关的经营状态。"""

    current_city_index: int
    features: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class CityObservation:
    """城市 ID、区域 ID 与可公开查阅的固定城市属性。"""

    city_index: int
    region_index: int
    features: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class ProductObservation:
    """商品 ID、类别 ID 与公开的物流和定价基础属性。"""

    product_index: int
    category_index: int
    features: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class MarketQuoteObservation:
    """一个城市商品对的四日行情与采购恢复状态。"""

    city_index: int
    product_index: int
    sale_log_history: tuple[float, ...]
    can_purchase: bool
    purchase_cooldown_fraction: float


@dataclass(frozen=True, slots=True)
class RouteObservation:
    """当前城市到一个目的城市的单种运输方式估算。"""

    destination_city_index: int
    transport_index: int
    available: bool
    features: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class CargoLotObservation:
    """保留真实产地、保质期和 FIFO 顺序的货物批次实体。"""

    product_index: int
    origin_city_index: int
    features: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class AgentObservation:
    """供后续 PyTorch 批处理器转换为张量的完整状态快照。"""

    region_names: tuple[str, ...]
    global_state: GlobalObservation
    cities: tuple[CityObservation, ...]
    products: tuple[ProductObservation, ...]
    market_quotes: tuple[tuple[MarketQuoteObservation, ...], ...]
    market_history_valid: tuple[bool, ...]
    routes: tuple[tuple[RouteObservation, ...], ...]
    cargo_lots: tuple[CargoLotObservation, ...]


def build_observation(session: GameSession, vocabulary: ActionVocabulary) -> AgentObservation:
    """从当前会话生成一次纯数据观测，不读取市场趋势或事件内部变量。"""

    catalog = session.catalog
    rules = session.rules
    state = session.state
    money_scale = _money_scale(rules.initial.initial_cash)
    horizon = rules.limits.challenge_max_days
    region_names = tuple(sorted({city.region for city in catalog.cities.values()}))
    region_indices = {name: index for index, name in enumerate(region_names)}

    cities = tuple(
        CityObservation(
            city_index=city_index,
            region_index=region_indices[catalog.city(city_name).region],
            features=(
                float(city_name == state.player.location),
                float(catalog.city(city_name).has_bank),
                float(catalog.city(city_name).has_port),
                float(catalog.city(city_name).is_high_consumption),
            ),
        )
        for city_index, city_name in enumerate(vocabulary.city_names)
    )
    products = tuple(
        _product_observation(catalog.product(product_id), product_index, money_scale, horizon)
        for product_index, product_id in enumerate(vocabulary.product_ids)
    )
    market_quotes, history_valid = _market_quotes(session, vocabulary)
    routes = _route_observations(session, vocabulary, money_scale, horizon)
    cargo_lots = _cargo_observations(session, vocabulary, horizon)

    player = state.player
    episode_days = max(1, horizon - rules.initial.initial_day)
    elapsed_days = max(0, state.day - rules.initial.initial_day)
    global_state = GlobalObservation(
        current_city_index=vocabulary.city_index(player.location),
        features=(
            elapsed_days / episode_days,
            max(0, horizon - state.day) / episode_days,
            _signed_money_log(player.cash, money_scale),
            _money_log(total_debt(state.loans), money_scale),
            _money_log(available_credit(catalog, rules, state), money_scale),
            log1p(player.truck_count),
            log1p(player.truck_total_capacity / rules.initial.initial_truck_capacity),
            free_capacity(player.cargo_lots, player.truck_total_capacity) / player.truck_total_capacity,
            float(player.truck_durability / Decimal("100")),
            _money_log(daily_interest_charge(state.loans, rules.finance.daily_interest_rate), money_scale),
            _money_log(daily_labor_cost(rules, state), money_scale),
        ),
    )
    return AgentObservation(
        region_names=region_names,
        global_state=global_state,
        cities=cities,
        products=products,
        market_quotes=market_quotes,
        market_history_valid=history_valid,
        routes=routes,
        cargo_lots=cargo_lots,
    )


def _product_observation(
    product: Product, product_index: int, money_scale: Decimal, horizon: int
) -> ProductObservation:
    """将目录中的公开商品规则压缩为类别索引和五个连续特征。"""

    # Product 的类型由已校验目录保证；这里不为观测层重复目录校验。
    category = product.category
    shelf_life = product.perishable_shelf_life_days
    return ProductObservation(
        product_index=product_index,
        category_index=PRODUCT_CATEGORY_NAMES.index(category.value),
        features=(
            _price_log(product.base_purchase_price, money_scale),
            float(product.profit_margin_rate),
            0.0 if shelf_life is None else shelf_life / horizon,
            float(shelf_life is not None),
            float(product.transport_loss_rate),
        ),
    )


def _market_quotes(
    session: GameSession, vocabulary: ActionVocabulary
) -> tuple[tuple[tuple[MarketQuoteObservation, ...], ...], tuple[bool, ...]]:
    catalog = session.catalog
    rows: list[tuple[MarketQuoteObservation, ...]] = []
    history_valid: tuple[bool, ...] | None = None
    for city_index, city_name in enumerate(vocabulary.city_names):
        row: list[MarketQuoteObservation] = []
        for product_index, product_id in enumerate(vocabulary.product_ids):
            product = catalog.product(product_id)
            prices = reference_sale_price_history(
                catalog, session.rules, session.state, product_id, city_name
            )
            assert len(prices) <= MARKET_HISTORY_OFFSETS[0] + 1
            sale_history, valid = _sample_market_history(prices, product.base_purchase_price)
            if history_valid is None:
                history_valid = valid
            else:
                assert valid == history_valid
            row.append(
                MarketQuoteObservation(
                    city_index=city_index,
                    product_index=product_index,
                    sale_log_history=sale_history,
                    can_purchase=(
                        city_name in product.origins
                        and purchase_cooldown_remaining(session.state, city_name, product_id) == 0
                    ),
                    purchase_cooldown_fraction=(
                        purchase_cooldown_remaining(session.state, city_name, product_id)
                        / max(1, session.rules.market.purchase_cooldown_days)
                    ),
                )
            )
        rows.append(tuple(row))
    assert history_valid is not None
    return tuple(rows), history_valid


def _route_observations(
    session: GameSession,
    vocabulary: ActionVocabulary,
    money_scale: Decimal,
    horizon: int,
) -> tuple[tuple[RouteObservation, ...], ...]:
    routes: list[tuple[RouteObservation, ...]] = []
    for city_index, city_name in enumerate(vocabulary.city_names):
        modes: list[RouteObservation] = []
        for transport_index, mode in enumerate(vocabulary.transport_modes):
            try:
                estimate = estimate_travel(
                    session.catalog,
                    session.rules,
                    session.state,
                    Travel(destination=city_name, mode=mode),
                )
            except RouteNotFound:
                modes.append(
                    RouteObservation(
                        destination_city_index=city_index,
                        transport_index=transport_index,
                        available=False,
                        features=(0.0,) * 6,
                    )
                )
                continue
            modes.append(
                RouteObservation(
                    destination_city_index=city_index,
                    transport_index=transport_index,
                    available=True,
                    features=(
                        log1p(estimate.distance_km / 1000),
                        _money_log(estimate.standard_cost, money_scale),
                        estimate.standard_days / horizon,
                        _money_log(estimate.fast_cost, money_scale),
                        estimate.fast_days / horizon,
                        float(estimate.truck_durability_loss / Decimal("100")),
                    ),
                )
            )
        routes.append(tuple(modes))
    return tuple(routes)


def _cargo_observations(
    session: GameSession, vocabulary: ActionVocabulary, horizon: int
) -> tuple[CargoLotObservation, ...]:
    state = session.state
    player = state.player
    totals = {
        product_id: sum(lot.quantity for lot in player.cargo_lots if lot.product_id == product_id)
        for product_id in vocabulary.product_ids
    }
    quantities_before = {product_id: 0 for product_id in vocabulary.product_ids}
    encoded: list[CargoLotObservation] = []
    for lot in player.cargo_lots:
        product = session.catalog.product(lot.product_id)
        product_total = totals[lot.product_id]
        sale_price = sale_unit_price(
            session.catalog,
            session.rules,
            state,
            lot.product_id,
            player.location,
            origin_city=lot.origin_city,
            remote_distance_premium=remote_sale_distance_premium(
                session.catalog, session.rules, lot.origin_city, player.location
            ),
        )
        shelf_life = product.perishable_shelf_life_days
        encoded.append(
            CargoLotObservation(
                product_index=vocabulary.product_index(lot.product_id),
                origin_city_index=vocabulary.city_index(lot.origin_city),
                features=(
                    lot.quantity / player.truck_total_capacity,
                    lot.age_days / horizon,
                    0.0 if lot.shelf_life_remaining_days is None else lot.shelf_life_remaining_days / shelf_life,
                    float(lot.shelf_life_remaining_days is not None),
                    quantities_before[lot.product_id] / product_total,
                    _price_log(sale_price, product.base_purchase_price),
                ),
            )
        )
        quantities_before[lot.product_id] += lot.quantity
    return tuple(encoded)


def _sample_market_history(
    prices: tuple[Decimal, ...], base_price: Decimal
) -> tuple[tuple[float, ...], tuple[bool, ...]]:
    values: list[float] = []
    valid: list[bool] = []
    for offset in MARKET_HISTORY_OFFSETS:
        if len(prices) > offset:
            values.append(_price_log(prices[-offset - 1], base_price))
            valid.append(True)
        else:
            values.append(0.0)
            valid.append(False)
    return tuple(values), tuple(valid)


def _money_scale(initial_cash: Decimal) -> Decimal:
    return max(initial_cash, Decimal("1"))


def _money_log(value: Decimal, scale: Decimal) -> float:
    return log1p(float(value / scale))


def _signed_money_log(value: Decimal, scale: Decimal) -> float:
    normalized = float(value / scale)
    return (1.0 if normalized >= 0 else -1.0) * log1p(abs(normalized))


def _price_log(price: Decimal, base_price: Decimal) -> float:
    return log(float(price / base_price))


__all__ = [
    "AgentObservation",
    "CARGO_FEATURE_NAMES",
    "CargoLotObservation",
    "CITY_FEATURE_NAMES",
    "CityObservation",
    "GLOBAL_FEATURE_NAMES",
    "GlobalObservation",
    "MARKET_HISTORY_OFFSETS",
    "MarketQuoteObservation",
    "PRODUCT_CATEGORY_NAMES",
    "PRODUCT_FEATURE_NAMES",
    "ProductObservation",
    "ROUTE_FEATURE_NAMES",
    "RouteObservation",
    "build_observation",
]
