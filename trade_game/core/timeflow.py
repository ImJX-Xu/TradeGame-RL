"""逐日结算贷款、车辆成本、易腐货物与长期市场行情。"""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
from random import Random
from types import MappingProxyType

from .catalog import Catalog
from .finance import accrue_daily_interest
from .models import (
    CargoLot,
    GameState,
    MarketBulletin,
    MarketEvent,
    MarketEventKind,
    MarketMessage,
    MarketState,
    Product,
    SpecialtyScope,
)
from .price_functions import money
from .results import GameEvent
from .rules import GameRules
from .vehicles import daily_labor_cost


PRICE_HISTORY_DAYS = 7


def settle_elapsed_days(
    catalog: Catalog,
    rules: GameRules,
    state: GameState,
    rng: Random,
    days: int,
) -> tuple[GameState, tuple[GameEvent, ...]]:
    """对已推进的天数逐日结算，并聚合为一条稳定的日结事件。"""

    if days <= 0:
        raise ValueError("结算天数必须大于 0")

    current = state
    interest_total = Decimal("0")
    labor_total = Decimal("0")
    expired_total = 0
    for offset in range(days):
        loans, interest = accrue_daily_interest(current.loans, rules.finance.daily_interest_rate)
        cargo_lots, expired = _age_cargo(current.player.cargo_lots)
        labor_cost = daily_labor_cost(rules, current)
        player = replace(
            current.player,
            cash=current.player.cash - labor_cost,
            cargo_lots=cargo_lots,
        )
        settlement_day = state.day - days + offset + 1
        current = replace(
            current,
            player=player,
            loans=loans,
            market=_refresh_market(
                catalog,
                rules,
                current.market,
                rng,
                settlement_day=settlement_day,
            ),
        )
        interest_total += interest
        labor_total += labor_cost
        expired_total += expired

    event = GameEvent(
        "days_settled",
        {
            "days": days,
            "interest_accrued": money(interest_total),
            "labor_cost": money(labor_total),
            "expired_cargo": expired_total,
        },
    )
    return current, (event,)


def market_messages(state: GameState, city_name: str) -> tuple[MarketMessage, ...]:
    """返回城市内可公开观察到的市场讯息，不暴露事件峰值等隐藏参数。"""

    messages = [
        MarketMessage(
            kind=event.kind,
            product_id=event.product_id,
            remaining_days=event.end_day - state.day + 1,
        )
        for event in state.market.active_events
        if city_name in event.cities and event.start_day <= state.day <= event.end_day
    ]
    return tuple(sorted(messages, key=lambda item: (item.kind.value, item.product_id)))


def market_bulletins(
    catalog: Catalog,
    rules: GameRules,
    state: GameState,
) -> tuple[MarketBulletin, ...]:
    """返回当前全局行情，按实际生效幅度、商品权重和覆盖范围排序。"""

    active_events = tuple(
        event
        for event in state.market.active_events
        if event.start_day <= state.day <= event.end_day
    )
    ranked_events = sorted(
        active_events,
        key=lambda event: (
            -_event_importance(catalog, rules, event, state.day),
            event.end_day - state.day,
            event.product_id,
            event.cities,
        ),
    )
    return tuple(
        MarketBulletin(
            kind=event.kind,
            product_id=event.product_id,
            cities=event.cities,
            remaining_days=event.end_day - state.day + 1,
        )
        for event in ranked_events
    )


def _age_cargo(lots: tuple[CargoLot, ...]) -> tuple[tuple[CargoLot, ...], int]:
    retained: list[CargoLot] = []
    expired = 0
    for lot in lots:
        if lot.shelf_life_remaining_days is None:
            retained.append(replace(lot, age_days=lot.age_days + 1))
            continue
        remaining_days = lot.shelf_life_remaining_days - 1
        if remaining_days == 0:
            expired += lot.quantity
            continue
        retained.append(
            replace(lot, shelf_life_remaining_days=remaining_days, age_days=lot.age_days + 1)
        )
    return tuple(retained), expired


def _refresh_market(
    catalog: Catalog,
    rules: GameRules,
    market: MarketState,
    rng: Random,
    *,
    settlement_day: int,
) -> MarketState:
    active_events = tuple(event for event in market.active_events if event.end_day >= settlement_day)
    if (
        len(active_events) < rules.market.max_active_events
        and rng.random() < float(rules.market.event_spawn_probability)
    ):
        event = _create_market_event(catalog, rules, active_events, rng, settlement_day)
        if event is not None:
            active_events = (*active_events, event)

    product_trends: dict[str, Decimal] = {}
    for product_id, product in catalog.products.items():
        previous_trend = market.product_trends[product_id]
        noise = Decimal(str(rng.gauss(0.0, float(product.trend_sigma))))
        trend = product.trend_persistence * previous_trend + noise
        product_trends[product_id] = min(
            product.price_adjustment_max * rules.market.trend_range_share,
            max(product.price_adjustment_min * rules.market.trend_range_share, trend),
        )

    local_spreads: dict[tuple[str, str], Decimal] = {}
    refreshed: dict[tuple[str, str], Decimal] = {}
    for city_name in catalog.cities:
        for product_id, product in catalog.products.items():
            key = (city_name, product_id)
            previous_spread = market.local_spreads[key]
            local_noise = Decimal(str(rng.gauss(0.0, float(product.local_spread_sigma))))
            local_spread = rules.market.local_spread_persistence * previous_spread + local_noise
            local_spread = min(product.local_spread_max, max(-product.local_spread_max, local_spread))
            event_effect = sum(
                (
                    _event_adjustment(event, settlement_day, rules.market.event_ramp_days)
                    for event in active_events
                    if event.product_id == product_id and city_name in event.cities
                ),
                start=Decimal("0"),
            )
            local_spreads[key] = local_spread
            refreshed[key] = min(
                product.price_adjustment_max,
                max(
                    product.price_adjustment_min,
                    product_trends[product_id] + local_spread + event_effect,
                ),
            )
    history = {
        key: (*market.price_adjustment_history[key], value)[-PRICE_HISTORY_DAYS:]
        for key, value in refreshed.items()
    }
    return MarketState(
        current_price_adjustments=MappingProxyType(refreshed),
        product_trends=MappingProxyType(product_trends),
        local_spreads=MappingProxyType(local_spreads),
        price_adjustment_history=MappingProxyType(history),
        active_events=active_events,
    )


def _create_market_event(
    catalog: Catalog,
    rules: GameRules,
    active_events: tuple[MarketEvent, ...],
    rng: Random,
    start_day: int,
) -> MarketEvent | None:
    kind = (
        MarketEventKind.SHORTAGE
        if rng.random() < float(rules.market.shortage_probability)
        else MarketEventKind.SURPLUS
    )
    candidates = [
        product
        for product in catalog.products.values()
        if _has_event_city(catalog, product, kind, active_events)
    ]
    if not candidates:
        return None
    product = _weighted_product_choice(candidates, rng)
    cities = _select_event_cities(catalog, rules, product, kind, active_events, rng)
    assert cities is not None
    duration = rng.randint(product.event_duration_min_days, product.event_duration_max_days)
    amplitude = product.event_amplitude_min + (
        product.event_amplitude_max - product.event_amplitude_min
    ) * Decimal(str(rng.random()))
    peak_adjustment = -amplitude if kind is MarketEventKind.SURPLUS else amplitude
    return MarketEvent(
        kind=kind,
        product_id=product.id,
        cities=cities,
        start_day=start_day,
        end_day=start_day + duration - 1,
        peak_adjustment=peak_adjustment,
    )


def _has_event_city(
    catalog: Catalog,
    product: Product,
    kind: MarketEventKind,
    active_events: tuple[MarketEvent, ...],
) -> bool:
    blocked = _blocked_cities(product.id, active_events)
    if kind is MarketEventKind.SURPLUS:
        return not (product.origins & blocked)
    return any(
        city_name not in product.origins
        and city_name not in blocked
        and city.market_roles & product.demand_roles
        for city_name, city in catalog.cities.items()
    )


def _select_event_cities(
    catalog: Catalog,
    rules: GameRules,
    product: Product,
    kind: MarketEventKind,
    active_events: tuple[MarketEvent, ...],
    rng: Random,
) -> tuple[str, ...] | None:
    blocked = _blocked_cities(product.id, active_events)
    if kind is MarketEventKind.SURPLUS:
        cities = tuple(sorted(product.origins))
        return None if any(city_name in blocked for city_name in cities) else cities

    cities = [
        city_name
        for city_name, city in catalog.cities.items()
        if city_name not in product.origins
        and city_name not in blocked
        and city.market_roles & product.demand_roles
    ]
    if not cities:
        return None
    primary_city = rng.choice(cities)
    if (
        product.specialty_scope is not SpecialtyScope.REGION
        or rng.random() >= float(rules.market.regional_scope_probability)
    ):
        return (primary_city,)
    regional_cities = tuple(
        city_name
        for city_name in cities
        if catalog.city(city_name).region == catalog.city(primary_city).region
    )
    return regional_cities or (primary_city,)


def _blocked_cities(product_id: str, active_events: tuple[MarketEvent, ...]) -> set[str]:
    return {
        city_name
        for event in active_events
        if event.product_id == product_id
        for city_name in event.cities
    }


def _weighted_product_choice(products: list[Product], rng: Random) -> Product:
    total_weight = sum((product.event_weight for product in products), start=Decimal("0"))
    target = Decimal(str(rng.random())) * total_weight
    cumulative = Decimal("0")
    for product in products:
        cumulative += product.event_weight
        if target <= cumulative:
            return product
    return products[-1]


def _event_adjustment(event: MarketEvent, day: int, ramp_days: int) -> Decimal:
    elapsed_days = day - event.start_day
    remaining_days = event.end_day - day
    ramp = min(
        Decimal("1"),
        Decimal(elapsed_days + 1) / Decimal(ramp_days + 1),
        Decimal(remaining_days + 1) / Decimal(ramp_days + 1),
    )
    return event.peak_adjustment * ramp


def _event_importance(
    catalog: Catalog,
    rules: GameRules,
    event: MarketEvent,
    day: int,
) -> Decimal:
    coverage = Decimal("1") + Decimal(len(event.cities) - 1) * Decimal("0.15")
    current_effect = abs(_event_adjustment(event, day, rules.market.event_ramp_days))
    return current_effect * catalog.product(event.product_id).event_weight * coverage
