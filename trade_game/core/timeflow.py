"""逐日结算：贷款利息、车辆成本、易腐货物和市场价格。"""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
from random import Random
from types import MappingProxyType

from .catalog import Catalog
from .finance import accrue_daily_interest
from .models import CargoLot, GameState, MarketState
from .price_functions import money
from .results import GameEvent
from .rules import GameRules


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
    for _ in range(days):
        loans, interest = accrue_daily_interest(current.loans, rules.finance.daily_interest_rate)
        cargo_lots, expired = _age_cargo(current.player.cargo_lots)
        labor_cost = money(
            rules.vehicles.daily_labor_cost_per_extra_truck * max(0, current.player.truck_count - 1)
        )
        player = replace(
            current.player,
            cash=current.player.cash - labor_cost,
            cargo_lots=cargo_lots,
        )
        current = replace(
            current,
            player=player,
            loans=loans,
            market=_refresh_market(catalog, rules, current.market, rng),
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


def _refresh_market(catalog: Catalog, rules: GameRules, market: MarketState, rng: Random) -> MarketState:
    refreshed: dict[tuple[str, str], Decimal] = {}
    for city_name, city in catalog.cities.items():
        for product_id, product in catalog.products.items():
            alpha = product.lambda_alpha * rules.market.lambda_alpha_adjustment
            sigma = product.lambda_sigma * rules.market.lambda_sigma_adjustment
            lower = product.lambda_min
            upper = product.lambda_max
            if city.is_high_consumption:
                alpha *= rules.market.high_consumption_alpha_multiplier
                sigma *= rules.market.high_consumption_sigma_multiplier
                lower *= rules.market.high_consumption_range_multiplier
                upper *= rules.market.high_consumption_range_multiplier
            previous = market.current_lambdas[(city_name, product_id)]
            noise = Decimal(str(rng.gauss(0.0, float(sigma))))
            refreshed[(city_name, product_id)] = min(upper, max(lower, alpha * previous + noise))
    history = {
        key: (*market.lambda_history[key], value)[-PRICE_HISTORY_DAYS:]
        for key, value in refreshed.items()
    }
    return MarketState(
        current_lambdas=MappingProxyType(refreshed),
        previous_lambdas=market.current_lambdas,
        lambda_history=MappingProxyType(history),
    )
