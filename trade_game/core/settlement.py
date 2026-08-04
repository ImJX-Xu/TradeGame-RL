"""破产判定与挑战模式的终局资产结算。"""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal

from .catalog import Catalog
from .finance import available_credit, total_debt
from .inventory import cargo_quantity
from .models import GameEndReason, GameMode, GameOutcome, GameState
from .price_functions import money, sale_unit_price
from .rules import GameRules
from .transport import remote_sale_distance_premium


def settlement_assets(catalog: Catalog, rules: GameRules, state: GameState) -> Decimal:
    """按当前市场变现货物、扣除全部债务后的终局资产。"""

    cargo_value = sum(
        (
            sale_unit_price(
                catalog,
                rules,
                state,
                lot.product_id,
                state.player.location,
                origin_city=lot.origin_city,
                remote_distance_premium=remote_sale_distance_premium(
                    catalog, rules, lot.origin_city, state.player.location
                ),
            )
            * lot.quantity
            for lot in state.player.cargo_lots
        ),
        start=Decimal("0"),
    )
    additional_trucks = max(0, state.player.truck_count - rules.initial.initial_truck_count)
    truck_value = rules.settlement.additional_truck_residual_value * additional_trucks
    return money(state.player.cash + cargo_value + truck_value - total_debt(state.loans))


def is_bankrupt(catalog: Catalog, rules: GameRules, state: GameState) -> bool:
    """现金耗尽、无货物且无法再授信时宣告破产。"""

    return (
        state.player.cash <= 0
        and cargo_quantity(state.player.cargo_lots) == 0
        and available_credit(catalog, rules, state) == 0
    )


def conclude_if_needed(catalog: Catalog, rules: GameRules, state: GameState) -> GameState:
    """在破产或挑战时间上限到达时写入不可逆的终局结果。"""

    if state.outcome is not None:
        return state
    if is_bankrupt(catalog, rules, state):
        reason = GameEndReason.BANKRUPTCY
    elif state.mode is GameMode.CHALLENGE and state.day >= rules.limits.challenge_max_days:
        reason = GameEndReason.TIME_LIMIT
    else:
        return state
    return replace(state, outcome=GameOutcome(reason=reason, final_assets=settlement_assets(catalog, rules, state)))
