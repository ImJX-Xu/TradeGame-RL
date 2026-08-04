"""货车维修和购买规则。"""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal, ROUND_HALF_UP

from .commands import BuyTruck, RepairTruck
from .models import GameState
from .price_functions import money
from .results import CommandRejection, CommandResult, GameEvent, RejectionCode
from .rules import GameRules


def maximum_truck_quantity(rules: GameRules, state: GameState) -> int:
    """返回当前现金可购买的最大货车数量。"""

    return int(state.player.cash / rules.vehicles.purchase_price)


def repair_truck(rules: GameRules, state: GameState, command: RepairTruck) -> CommandResult:
    """将货车耐久度维修到 100%，并记录维修耗时。"""

    damaged_percent = int((Decimal("100") - state.player.truck_durability).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
    if damaged_percent <= 0:
        return _reject(command, state, RejectionCode.NOT_ALLOWED, "货车无需维修")
    cost = money(rules.vehicles.repair_cost_per_percent * damaged_percent * state.player.truck_count)
    if state.player.cash < cost:
        return _reject(command, state, RejectionCode.INSUFFICIENT_CASH, "现金不足以维修货车")
    player = replace(
        state.player,
        cash=state.player.cash - cost,
        truck_durability=Decimal("100"),
    )
    next_state = replace(state, player=player, day=state.day + rules.vehicles.repair_days)
    return CommandResult.succeed(
        command,
        next_state,
        GameEvent(
            "truck_repaired",
            {"cost": cost, "days": rules.vehicles.repair_days, "truck_count": state.player.truck_count},
        ),
    )


def buy_truck(rules: GameRules, state: GameState, command: BuyTruck) -> CommandResult:
    """购买货车并占用一个经营日。"""

    cost = money(rules.vehicles.purchase_price * command.quantity)
    if state.player.cash < cost:
        return _reject(command, state, RejectionCode.INSUFFICIENT_CASH, "现金不足以购买货车")
    player = replace(
        state.player,
        cash=state.player.cash - cost,
        truck_count=state.player.truck_count + command.quantity,
        truck_total_capacity=state.player.truck_total_capacity + rules.vehicles.capacity_per_vehicle * command.quantity,
    )
    next_state = replace(state, player=player, day=state.day + 1)
    return CommandResult.succeed(
        command,
        next_state,
        GameEvent(
            "trucks_bought",
            {
                "quantity": command.quantity,
                "cost": cost,
                "capacity_added": rules.vehicles.capacity_per_vehicle * command.quantity,
            },
        ),
    )


def _reject(command: RepairTruck | BuyTruck, state: GameState, code: RejectionCode, message: str) -> CommandResult:
    return CommandResult.reject(command, state, CommandRejection(code, message))
