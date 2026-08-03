"""货车维修和购买规则。"""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal, ROUND_HALF_UP

from .commands import BuyTruck, RepairTruck
from .models import GameState
from .price_functions import money
from .results import CommandRejection, CommandResult, GameEvent, RejectionCode


# 这些平衡参数将在下一阶段迁入 rules.toml。
TRUCK_PURCHASE_PRICE = Decimal("40000")
TRUCK_CAPACITY_PER_VEHICLE = 100
TRUCK_REPAIR_COST_PER_PERCENT = Decimal("100")
TRUCK_REPAIR_DAYS = 7


def repair_truck(state: GameState, command: RepairTruck) -> CommandResult:
    """将货车耐久度维修到 100%，并记录维修耗时。"""

    damaged_percent = int((Decimal("100") - state.player.truck_durability).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
    if damaged_percent <= 0:
        return _reject(command, state, RejectionCode.NOT_ALLOWED, "货车无需维修")
    cost = money(TRUCK_REPAIR_COST_PER_PERCENT * damaged_percent * state.player.truck_count)
    if state.player.cash < cost:
        return _reject(command, state, RejectionCode.INSUFFICIENT_CASH, "现金不足以维修货车")
    player = replace(
        state.player,
        cash=state.player.cash - cost,
        truck_durability=Decimal("100"),
    )
    next_state = replace(state, player=player, day=state.day + TRUCK_REPAIR_DAYS)
    return CommandResult.succeed(
        command,
        next_state,
        GameEvent(
            "truck_repaired",
            {"cost": cost, "days": TRUCK_REPAIR_DAYS, "truck_count": state.player.truck_count},
        ),
    )


def buy_truck(state: GameState, command: BuyTruck) -> CommandResult:
    """购买货车；购车即时生效且不推进游戏日期。"""

    cost = money(TRUCK_PURCHASE_PRICE * command.quantity)
    if state.player.cash < cost:
        return _reject(command, state, RejectionCode.INSUFFICIENT_CASH, "现金不足以购买货车")
    player = replace(
        state.player,
        cash=state.player.cash - cost,
        truck_count=state.player.truck_count + command.quantity,
        truck_total_capacity=state.player.truck_total_capacity + TRUCK_CAPACITY_PER_VEHICLE * command.quantity,
    )
    next_state = replace(state, player=player)
    return CommandResult.succeed(
        command,
        next_state,
        GameEvent(
            "trucks_bought",
            {
                "quantity": command.quantity,
                "cost": cost,
                "capacity_added": TRUCK_CAPACITY_PER_VEHICLE * command.quantity,
            },
        ),
    )


def _reject(command: RepairTruck | BuyTruck, state: GameState, code: RejectionCode, message: str) -> CommandResult:
    return CommandResult.reject(command, state, CommandRejection(code, message))
