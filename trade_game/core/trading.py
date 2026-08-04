"""商品买卖规则；所有函数返回新的不可变状态快照。"""

from __future__ import annotations

from dataclasses import replace

from .catalog import Catalog
from .commands import Buy, Sell
from .inventory import add_cargo, cargo_quantity, free_capacity, remove_cargo_fifo
from .models import CargoLot, GameState
from .price_functions import purchase_unit_price, sale_unit_price, trade_total
from .results import CommandRejection, CommandResult, GameEvent, RejectionCode
from .rules import GameRules
from .transport import remote_sale_distance_multiplier


def buy(catalog: Catalog, rules: GameRules, state: GameState, command: Buy) -> CommandResult:
    """执行采购并占用一个经营日；失败时返回原状态且不产生副作用。"""

    if command.product_id not in catalog.products:
        return _reject(command, state, RejectionCode.UNKNOWN_ENTITY, "商品不存在")
    product = catalog.product(command.product_id)
    city_name = state.player.location
    if city_name not in product.origins:
        return _reject(command, state, RejectionCode.NOT_ALLOWED, "当前城市不是该商品产地")
    if command.quantity > free_capacity(state.player.cargo_lots, state.player.truck_total_capacity):
        return _reject(command, state, RejectionCode.INSUFFICIENT_CAPACITY, "货车剩余容量不足")

    unit_price = purchase_unit_price(catalog, rules, state, command.product_id, city_name)
    total = trade_total(unit_price, command.quantity)
    if state.player.cash < total:
        return _reject(command, state, RejectionCode.INSUFFICIENT_CASH, "现金不足")

    cargo_lots = add_cargo(
        state.player.cargo_lots,
        CargoLot(
            product_id=command.product_id,
            quantity=command.quantity,
            origin_city=city_name,
            shelf_life_remaining_days=product.perishable_shelf_life_days,
        ),
    )
    player = replace(state.player, cash=state.player.cash - total, cargo_lots=cargo_lots)
    next_state = replace(state, player=player, day=state.day + 1)
    return CommandResult.succeed(
        command,
        next_state,
        GameEvent(
            "goods_bought",
            {
                "product_id": command.product_id,
                "quantity": command.quantity,
                "unit_price": unit_price,
                "total": total,
            },
        ),
    )


def sell(catalog: Catalog, rules: GameRules, state: GameState, command: Sell) -> CommandResult:
    """执行精确数量的售卖并占用一个经营日；库存不足时拒绝而非部分成交。"""

    if command.product_id not in catalog.products:
        return _reject(command, state, RejectionCode.UNKNOWN_ENTITY, "商品不存在")
    if cargo_quantity(state.player.cargo_lots, command.product_id) < command.quantity:
        return _reject(command, state, RejectionCode.NOT_ALLOWED, "持有商品数量不足")

    city_name = state.player.location
    unit_price = sale_unit_price(
        catalog,
        rules,
        state,
        command.product_id,
        city_name,
        remote_distance_multiplier=remote_sale_distance_multiplier(
            catalog, rules, catalog.product(command.product_id), city_name
        ),
    )
    total = trade_total(unit_price, command.quantity)
    cargo_lots, _removed_lots = remove_cargo_fifo(
        state.player.cargo_lots, command.product_id, command.quantity
    )
    player = replace(state.player, cash=state.player.cash + total, cargo_lots=cargo_lots)
    next_state = replace(state, player=player, day=state.day + 1)
    return CommandResult.succeed(
        command,
        next_state,
        GameEvent(
            "goods_sold",
            {
                "product_id": command.product_id,
                "quantity": command.quantity,
                "unit_price": unit_price,
                "total": total,
            },
        ),
    )


def _reject(command: Buy | Sell, state: GameState, code: RejectionCode, message: str) -> CommandResult:
    return CommandResult.reject(command, state, CommandRejection(code, message))
