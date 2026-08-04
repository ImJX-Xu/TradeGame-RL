"""商品买卖规则；所有函数返回新的不可变状态快照。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal

from .catalog import Catalog
from .commands import Buy, Sell
from .inventory import add_cargo, cargo_quantity, free_capacity, remove_cargo_fifo
from .models import CargoLot, GameState
from .price_functions import money, purchase_unit_price, sale_unit_price, trade_total
from .results import CommandRejection, CommandResult, GameEvent, RejectionCode
from .rules import GameRules
from .transport import remote_sale_distance_premium


@dataclass(frozen=True, slots=True)
class SaleQuote:
    """按 FIFO 批次来源汇总出的实际出售报价。"""

    quantity: int
    total: Decimal
    average_unit_price: Decimal


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


def quote_sale(
    catalog: Catalog,
    rules: GameRules,
    state: GameState,
    product_id: str,
    quantity: int,
) -> SaleQuote:
    """按即将被 FIFO 移除的批次，给出准确的出售总额与平均单价。"""

    _remaining_lots, removed_lots = remove_cargo_fifo(state.player.cargo_lots, product_id, quantity)
    total = _sale_total(catalog, rules, state, removed_lots)
    return SaleQuote(
        quantity=quantity,
        total=total,
        average_unit_price=money(total / Decimal(quantity)),
    )


def sell(catalog: Catalog, rules: GameRules, state: GameState, command: Sell) -> CommandResult:
    """执行精确数量的出售并占用一个经营日；库存不足时不部分成交。"""

    if command.product_id not in catalog.products:
        return _reject(command, state, RejectionCode.UNKNOWN_ENTITY, "商品不存在")
    if cargo_quantity(state.player.cargo_lots, command.product_id) < command.quantity:
        return _reject(command, state, RejectionCode.NOT_ALLOWED, "持有商品数量不足")

    cargo_lots, removed_lots = remove_cargo_fifo(
        state.player.cargo_lots, command.product_id, command.quantity
    )
    total = _sale_total(catalog, rules, state, removed_lots)
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
                "unit_price": money(total / Decimal(command.quantity)),
                "total": total,
            },
        ),
    )


def _sale_total(
    catalog: Catalog,
    rules: GameRules,
    state: GameState,
    lots: tuple[CargoLot, ...],
) -> Decimal:
    city_name = state.player.location
    total = Decimal("0")
    for lot in lots:
        unit_price = sale_unit_price(
            catalog,
            rules,
            state,
            lot.product_id,
            city_name,
            origin_city=lot.origin_city,
            remote_distance_premium=remote_sale_distance_premium(
                catalog,
                rules,
                lot.origin_city,
                city_name,
            ),
        )
        total += trade_total(unit_price, lot.quantity)
    return total


def _reject(command: Buy | Sell, state: GameState, code: RejectionCode, message: str) -> CommandResult:
    return CommandResult.reject(command, state, CommandRejection(code, message))
