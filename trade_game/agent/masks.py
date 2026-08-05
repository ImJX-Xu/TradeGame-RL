"""基于核心规则生成智能体的合法动作掩码。"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from trade_game.core import (
    CommandType,
    GameSession,
    RepairTruck,
    Travel,
    available_credit,
    can_travel,
    cargo_quantity,
    maximum_purchase_quantity,
    maximum_truck_quantity,
    repair_truck,
    total_debt,
)

from .actions import (
    ActionVocabulary,
    QuantityBin,
    integer_quantity_from_bin,
    money_amount_from_bin,
)


_Mask = tuple[bool, ...]


@dataclass(frozen=True, slots=True)
class ActionMask:
    """一个游戏状态下的条件化合法动作集合。

    ``action`` 与 ``ActionVocabulary.action_types`` 对齐。商品、运输方式和数量掩码
    只有在对应命令已被选中时才读取；例如 ``buy_quantity[product_index]``。
    """

    action: _Mask
    buy_product: _Mask
    sell_product: _Mask
    buy_quantity: tuple[_Mask, ...]
    sell_quantity: tuple[_Mask, ...]
    travel_city: _Mask
    travel_transport: tuple[_Mask, ...]
    travel_fast: tuple[tuple[_Mask, ...], ...]
    borrow_quantity: _Mask
    repay_quantity: _Mask
    buy_truck_quantity: _Mask


def build_action_mask(session: GameSession, vocabulary: ActionVocabulary) -> ActionMask:
    """按当前会话和固定词表生成一次完整的合法动作快照。"""

    empty_quantity = _empty_mask(len(QuantityBin))
    empty_product = _empty_mask(len(vocabulary.product_ids))
    empty_city = _empty_mask(len(vocabulary.city_names))
    empty_transport = _empty_mask(len(vocabulary.transport_modes))
    empty_fast = (False, False)
    if session.state.outcome is not None:
        return ActionMask(
            action=_empty_mask(len(vocabulary.action_types)),
            buy_product=empty_product,
            sell_product=empty_product,
            buy_quantity=tuple(empty_quantity for _ in vocabulary.product_ids),
            sell_quantity=tuple(empty_quantity for _ in vocabulary.product_ids),
            travel_city=empty_city,
            travel_transport=tuple(empty_transport for _ in vocabulary.city_names),
            travel_fast=tuple(
                tuple(empty_fast for _ in vocabulary.transport_modes)
                for _ in vocabulary.city_names
            ),
            borrow_quantity=empty_quantity,
            repay_quantity=empty_quantity,
            buy_truck_quantity=empty_quantity,
        )

    state = session.state
    buy_quantity = tuple(
        _integer_quantity_mask(
            maximum_purchase_quantity(session.catalog, session.rules, state, product_id)
        )
        for product_id in vocabulary.product_ids
    )
    sell_quantity = tuple(
        _integer_quantity_mask(cargo_quantity(state.player.cargo_lots, product_id))
        for product_id in vocabulary.product_ids
    )
    buy_product = tuple(any(mask) for mask in buy_quantity)
    sell_product = tuple(any(mask) for mask in sell_quantity)
    travel_city, travel_transport, travel_fast = _travel_masks(session, vocabulary)

    has_bank = session.catalog.city(state.player.location).has_bank
    borrow_quantity = (
        _money_quantity_mask(available_credit(session.catalog, session.rules, state))
        if has_bank
        else empty_quantity
    )
    repay_quantity = (
        _money_quantity_mask(min(total_debt(state.loans), state.player.cash))
        if has_bank
        else empty_quantity
    )
    buy_truck_quantity = _integer_quantity_mask(maximum_truck_quantity(session.rules, state))

    active = {
        CommandType.BUY: any(buy_product),
        CommandType.SELL: any(sell_product),
        CommandType.TRAVEL: any(travel_city),
        CommandType.BORROW: any(borrow_quantity),
        CommandType.REPAY: any(repay_quantity),
        CommandType.REPAIR_TRUCK: repair_truck(session.rules, state, RepairTruck()).accepted,
        CommandType.BUY_TRUCK: any(buy_truck_quantity),
        CommandType.NEXT_DAY: True,
    }
    return ActionMask(
        action=tuple(active[command_type] for command_type in vocabulary.action_types),
        buy_product=buy_product,
        sell_product=sell_product,
        buy_quantity=buy_quantity,
        sell_quantity=sell_quantity,
        travel_city=travel_city,
        travel_transport=travel_transport,
        travel_fast=travel_fast,
        borrow_quantity=borrow_quantity,
        repay_quantity=repay_quantity,
        buy_truck_quantity=buy_truck_quantity,
    )


def _travel_masks(
    session: GameSession, vocabulary: ActionVocabulary
) -> tuple[_Mask, tuple[_Mask, ...], tuple[tuple[_Mask, ...], ...]]:
    city_mask: list[bool] = []
    transport_masks: list[_Mask] = []
    fast_masks: list[tuple[_Mask, ...]] = []
    for city_name in vocabulary.city_names:
        city_fast_masks: list[_Mask] = []
        city_transport_mask: list[bool] = []
        for mode in vocabulary.transport_modes:
            fast_mask = tuple(
                can_travel(
                    session.catalog,
                    session.rules,
                    session.state,
                    Travel(destination=city_name, mode=mode, fast=bool(fast_index)),
                )
                for fast_index in range(2)
            )
            city_fast_masks.append(fast_mask)
            city_transport_mask.append(any(fast_mask))
        transport_mask = tuple(city_transport_mask)
        fast_masks.append(tuple(city_fast_masks))
        transport_masks.append(transport_mask)
        city_mask.append(any(transport_mask))
    return tuple(city_mask), tuple(transport_masks), tuple(fast_masks)


def _integer_quantity_mask(maximum: int) -> _Mask:
    if maximum <= 0:
        return _empty_mask(len(QuantityBin))
    return _distinct_quantity_mask(
        tuple(integer_quantity_from_bin(quantity_bin, maximum) for quantity_bin in QuantityBin)
    )


def _money_quantity_mask(maximum: Decimal) -> _Mask:
    if maximum <= 0:
        return _empty_mask(len(QuantityBin))
    return _distinct_quantity_mask(
        tuple(money_amount_from_bin(quantity_bin, maximum) for quantity_bin in QuantityBin)
    )


def _distinct_quantity_mask(quantities: tuple[int | Decimal, ...]) -> _Mask:
    """每个实际数量只保留一个档位，优先单个单位和较高百分比。"""

    enabled = [False] * len(QuantityBin)
    seen: set[int | Decimal] = set()
    for quantity_bin in (QuantityBin.ONE, *reversed(tuple(QuantityBin)[1:])):
        quantity = quantities[int(quantity_bin)]
        if quantity > 0 and quantity not in seen:
            enabled[int(quantity_bin)] = True
            seen.add(quantity)
    return tuple(enabled)


def _empty_mask(size: int) -> _Mask:
    return (False,) * size


__all__ = ["ActionMask", "build_action_mask"]
