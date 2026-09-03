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
    buy: tuple[_Mask, ...]
    sell: tuple[_Mask, ...]
    travel: tuple[tuple[_Mask, ...], ...]
    borrow_quantity: _Mask
    repay_quantity: _Mask
    buy_truck_quantity: _Mask


def build_action_mask(session: GameSession, vocabulary: ActionVocabulary) -> ActionMask:
    """按当前会话和固定词表生成一次完整的合法动作快照。"""

    empty_quantity = _empty_mask(len(QuantityBin))
    empty_product_quantity = tuple(empty_quantity for _ in vocabulary.product_ids)
    empty_travel = tuple(
        tuple((False, False) for _ in vocabulary.transport_modes)
        for _ in vocabulary.city_names
    )
    if session.state.outcome is not None:
        return ActionMask(
            action=_empty_mask(len(vocabulary.action_types)),
            buy=empty_product_quantity,
            sell=empty_product_quantity,
            travel=empty_travel,
            borrow_quantity=empty_quantity,
            repay_quantity=empty_quantity,
            buy_truck_quantity=empty_quantity,
        )

    state = session.state
    buy = tuple(
        _integer_quantity_mask(
            maximum_purchase_quantity(session.catalog, session.rules, state, product_id)
        )
        for product_id in vocabulary.product_ids
    )
    sell = tuple(
        _integer_quantity_mask(cargo_quantity(state.player.cargo_lots, product_id))
        for product_id in vocabulary.product_ids
    )
    travel = _travel_mask(session, vocabulary)

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
        CommandType.BUY: any(any(mask) for mask in buy),
        CommandType.SELL: any(any(mask) for mask in sell),
        CommandType.TRAVEL: any(any(any(fast) for fast in transports) for transports in travel),
        CommandType.BORROW: any(borrow_quantity),
        CommandType.REPAY: any(repay_quantity),
        CommandType.REPAIR_TRUCK: repair_truck(session.rules, state, RepairTruck()).accepted,
        CommandType.BUY_TRUCK: any(buy_truck_quantity),
        CommandType.NEXT_DAY: True,
    }
    return ActionMask(
        action=tuple(active[command_type] for command_type in vocabulary.action_types),
        buy=buy,
        sell=sell,
        travel=travel,
        borrow_quantity=borrow_quantity,
        repay_quantity=repay_quantity,
        buy_truck_quantity=buy_truck_quantity,
    )


def _travel_mask(
    session: GameSession, vocabulary: ActionVocabulary
) -> tuple[tuple[_Mask, ...], ...]:
    travel: list[tuple[_Mask, ...]] = []
    for city_name in vocabulary.city_names:
        city_fast_masks: list[_Mask] = []
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
        travel.append(tuple(city_fast_masks))
    return tuple(travel)


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
