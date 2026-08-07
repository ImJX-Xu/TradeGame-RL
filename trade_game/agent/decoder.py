"""将 Agent 动作头解码为核心游戏命令。"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from trade_game.core import (
    Borrow,
    Buy,
    BuyTruck,
    Command,
    CommandType,
    GameSession,
    NextDay,
    RepairTruck,
    Repay,
    Sell,
    Travel,
    available_credit,
    cargo_quantity,
    maximum_purchase_quantity,
    maximum_truck_quantity,
    money,
    total_debt,
)

from .actions import (
    ActionHead,
    ActionProtocolError,
    ActionVocabulary,
    QuantityBin,
    integer_quantity_from_bin,
    money_amount_from_bin,
)


class ActionDecodeError(ActionProtocolError):
    """动作头无法在当前会话中转换成有效的核心命令参数。"""


@dataclass(frozen=True, slots=True)
class ActionDecoder:
    """使用固定词表和当前游戏会话完成动作解码。"""

    vocabulary: ActionVocabulary

    def decode(self, session: GameSession, action: ActionHead) -> Command:
        """将分类索引转换为核心 ``Command``，不直接执行命令。"""

        command_type = self.vocabulary.validate(action)
        if command_type is CommandType.BUY:
            return self._decode_buy(session, action)
        if command_type is CommandType.SELL:
            return self._decode_sell(session, action)
        if command_type is CommandType.TRAVEL:
            return Travel(
                destination=self.vocabulary.city_name(action.city_index),
                mode=self.vocabulary.transport_mode(action.transport_index),
                fast=bool(action.fast_index),
            )
        if command_type is CommandType.BORROW:
            return Borrow(
                amount=_resolve_amount(
                    QuantityBin(action.quantity_index),
                    available_credit(session.catalog, session.rules, session.state),
                    "可借额度",
                )
            )
        if command_type is CommandType.REPAY:
            return Repay(
                amount=_resolve_amount(
                    QuantityBin(action.quantity_index),
                    min(total_debt(session.state.loans), session.state.player.cash),
                    "可还金额",
                )
            )
        if command_type is CommandType.REPAIR_TRUCK:
            return RepairTruck()
        if command_type is CommandType.BUY_TRUCK:
            return BuyTruck(
                quantity=_resolve_integer_quantity(
                    QuantityBin(action.quantity_index),
                    maximum_truck_quantity(session.rules, session.state),
                    "可购买货车数量",
                )
            )
        if command_type is CommandType.NEXT_DAY:
            return NextDay()
        raise AssertionError(f"动作协议未覆盖核心命令：{command_type.value}")

    def _decode_buy(self, session: GameSession, action: ActionHead) -> Buy:
        product_id = self.vocabulary.product_id(action.product_index)
        quantity = _resolve_integer_quantity(
            QuantityBin(action.quantity_index),
            maximum_purchase_quantity(session.catalog, session.rules, session.state, product_id),
            "可采购数量",
        )
        return Buy(product_id=product_id, quantity=quantity)

    def _decode_sell(self, session: GameSession, action: ActionHead) -> Sell:
        product_id = self.vocabulary.product_id(action.product_index)
        quantity = _resolve_integer_quantity(
            QuantityBin(action.quantity_index),
            cargo_quantity(session.state.player.cargo_lots, product_id),
            "可出售数量",
        )
        return Sell(product_id=product_id, quantity=quantity)


def decode_action(
    session: GameSession,
    action: ActionHead,
    vocabulary: ActionVocabulary,
) -> Command:
    """使用目录词表把 Agent 动作头转换成核心命令。"""

    return ActionDecoder(vocabulary).decode(session, action)


def encode_command(
    session: GameSession,
    command: Command,
    vocabulary: ActionVocabulary,
) -> ActionHead:
    """将核心命令编码为当前会话下可解码的动作头。

    贪心教师使用核心命令工作，而策略网络只输出固定的六个离散索引。数量
    头只有 21 个档位，因此这里会在当前最大可行数量上选择能最接近命令的
    档位；若恰好可表示，则优先使用精确档位。
    """

    if isinstance(command, Buy):
        maximum = maximum_purchase_quantity(
            session.catalog,
            session.rules,
            session.state,
            command.product_id,
        )
        quantity_index = _encode_integer_quantity(command.quantity, maximum, "采购数量")
        return ActionHead(
            action_index=vocabulary.action_index(CommandType.BUY),
            product_index=vocabulary.product_index(command.product_id),
            quantity_index=quantity_index,
        )
    if isinstance(command, Sell):
        maximum = cargo_quantity(session.state.player.cargo_lots, command.product_id)
        quantity_index = _encode_integer_quantity(command.quantity, maximum, "出售数量")
        return ActionHead(
            action_index=vocabulary.action_index(CommandType.SELL),
            product_index=vocabulary.product_index(command.product_id),
            quantity_index=quantity_index,
        )
    if isinstance(command, Travel):
        return ActionHead(
            action_index=vocabulary.action_index(CommandType.TRAVEL),
            city_index=vocabulary.city_index(command.destination),
            transport_index=vocabulary.transport_index(command.mode),
            fast_index=int(command.fast),
        )
    if isinstance(command, Borrow):
        maximum = available_credit(session.catalog, session.rules, session.state)
        quantity_index = _encode_money_amount(command.amount, maximum, "借款金额")
        return ActionHead(
            action_index=vocabulary.action_index(CommandType.BORROW),
            quantity_index=quantity_index,
        )
    if isinstance(command, Repay):
        maximum = min(total_debt(session.state.loans), session.state.player.cash)
        quantity_index = _encode_money_amount(command.amount, maximum, "还款金额")
        return ActionHead(
            action_index=vocabulary.action_index(CommandType.REPAY),
            quantity_index=quantity_index,
        )
    if isinstance(command, BuyTruck):
        maximum = maximum_truck_quantity(session.rules, session.state)
        quantity_index = _encode_integer_quantity(command.quantity, maximum, "购车数量")
        return ActionHead(
            action_index=vocabulary.action_index(CommandType.BUY_TRUCK),
            quantity_index=quantity_index,
        )
    if isinstance(command, RepairTruck):
        return ActionHead(action_index=vocabulary.action_index(CommandType.REPAIR_TRUCK))
    if isinstance(command, NextDay):
        return ActionHead(action_index=vocabulary.action_index(CommandType.NEXT_DAY))
    raise AssertionError(f"动作协议未覆盖核心命令：{type(command).__name__}")


def _resolve_integer_quantity(quantity_bin: QuantityBin, maximum: int, field: str) -> int:
    if maximum <= 0:
        raise ActionDecodeError(f"当前没有{field}")
    return integer_quantity_from_bin(quantity_bin, maximum)


def _resolve_amount(quantity_bin: QuantityBin, maximum: Decimal, field: str) -> Decimal:
    if maximum <= 0:
        raise ActionDecodeError(f"当前没有{field}")
    amount = money_amount_from_bin(quantity_bin, maximum)
    if amount <= 0:
        raise ActionDecodeError(f"当前{field}低于最小金额单位")
    return amount


def _encode_integer_quantity(target: int, maximum: int, field: str) -> int:
    if target <= 0 or maximum <= 0 or target > maximum:
        raise ActionDecodeError(f"{field}超出当前可行范围")
    candidates = tuple(
        (index, integer_quantity_from_bin(quantity_bin, maximum))
        for index, quantity_bin in enumerate(QuantityBin)
    )
    return _closest_quantity_index(target, candidates)


def _encode_money_amount(target: Decimal, maximum: Decimal, field: str) -> int:
    target = money(target)
    maximum = money(maximum)
    if target <= 0 or maximum <= 0 or target > maximum:
        raise ActionDecodeError(f"{field}超出当前可行范围")
    candidates = tuple(
        (index, money_amount_from_bin(quantity_bin, maximum))
        for index, quantity_bin in enumerate(QuantityBin)
    )
    return _closest_quantity_index(target, candidates)


def _closest_quantity_index(
    target: int | Decimal,
    candidates: tuple[tuple[int, int | Decimal], ...],
) -> int:
    exact = tuple(index for index, value in candidates if value == target)
    if exact:
        return min(exact, key=lambda index: (index != int(QuantityBin.ONE), -index))
    return min(
        candidates,
        key=lambda item: (
            abs(item[1] - target),
            item[1] > target,
            -item[0],
        ),
    )[0]


__all__ = ["ActionDecodeError", "ActionDecoder", "decode_action", "encode_command"]
