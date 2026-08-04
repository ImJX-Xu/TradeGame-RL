"""将 Agent 动作头解码为核心游戏命令。"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING

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
    free_capacity,
    money,
    purchase_unit_price,
    total_debt,
)

from .actions import (
    ActionHead,
    ActionProtocolError,
    ActionVocabulary,
    QuantityBin,
)


class ActionDecodeError(ActionProtocolError):
    """动作头无法在当前会话中转换成有效的核心命令参数。"""


@dataclass(frozen=True, slots=True)
class ActionDecoder:
    """使用固定词表和当前游戏会话完成动作解码。"""

    vocabulary: ActionVocabulary

    def decode(self, session: GameSession, action: ActionHead) -> Command:
        """将分类索引转换为核心 ``Command``，不直接执行命令。"""

        self.vocabulary.validate_catalog(session.catalog)
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
                    self._maximum_truck_quantity(session),
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
            self._maximum_buy_quantity(session, product_id),
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

    def _maximum_buy_quantity(self, session: GameSession, product_id: str) -> int:
        state = session.state
        maximum = free_capacity(state.player.cargo_lots, state.player.truck_total_capacity)
        product = session.catalog.product(product_id)
        if state.player.location not in product.origins:
            return 0
        unit_price = purchase_unit_price(
            session.catalog,
            session.rules,
            state,
            product_id,
            state.player.location,
        )
        return min(maximum, int(state.player.cash / unit_price))

    @staticmethod
    def _maximum_truck_quantity(session: GameSession) -> int:
        price = session.rules.vehicles.purchase_price
        return int(session.state.player.cash / price)


def decode_action(
    session: GameSession,
    action: ActionHead,
    vocabulary: ActionVocabulary,
) -> Command:
    """使用目录词表把 Agent 动作头转换成核心命令。"""

    return ActionDecoder(vocabulary).decode(session, action)


def _resolve_integer_quantity(quantity_bin: QuantityBin, maximum: int, field: str) -> int:
    if maximum <= 0:
        raise ActionDecodeError(f"当前没有{field}")
    ratio = quantity_bin.ratio
    if ratio is None:
        return 1
    return int(
        (Decimal(maximum) * ratio).to_integral_value(rounding=ROUND_CEILING)
    )


def _resolve_amount(quantity_bin: QuantityBin, maximum: Decimal, field: str) -> Decimal:
    if maximum <= 0:
        raise ActionDecodeError(f"当前没有{field}")
    ratio = quantity_bin.ratio
    amount = min(Decimal("1"), maximum) if ratio is None else maximum * ratio
    amount = money(amount)
    if amount <= 0:
        raise ActionDecodeError(f"当前{field}低于最小金额单位")
    return amount


__all__ = ["ActionDecodeError", "ActionDecoder", "decode_action"]
