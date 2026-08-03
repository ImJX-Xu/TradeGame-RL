"""游戏核心接受的强类型命令协议。"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import ClassVar, TypeAlias

from .models import TransportMode


class CommandType(StrEnum):
    """玩家和智能体可发出的全部根命令。"""

    NEXT_DAY = "next_day"
    BUY = "buy"
    SELL = "sell"
    TRAVEL = "travel"
    REPAIR_TRUCK = "repair_truck"
    BUY_TRUCK = "buy_truck"
    BORROW = "borrow"
    REPAY = "repay"


class CommandValidationError(ValueError):
    """命令自身的参数不满足基本格式约束。"""


def _non_empty_text(value: str, field: str) -> str:
    if not isinstance(value, str):
        raise CommandValidationError(f"{field} 必须是字符串")
    normalized = value.strip()
    if not normalized:
        raise CommandValidationError(f"{field} 不能为空")
    return normalized


def _positive_quantity(value: int, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CommandValidationError(f"{field} 必须是正整数")
    return value


def _positive_amount(value: Decimal, field: str) -> Decimal:
    if not isinstance(value, Decimal) or not value.is_finite() or value <= 0:
        raise CommandValidationError(f"{field} 必须是有限正 Decimal")
    return value


@dataclass(frozen=True, slots=True)
class NextDay:
    """正常推进一个游戏日。"""

    command_type: ClassVar[CommandType] = CommandType.NEXT_DAY


@dataclass(frozen=True, slots=True)
class Buy:
    """在当前位置购买指定数量的商品。"""

    product_id: str
    quantity: int
    command_type: ClassVar[CommandType] = CommandType.BUY

    def __post_init__(self) -> None:
        object.__setattr__(self, "product_id", _non_empty_text(self.product_id, "product_id"))
        _positive_quantity(self.quantity, "quantity")


@dataclass(frozen=True, slots=True)
class Sell:
    """在当前位置出售指定数量的商品。"""

    product_id: str
    quantity: int
    command_type: ClassVar[CommandType] = CommandType.SELL

    def __post_init__(self) -> None:
        object.__setattr__(self, "product_id", _non_empty_text(self.product_id, "product_id"))
        _positive_quantity(self.quantity, "quantity")


@dataclass(frozen=True, slots=True)
class Travel:
    """使用指定方式前往目标城市。"""

    destination: str
    mode: TransportMode
    fast: bool = False
    command_type: ClassVar[CommandType] = CommandType.TRAVEL

    def __post_init__(self) -> None:
        object.__setattr__(self, "destination", _non_empty_text(self.destination, "destination"))
        if not isinstance(self.mode, TransportMode):
            raise CommandValidationError("mode 必须是 TransportMode")
        if not isinstance(self.fast, bool):
            raise CommandValidationError("fast 必须是 bool")


@dataclass(frozen=True, slots=True)
class RepairTruck:
    """维修当前货车车队。"""

    command_type: ClassVar[CommandType] = CommandType.REPAIR_TRUCK


@dataclass(frozen=True, slots=True)
class BuyTruck:
    """购买指定数量的货车。"""

    quantity: int
    command_type: ClassVar[CommandType] = CommandType.BUY_TRUCK

    def __post_init__(self) -> None:
        _positive_quantity(self.quantity, "quantity")


@dataclass(frozen=True, slots=True)
class Borrow:
    """在当前位置申请一笔贷款。"""

    amount: Decimal
    command_type: ClassVar[CommandType] = CommandType.BORROW

    def __post_init__(self) -> None:
        _positive_amount(self.amount, "amount")


@dataclass(frozen=True, slots=True)
class Repay:
    """在当前位置偿还指定金额。"""

    amount: Decimal
    command_type: ClassVar[CommandType] = CommandType.REPAY

    def __post_init__(self) -> None:
        _positive_amount(self.amount, "amount")


Command: TypeAlias = NextDay | Buy | Sell | Travel | RepairTruck | BuyTruck | Borrow | Repay


__all__ = [
    "Borrow",
    "Buy",
    "BuyTruck",
    "Command",
    "CommandType",
    "CommandValidationError",
    "NextDay",
    "RepairTruck",
    "Repay",
    "Sell",
    "Travel",
]
