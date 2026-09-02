"""智能体动作的离散协议与目录索引。

本模块只定义动作空间，不依赖 PyTorch、Gymnasium 或具体网络。当前策略只将数量
档位的公开语义作为输入；商品、城市和运输方式索引仅用于选择相应候选行，最终都
必须还原为 ``ActionHead``。
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING
from enum import IntEnum
from typing import ClassVar, Mapping

from trade_game.core import Catalog, CommandType, TransportMode, money


class ActionProtocolError(ValueError):
    """动作头的索引或目录定义不符合协议。"""


class QuantityBin(IntEnum):
    """统一数量头的 21 个档位。

    ``ONE`` 表示 1 个单位；其余档位表示当前最大可行数量的百分比。
    """

    ONE = 0
    PERCENT_5 = 1
    PERCENT_10 = 2
    PERCENT_15 = 3
    PERCENT_20 = 4
    PERCENT_25 = 5
    PERCENT_30 = 6
    PERCENT_35 = 7
    PERCENT_40 = 8
    PERCENT_45 = 9
    PERCENT_50 = 10
    PERCENT_55 = 11
    PERCENT_60 = 12
    PERCENT_65 = 13
    PERCENT_70 = 14
    PERCENT_75 = 15
    PERCENT_80 = 16
    PERCENT_85 = 17
    PERCENT_90 = 18
    PERCENT_95 = 19
    PERCENT_100 = 20

    @property
    def label(self) -> str:
        """返回供日志、调试和后续界面使用的稳定档位名称。"""

        return "1" if self is QuantityBin.ONE else f"{self.value * 5}%"

    @property
    def ratio(self) -> Decimal | None:
        """返回百分比档位的小数比例；单个单位档位没有比例。"""

        if self is QuantityBin.ONE:
            return None
        return Decimal(self.value * 5) / Decimal("100")


def integer_quantity_from_bin(quantity_bin: QuantityBin, maximum: int) -> int:
    """将数量档位换算为整数数量；调用方保证最大数量为正。"""

    ratio = quantity_bin.ratio
    if ratio is None:
        return 1
    return int((Decimal(maximum) * ratio).to_integral_value(rounding=ROUND_CEILING))


def money_amount_from_bin(quantity_bin: QuantityBin, maximum: Decimal) -> Decimal:
    """将数量档位换算为货币金额；调用方保证最大金额为正。"""

    ratio = quantity_bin.ratio
    amount = min(Decimal("1"), maximum) if ratio is None else maximum * ratio
    return money(amount)


ACTION_TYPES: tuple[CommandType, ...] = (
    CommandType.BUY,
    CommandType.SELL,
    CommandType.TRAVEL,
    CommandType.BORROW,
    CommandType.REPAY,
    CommandType.REPAIR_TRUCK,
    CommandType.BUY_TRUCK,
    CommandType.NEXT_DAY,
)


@dataclass(frozen=True, slots=True)
class ActionHead:
    """策略输出的离散动作头。

    所有字段都是分类索引。无关字段仍需落在各自词表范围内，便于统一的
    多头网络直接输出固定宽度向量。字段顺序也是后续轨迹持久化的固定顺序。
    """

    action_index: int
    product_index: int = 0
    city_index: int = 0
    transport_index: int = 0
    quantity_index: int = int(QuantityBin.ONE)
    fast_index: int = 0

    def __post_init__(self) -> None:
        for name, value in (
            ("action_index", self.action_index),
            ("product_index", self.product_index),
            ("city_index", self.city_index),
            ("transport_index", self.transport_index),
            ("quantity_index", self.quantity_index),
            ("fast_index", self.fast_index),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ActionProtocolError(f"{name} 必须是非负整数")
        if self.fast_index not in (0, 1):
            raise ActionProtocolError("fast_index 必须为 0 或 1")
        if self.quantity_index >= len(QuantityBin):
            raise ActionProtocolError("quantity_index 超出 21 档数量空间")

    def as_tuple(self) -> tuple[int, int, int, int, int, int]:
        """按固定顺序返回动作头，供网络输出和轨迹记录使用。"""

        return (
            self.action_index,
            self.product_index,
            self.city_index,
            self.transport_index,
            self.quantity_index,
            self.fast_index,
        )


@dataclass(frozen=True, slots=True)
class ActionVocabulary:
    """由当前游戏目录生成的稳定动作词表。"""

    product_ids: tuple[str, ...]
    city_names: tuple[str, ...]
    action_types: ClassVar[tuple[CommandType, ...]] = ACTION_TYPES
    transport_modes: ClassVar[tuple[TransportMode, ...]] = (
        TransportMode.LAND,
        TransportMode.SEA,
    )

    @classmethod
    def from_catalog(cls, catalog: Catalog) -> "ActionVocabulary":
        """按已校验目录顺序生成商品和城市索引。"""

        return cls(product_ids=tuple(catalog.products), city_names=tuple(catalog.cities))

    @property
    def head_sizes(self) -> Mapping[str, int]:
        """返回多头网络需要的每个分类头宽度。"""

        return {
            "action": len(self.action_types),
            "product": len(self.product_ids),
            "city": len(self.city_names),
            "transport": len(self.transport_modes),
            "quantity": len(QuantityBin),
            "fast": 2,
        }

    def validate_catalog(self, catalog: Catalog) -> None:
        """确保模型词表与当前游戏目录使用完全相同的实体顺序。"""

        if self.product_ids != tuple(catalog.products):
            raise ActionProtocolError("商品词表与当前游戏目录不一致")
        if self.city_names != tuple(catalog.cities):
            raise ActionProtocolError("城市词表与当前游戏目录不一致")

    def action_type(self, index: int) -> CommandType:
        return self.action_types[_require_index(index, len(self.action_types), "action_index")]

    def product_id(self, index: int) -> str:
        return self.product_ids[_require_index(index, len(self.product_ids), "product_index")]

    def city_name(self, index: int) -> str:
        return self.city_names[_require_index(index, len(self.city_names), "city_index")]

    def transport_mode(self, index: int) -> TransportMode:
        return self.transport_modes[_require_index(index, len(self.transport_modes), "transport_index")]

    def action_index(self, action_type: CommandType) -> int:
        return self.action_types.index(action_type)

    def product_index(self, product_id: str) -> int:
        return self.product_ids.index(product_id)

    def city_index(self, city_name: str) -> int:
        return self.city_names.index(city_name)

    def transport_index(self, mode: TransportMode) -> int:
        return self.transport_modes.index(mode)

    def validate(self, action: ActionHead) -> CommandType:
        """校验动作及各参数头是否落在固定词表范围内。"""

        action_type = self.action_type(action.action_index)
        self.product_id(action.product_index)
        self.city_name(action.city_index)
        self.transport_mode(action.transport_index)
        return action_type


def _require_index(value: int, size: int, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < size:
        raise ActionProtocolError(f"{field} 超出动作词表范围")
    return value


__all__ = [
    "ACTION_TYPES",
    "ActionHead",
    "ActionProtocolError",
    "ActionVocabulary",
    "QuantityBin",
    "integer_quantity_from_bin",
    "money_amount_from_bin",
]
