"""领域对象定义，不包含任何规则或状态变更。"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import Mapping, TypeAlias


CityProductKey: TypeAlias = tuple[str, str]


class TransportMode(StrEnum):
    """城市间可用的运输方式。"""

    LAND = "land"
    SEA = "sea"


class ProductCategory(StrEnum):
    """商品的经济与运输类别。"""

    BASE = "base"
    LIGHT_INDUSTRY = "light_industry"
    ELECTRONICS = "electronics"
    PERISHABLE = "perishable"


class SpecialtyScope(StrEnum):
    """商品产地是单城还是区域。"""

    CITY = "city"
    REGION = "region"


class GameMode(StrEnum):
    """核心支持的游戏模式。具体限制由后续规则层定义。"""

    FREE = "free"
    CHALLENGE = "challenge"


class GameEndReason(StrEnum):
    """终局的领域原因。"""

    BANKRUPTCY = "bankruptcy"
    TIME_LIMIT = "time_limit"


@dataclass(frozen=True, slots=True)
class City:
    """静态城市资料。城市名称是当前版本中的稳定标识。"""

    name: str
    region: str
    modes: frozenset[TransportMode]
    has_bank: bool
    has_port: bool
    latitude: Decimal
    longitude: Decimal
    is_high_consumption: bool


@dataclass(frozen=True, slots=True)
class Product:
    """静态商品资料及其价格波动参数。"""

    id: str
    name: str
    category: ProductCategory
    base_purchase_price: Decimal
    profit_margin_rate: Decimal
    origins: frozenset[str]
    specialty_scope: SpecialtyScope
    specialty_region: str | None
    perishable_shelf_life_days: int | None
    perishable_aging_strength: Decimal | None
    lambda_min: Decimal
    lambda_max: Decimal
    lambda_alpha: Decimal
    lambda_sigma: Decimal
    transport_loss_rate: Decimal


@dataclass(frozen=True, slots=True)
class Route:
    """一条无向路线的原始数据记录。"""

    from_city: str
    to_city: str
    mode: TransportMode
    distance_km: int


@dataclass(frozen=True, slots=True)
class CargoLot:
    """玩家持有的一批货物。库存规则将在后续阶段实现。"""

    product_id: str
    quantity: int
    origin_city: str
    shelf_life_remaining_days: int | None
    age_days: int = 0

    def __post_init__(self) -> None:
        if not self.product_id.strip():
            raise ValueError("货物批次的商品 ID 不能为空")
        if self.quantity <= 0:
            raise ValueError("货物批次数量必须大于 0")
        if not self.origin_city.strip():
            raise ValueError("货物批次的产地不能为空")
        if self.shelf_life_remaining_days is not None and self.shelf_life_remaining_days <= 0:
            raise ValueError("货物批次的剩余保质期必须大于 0")
        if self.age_days < 0:
            raise ValueError("货物批次年龄不能为负")


@dataclass(frozen=True, slots=True)
class Loan:
    """一笔未偿还贷款的状态记录。"""

    principal: Decimal
    start_day: int
    accrued_interest: Decimal = Decimal("0")


@dataclass(frozen=True, slots=True)
class GameOutcome:
    """已结束对局的结算结果。"""

    reason: GameEndReason
    final_assets: Decimal


@dataclass(frozen=True, slots=True)
class PlayerState:
    """玩家的可变游戏状态快照。"""

    cash: Decimal
    location: str
    truck_count: int
    truck_total_capacity: int
    truck_durability: Decimal
    sea_departure_port: str | None
    cargo_lots: tuple[CargoLot, ...]


@dataclass(frozen=True, slots=True)
class MarketState:
    """市场状态；价格规则只在后续阶段消费这些数据。"""

    current_lambdas: Mapping[CityProductKey, Decimal]
    previous_lambdas: Mapping[CityProductKey, Decimal]


@dataclass(frozen=True, slots=True)
class GameState:
    """完整的游戏状态快照，由后续 GameSession 唯一拥有并替换。"""

    player: PlayerState
    day: int
    mode: GameMode
    market: MarketState
    loans: tuple[Loan, ...]
    visited_cities: frozenset[str]
    loss_by_product: Mapping[str, int]
    outcome: GameOutcome | None = None
