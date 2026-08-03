"""新游戏的静态初始化配置与状态工厂。"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from types import MappingProxyType

from .catalog import Catalog
from .models import GameMode, GameState, MarketState, PlayerState


@dataclass(frozen=True, slots=True)
class NewGameConfig:
    """创建新游戏所需的初始条件，不含任何经济规则参数。"""

    initial_cash: Decimal = Decimal("1000")
    initial_location: str = "郑州"
    initial_day: int = 1
    initial_truck_count: int = 1
    initial_truck_capacity: int = 100
    initial_truck_durability: Decimal = Decimal("100")
    mode: GameMode = GameMode.FREE


DEFAULT_NEW_GAME_CONFIG = NewGameConfig()


def create_initial_state(
    catalog: Catalog, config: NewGameConfig = DEFAULT_NEW_GAME_CONFIG
) -> GameState:
    """创建零库存、零贷款、零价格扰动的全新游戏状态。"""

    _validate_config(catalog, config)
    # 价格波动由城市和商品共同决定，不能把不同城市的市场价格混为一项。
    zero_lambdas = MappingProxyType(
        {
            (city_name, product_id): Decimal("0")
            for city_name in catalog.cities
            for product_id in catalog.products
        }
    )
    return GameState(
        player=PlayerState(
            cash=config.initial_cash,
            location=config.initial_location,
            truck_count=config.initial_truck_count,
            truck_total_capacity=config.initial_truck_capacity,
            truck_durability=config.initial_truck_durability,
            sea_departure_port=None,
            cargo_lots=(),
        ),
        day=config.initial_day,
        mode=config.mode,
        market=MarketState(current_lambdas=zero_lambdas, previous_lambdas=zero_lambdas),
        loans=(),
        visited_cities=frozenset({config.initial_location}),
        loss_by_product=MappingProxyType({}),
    )


def _validate_config(catalog: Catalog, config: NewGameConfig) -> None:
    if config.initial_location not in catalog.cities:
        raise ValueError(f"初始城市不存在：{config.initial_location}")
    if config.initial_cash < 0:
        raise ValueError("初始现金不能为负")
    if config.initial_day < 1:
        raise ValueError("初始天数必须从 1 开始")
    if config.initial_truck_count < 1:
        raise ValueError("初始货车数量必须至少为 1")
    if config.initial_truck_capacity <= 0:
        raise ValueError("初始货车容量必须大于 0")
    if not Decimal("0") <= config.initial_truck_durability <= Decimal("100"):
        raise ValueError("初始货车耐久度必须在 0 到 100 之间")
