"""新游戏的静态初始化配置与状态工厂。"""

from __future__ import annotations

from decimal import Decimal
from types import MappingProxyType

from .catalog import Catalog
from .models import GameMode, GameState, MarketState, PlayerState
from .rules import GameRules


def create_initial_state(
    catalog: Catalog, rules: GameRules, *, mode: GameMode = GameMode.FREE
) -> GameState:
    """创建零库存、零贷款、零价格扰动的全新游戏状态。"""

    initial = rules.initial
    # 价格波动由城市和商品共同决定，不能把不同城市的市场价格混为一项。
    zero_adjustments = MappingProxyType(
        {
            (city_name, product_id): Decimal("0")
            for city_name in catalog.cities
            for product_id in catalog.products
        }
    )
    price_history = MappingProxyType({key: (value,) for key, value in zero_adjustments.items()})
    purchase_available_days = MappingProxyType(
        {key: initial.initial_day for key in zero_adjustments}
    )
    product_trends = MappingProxyType({product_id: Decimal("0") for product_id in catalog.products})
    return GameState(
        player=PlayerState(
            cash=initial.initial_cash,
            location=initial.initial_location,
            truck_count=initial.initial_truck_count,
            truck_total_capacity=initial.initial_truck_capacity,
            truck_durability=initial.initial_truck_durability,
            sea_departure_port=None,
            cargo_lots=(),
        ),
        day=initial.initial_day,
        mode=mode,
        market=MarketState(
            current_price_adjustments=zero_adjustments,
            product_trends=product_trends,
            local_spreads=zero_adjustments,
            price_adjustment_history=price_history,
            purchase_available_days=purchase_available_days,
            active_events=(),
        ),
        loans=(),
        visited_cities=frozenset({initial.initial_location}),
        loss_by_product=MappingProxyType({}),
    )
