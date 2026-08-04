"""面向界面与智能体的公开市场报价视图。"""

from __future__ import annotations

from decimal import Decimal

from .catalog import Catalog
from .models import GameState
from .price_functions import sale_unit_price_at_adjustment
from .rules import GameRules
from .transport import reference_sale_origin, remote_sale_distance_premium


def reference_sale_price_history(
    catalog: Catalog,
    rules: GameRules,
    state: GameState,
    product_id: str,
    city_name: str,
) -> tuple[Decimal, ...]:
    """将核心保存的行情历史转换为公开的参考出售价格序列。"""

    product = catalog.product(product_id)
    origin_city = reference_sale_origin(catalog, product, city_name)
    distance_premium = remote_sale_distance_premium(catalog, rules, origin_city, city_name)
    return tuple(
        sale_unit_price_at_adjustment(
            catalog,
            rules,
            product_id,
            city_name,
            origin_city=origin_city,
            market_adjustment=adjustment,
            remote_distance_premium=distance_premium,
        )
        for adjustment in state.market.price_adjustment_history[(city_name, product_id)]
    )


__all__ = ["reference_sale_price_history"]
