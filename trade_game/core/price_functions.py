"""价格函数模块：采购价、售价、价格扰动和金额量化。"""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP

from .catalog import Catalog
from .models import GameState, Product
from .rules import GameRules


MONEY_QUANTUM = Decimal("0.01")


def money(value: Decimal) -> Decimal:
    """将金额按分进行四舍五入，作为所有现金变动的唯一量化函数。"""

    return value.quantize(MONEY_QUANTUM, rounding=ROUND_HALF_UP)


def can_purchase(product: Product, city_name: str) -> bool:
    """商品只能在其 CSV 声明的产地采购。"""

    return city_name in product.origins


def price_lambda(state: GameState, city_name: str, product_id: str) -> Decimal:
    """读取一个城市商品价格扰动。内部状态缺键时直接抛出 KeyError。"""

    return state.market.current_lambdas[(city_name, product_id)]


def purchase_unit_price(
    catalog: Catalog, rules: GameRules, state: GameState, product_id: str, city_name: str
) -> Decimal:
    """计算当前城市采购一单位商品的价格。"""

    product = catalog.product(product_id)
    catalog.city(city_name)
    if not can_purchase(product, city_name):
        raise ValueError(f"{city_name} 不能采购商品 {product_id}")
    price = product.base_purchase_price * (Decimal("1") + price_lambda(state, city_name, product_id))
    return _positive_money(price, product_id, city_name)


def sale_unit_price(
    catalog: Catalog,
    rules: GameRules,
    state: GameState,
    product_id: str,
    city_name: str,
    *,
    remote_distance_multiplier: Decimal = Decimal("1"),
) -> Decimal:
    """计算当前城市出售一单位商品的价格。

    异地距离乘数由路线规则层在后续阶段提供；本阶段默认值为 1。
    """

    product = catalog.product(product_id)
    city = catalog.city(city_name)
    if remote_distance_multiplier < Decimal("1"):
        raise ValueError("异地距离乘数不能小于 1")
    price = product.base_purchase_price * (Decimal("1") + price_lambda(state, city_name, product_id))
    if not can_purchase(product, city_name):
        price *= Decimal("1") + product.profit_margin_rate
        price *= remote_distance_multiplier
        if city.is_high_consumption:
            price *= rules.pricing.high_consumption_multiplier
    return _positive_money(price, product_id, city_name)


def trade_total(unit_price: Decimal, quantity: int) -> Decimal:
    """计算指定数量商品的现金总额。"""

    if unit_price <= 0:
        raise ValueError("单价必须大于 0")
    if quantity <= 0:
        raise ValueError("数量必须大于 0")
    return money(unit_price * quantity)


def _positive_money(value: Decimal, product_id: str, city_name: str) -> Decimal:
    rounded = money(value)
    if rounded <= 0:
        raise ValueError(f"商品 {product_id} 在 {city_name} 的价格必须大于 0")
    return rounded
