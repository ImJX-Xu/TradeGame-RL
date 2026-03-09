from __future__ import annotations

import random
from typing import Mapping

from .data import CITY_REGIONS, HIGH_CONSUMPTION_CITIES, CITIES, Product, PRODUCTS, SpecialtyScope
from .game_config import (
    HIGH_CITY_LAMBDA_ALPHA_MULTIPLIER,
    HIGH_CITY_LAMBDA_RANGE_MULTIPLIER,
    HIGH_CITY_LAMBDA_SIGMA_MULTIPLIER,
    HIGH_CONSUMPTION_CITY_MULTIPLIER,
    LAMBDA_ALPHA_ADJUSTMENT,
    LAMBDA_SIGMA_ADJUSTMENT,
    REMOTE_SALE_MULTIPLIER_MAX,
    REMOTE_SALE_MULTIPLIER_MIN,
)
from .transport import get_route_km_range, route_km_any, RouteNotFound

# 全局里程范围缓存，供异地乘数线性插值使用
_route_km_range: tuple[int, int] | None = None


def _get_cached_route_km_range() -> tuple[int, int]:
    global _route_km_range
    if _route_km_range is None:
        _route_km_range = get_route_km_range()
    return _route_km_range


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def refresh_daily_lambdas(
    rng: random.Random, previous_lambdas: Mapping[str, float] | None = None
) -> dict[str, float]:
    """
    每日凌晨刷新一次：为"城市×商品"生成当日 λ_city,prod（带惯性的区间随机波动）。
    
    公式（逐城市逐商品）：
      当日 λ_city,prod = α' × 昨日 λ_city,prod + ε
      ε ~ N(0, σ')
      然后截断到 [λ_min', λ_max']

    其中：
      - α' 在原始 alpha 的基础上略降低（减小惯性，增加随机性）
      - σ' 在原始 sigma 的基础上放大
      - 北上广深台等高消费城市可拥有更大的波动区间

    键格式："{city}|{product_id}"
    """
    new_lambdas: dict[str, float] = {}

    for city in CITIES.keys():
        is_high_city = city in HIGH_CONSUMPTION_CITIES
        for pid, prod in PRODUCTS.items():
            key = f"{city}|{pid}"

            # 惯性和波动参数：来自 game_config；高消费城市再乘 HIGH_CITY_* 系数
            alpha = prod.lambda_alpha * LAMBDA_ALPHA_ADJUSTMENT
            sigma = prod.lambda_sigma * LAMBDA_SIGMA_ADJUSTMENT
            lam_min = prod.lambda_min
            lam_max = prod.lambda_max

            if is_high_city:
                alpha *= HIGH_CITY_LAMBDA_ALPHA_MULTIPLIER
                sigma *= HIGH_CITY_LAMBDA_SIGMA_MULTIPLIER
                lam_min *= HIGH_CITY_LAMBDA_RANGE_MULTIPLIER
                lam_max *= HIGH_CITY_LAMBDA_RANGE_MULTIPLIER

            if previous_lambdas is None:
                prev_lam = 0.0
            else:
                prev_lam = float(previous_lambdas.get(key, 0.0))

            epsilon = rng.gauss(0.0, sigma)
            new_lam = alpha * prev_lam + epsilon
            new_lambdas[key] = _clamp(new_lam, lam_min, lam_max)

    return new_lambdas


def purchase_price(
    product: Product,
    city: str,
    lambdas: Mapping[str, float],
) -> float | None:
    """
    返回该城市当日"采购单价"。
    根据新策划案：采购价 = 基础采购价 × (1+λ)
    若该城市不可采购该商品（非产地），返回 None。
    """
    if city not in product.origins:
        return None
    
    lam_key = f"{city}|{product.id}"
    lam = float(lambdas.get(lam_key, 0.0))
    price = product.base_purchase_price * (1.0 + lam)
    
    # 台湾商品需额外缴纳10%关税（基于基础采购价）
    if city in ("台北", "高雄"):
        tariff = product.base_purchase_price * 0.10
        price += tariff
    
    return round(price, 2)


def can_sell_product_here(product: Product, city: str) -> bool:
    """
    是否允许在指定城市售卖该商品。
    现在允许所有城市售卖所有商品（包括本地特产）。
    本地售卖时价格会有特殊处理（见sell_unit_price）。
    """
    return True


def sell_unit_price(
    product: Product,
    city: str,
    lambdas: Mapping[str, float],
    *,
    quantity_sold: int,
    shelf_life_remaining_days: int | None = None,
) -> float:
    """
    返回该城市当日"售卖单价"。
    
    公式：售卖价 = 基础采购价 × (1+异地利润率) × (1+λ) × K异 × K高消
    
    本地售卖（城市特产在产地，区域特产在同区域）：
    - 异地利润率=0（即不使用profit_margin_rate）
    - K异=1
    - K高消=1
    - 公式简化为：售卖价 = 基础采购价 × (1+λ)
    
    异地售卖：
    - 异地利润率正常使用
    - K异 与「产地→售卖城」里程正相关，在全局最小/最大里程间线性插值，范围 [REMOTE_SALE_MULTIPLIER_MIN, REMOTE_SALE_MULTIPLIER_MAX]
    - K高消：普通区=1，高消区=HIGH_CONSUMPTION_CITY_MULTIPLIER
    """
    lam_key = f"{city}|{product.id}"
    lam = float(lambdas.get(lam_key, 0.0))
    
    # 判断是否为本地售卖
    is_local_sale = False
    if product.specialty_scope == SpecialtyScope.REGION and product.specialty_region:
        city_region = CITY_REGIONS.get(city)
        is_local_sale = (city_region == product.specialty_region)
    elif product.specialty_scope == SpecialtyScope.CITY:
        is_local_sale = (city in product.origins)
    
    if is_local_sale:
        # 本地售卖：异地利润率=0，K异=1，K高消=1
        # 公式：基础采购价 × (1+λ)
        base_price = product.base_purchase_price
        price_with_lambda = base_price * (1.0 + lam)
        k_place = 1.0
        k_high_consumption = 1.0
    else:
        # 异地售卖：K异 与里程正相关（1～2），多产地取最远距离
        base_profit = product.base_purchase_price * (1.0 + product.profit_margin_rate)
        price_with_lambda = base_profit * (1.0 + lam)
        km_min, km_max = _get_cached_route_km_range()
        km = 0
        for origin in product.origins:
            try:
                d = route_km_any(origin, city)
                if d > km:
                    km = d
            except RouteNotFound:
                pass
        if km_max > km_min and km >= km_min:
            t = (km - km_min) / (km_max - km_min)
            k_place = REMOTE_SALE_MULTIPLIER_MIN + t * (REMOTE_SALE_MULTIPLIER_MAX - REMOTE_SALE_MULTIPLIER_MIN)
            k_place = max(REMOTE_SALE_MULTIPLIER_MIN, min(REMOTE_SALE_MULTIPLIER_MAX, k_place))
        else:
            k_place = REMOTE_SALE_MULTIPLIER_MIN
        if city in HIGH_CONSUMPTION_CITIES:
            k_high_consumption = HIGH_CONSUMPTION_CITY_MULTIPLIER
        else:
            k_high_consumption = 1.0
    
    # 最终价格（不再区分临期折扣）
    final_price = price_with_lambda * k_place * k_high_consumption
    return round(final_price, 2)
