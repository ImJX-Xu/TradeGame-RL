"""
价格系统测试脚本

此脚本测试当前配置下各商品在不同城市的价格范围。
用于验证价格平衡性和生成价格速查表。

运行方式：
    python test_prices.py
"""

from trade_game.data import PRODUCTS, CITIES, HIGH_CONSUMPTION_CITIES, CITY_REGIONS
from trade_game.economy import purchase_price, sell_unit_price
from trade_game.game_config import (
    REMOTE_SALE_MULTIPLIER_MIN,
    REMOTE_SALE_MULTIPLIER_MAX,
    HIGH_CONSUMPTION_CITY_MULTIPLIER,
    TAIWAN_PURCHASE_TARIFF_RATE,
    LAND_COST_PER_KM,
    SEA_COST_PER_KM,
    TAIWAN_CUSTOMS,
    FAST_TRAVEL_COST_MULTIPLIER,
    FAST_TRAVEL_TIME_DIVISOR,
)


def test_price_ranges():
    """
    测试所有商品在不同城市的价格范围（λ=0时）
    生成类似价格速查表的输出
    """
    print("=" * 80)
    print("价格系统测试报告（λ=0时）")
    print("=" * 80)
    print()

    # 创建空的lambdas字典（所有λ=0）
    lambdas = {}
    for city in CITIES.keys():
        for pid in PRODUCTS.keys():
            key = f"{city}|{pid}"
            lambdas[key] = 0.0

    # 按商品类别分组
    categories = {
        "基础民生品【民】": [],
        "轻工消费品【轻】": [],
        "易损商品【损】": [],
        "生鲜特产【鲜】": [],
    }

    for pid, product in PRODUCTS.items():
        if product.category.value == "base":
            categories["基础民生品【民】"].append(product)
        elif product.category.value == "light_industry":
            categories["轻工消费品【轻】"].append(product)
        elif product.category.value == "electronics":
            categories["易损商品【损】"].append(product)
        elif product.category.value == "perishable":
            categories["生鲜特产【鲜】"].append(product)

    # 选择几个代表性城市进行测试
    test_cities = {
        "普通城市": ["郑州", "沈阳", "太原"],
        "高消费城市": ["北京", "上海", "广州"],
    }

    print("## 核心公式")
    print()
    print("### 采购价")
    print("```")
    print("采购价 = 基础采购价 × (1 + λ)")
    print("台北/高雄产地：+ 基础采购价 × 10%（关税）")
    print("```")
    print()
    print("### 售卖价")
    print("```")
    print("本地售卖（产地或同区域）：  基础采购价 × (1 + λ)")
    print()
    print(
        "异地普通城市：基础采购价 × (1 + 异地利润率) × (1 + λ) × K异，"
        f"其中 K异 ∈ [{REMOTE_SALE_MULTIPLIER_MIN}, {REMOTE_SALE_MULTIPLIER_MAX}]，随产地→售卖城里程线性变化"
    )
    print(
        "异地高消费城市：在上述基础上再乘以 K高消，"
        f"K高消 = {HIGH_CONSUMPTION_CITY_MULTIPLIER}（北京/上海/广州/深圳/台北/高雄）"
    )
    print("```")
    print()
    print("**高消费城市**：", ", ".join(sorted(HIGH_CONSUMPTION_CITIES)))
    print()
    print("-" * 80)
    print()

    # 测试每个商品
    for category_name, products in categories.items():
        if not products:
            continue

        print(f"## {category_name}")
        print()
        print(
            "| 商品 | 基础价 | 异地利润率 | 采购价 | 本地售卖 | 异地普通 | 异地高消 | 台湾关税 |"
        )
        print("|------|--------|-----------|--------|----------|----------|----------|----------|")

        for product in products:
            # 获取产地（取第一个）
            origin_city = list(product.origins)[0]
            is_taiwan_origin = origin_city in ("台北", "高雄")

            # 计算采购价（λ=0）
            base_price = product.base_purchase_price
            purchase = base_price * (1.0 + 0.0)  # λ=0
            if is_taiwan_origin:
                purchase += base_price * TAIWAN_PURCHASE_TARIFF_RATE
                tariff_str = f"+{int(base_price * TAIWAN_PURCHASE_TARIFF_RATE)}"
            else:
                tariff_str = "-"

            # 计算本地售卖价（λ=0）——直接调用实际定价函数，确保与游戏逻辑一致
            local_sale = sell_unit_price(
                product,
                origin_city,
                lambdas,
                quantity_sold=1,
                shelf_life_remaining_days=None,
            )

            # 计算异地普通 / 高消费城市售卖价（λ=0）——同样复用 sell_unit_price
            # 选取一个代表性的异地普通城市和高消费城市
            remote_normal_city = "太原" if origin_city != "太原" else "沈阳"
            remote_normal = sell_unit_price(
                product,
                remote_normal_city,
                lambdas,
                quantity_sold=1,
                shelf_life_remaining_days=None,
            )

            remote_high_city = "北京"
            remote_high = sell_unit_price(
                product,
                remote_high_city,
                lambdas,
                quantity_sold=1,
                shelf_life_remaining_days=None,
            )

            # 格式化输出
            profit_rate_pct = product.profit_margin_rate * 100
            print(
                f"| {product.name} | {base_price:.0f} | {profit_rate_pct:.1f}% | "
                f"{purchase:.2f} | {local_sale:.2f} | {remote_normal:.2f} | "
                f"{remote_high:.2f} | {tariff_str} |"
            )

        print()

    # 测试价格范围（考虑λ波动）
    print("-" * 80)
    print()
    print("## λ波动对价格的影响")
    print()
    print("以下测试几个典型商品在不同λ值下的价格范围：")
    print()

    test_products = [
        ("郑州特精粉", "zz_flour", "郑州"),
        ("深圳传呼机", "sz_pager", "深圳"),
        ("台北高山茶", "tp_tea", "台北"),
        ("福州带鱼", "fz_fish", "福州"),
    ]

    for name, pid, origin in test_products:
        product = PRODUCTS[pid]
        print(f"### {name}（基础价 {product.base_purchase_price:.0f}元，异地利润率 {product.profit_margin_rate*100:.1f}%）")
        print()
        print("| λ值 | 采购价 | 本地售卖 | 异地普通 | 异地高消 |")
        print("|-----|--------|----------|----------|----------|")

        # 获取λ范围
        lambda_min = product.lambda_min
        lambda_max = product.lambda_max

        # 测试几个关键λ值
        test_lambdas = [
            (lambda_min, f"最低({lambda_min*100:.1f}%)"),
            (0.0, "正常(0.0%)"),
            (lambda_max, f"最高({lambda_max*100:.1f}%)"),
        ]

        for lam, lam_label in test_lambdas:
            # 更新lambdas
            key = f"{origin}|{pid}"
            lambdas[key] = lam

            # 计算价格
            purchase = purchase_price(product, origin, lambdas)
            if purchase is None:
                purchase = 0.0

            local_sale = sell_unit_price(
                product, origin, lambdas, quantity_sold=1, shelf_life_remaining_days=None
            )

            # 选择一个异地普通城市
            remote_normal_city = "太原" if origin != "太原" else "沈阳"
            remote_normal = sell_unit_price(
                product,
                remote_normal_city,
                lambdas,
                quantity_sold=1,
                shelf_life_remaining_days=None,
            )

            # 选择一个高消费城市
            remote_high_city = "北京"
            remote_high = sell_unit_price(
                product,
                remote_high_city,
                lambdas,
                quantity_sold=1,
                shelf_life_remaining_days=None,
            )

            print(
                f"| {lam_label} | {purchase:.2f} | {local_sale:.2f} | "
                f"{remote_normal:.2f} | {remote_high:.2f} |"
            )

        print()

    # 运输成本测试
    print("-" * 80)
    print()
    print("## 运输成本测试")
    print()
    print("### 陆运成本")
    print(f"- 每公里费用：{LAND_COST_PER_KM} 元/km")
    print(f"- 1辆车，100km：{LAND_COST_PER_KM * 100:.2f} 元")
    print(f"- 1辆车，500km：{LAND_COST_PER_KM * 500:.2f} 元")
    print(f"- 1辆车，1000km：{LAND_COST_PER_KM * 1000:.2f} 元")
    print(f"- 2辆车，1000km：{LAND_COST_PER_KM * 1000 * 2:.2f} 元")
    print()
    print("### 海运成本")
    print(f"- 每公里费用：{SEA_COST_PER_KM} 元/km")
    print(f"- 1艘船，100km：{SEA_COST_PER_KM * 100:.2f} 元")
    print(f"- 1艘船，500km：{SEA_COST_PER_KM * 500:.2f} 元")
    print(f"- 1艘船，1000km：{SEA_COST_PER_KM * 1000:.2f} 元")
    print(f"- 台湾航线额外关税：{TAIWAN_CUSTOMS} 元")
    print()
    print("### 快速出行")
    print(f"- 价格倍数：{FAST_TRAVEL_COST_MULTIPLIER}倍")
    print(f"- 时间除数：{FAST_TRAVEL_TIME_DIVISOR}倍（向下取整，最少1天）")
    print(f"- 示例：普通出行1000元，3天 → 快速出行3000元，1天")
    print()

    # 利润空间分析
    print("-" * 80)
    print()
    print("## 利润空间分析（λ=0，不考虑损耗和运输成本）")
    print()
    print("| 商品 | 单单位利润（异地普通） | 单单位利润（异地高消） |")
    print("|------|---------------------|----------------------|")

    for name, pid, origin in test_products:
        product = PRODUCTS[pid]
        key = f"{origin}|{pid}"
        lambdas[key] = 0.0

        purchase = purchase_price(product, origin, lambdas)
        if purchase is None:
            purchase = product.base_purchase_price
        if origin in ("台北", "高雄"):
            purchase = product.base_purchase_price * (1.0 + TAIWAN_PURCHASE_TARIFF_RATE)

        remote_normal_city = "太原" if origin != "太原" else "沈阳"
        remote_normal = sell_unit_price(
            product,
            remote_normal_city,
            lambdas,
            quantity_sold=1,
            shelf_life_remaining_days=None,
        )

        remote_high_city = "北京"
        remote_high = sell_unit_price(
            product,
            remote_high_city,
            lambdas,
            quantity_sold=1,
            shelf_life_remaining_days=None,
        )

        profit_normal = remote_normal - purchase
        profit_high = remote_high - purchase

        print(f"| {name} | {profit_normal:.2f}元 | {profit_high:.2f}元 |")

    print()
    print("=" * 80)
    print("测试完成")
    print("=" * 80)


if __name__ == "__main__":
    test_price_ranges()
