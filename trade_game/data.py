from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import FrozenSet, Iterable
import csv


class TransportMode(str, Enum):
    LAND = "land"
    SEA = "sea"


@dataclass(frozen=True, slots=True)
class City:
    name: str
    modes: FrozenSet[TransportMode]
    has_bank: bool
    has_port: bool
    lat: float
    lon: float


class ProductCategory(str, Enum):
    BASE = "base"  # 基础民生品
    LIGHT_INDUSTRY = "light_industry"  # 轻工消费品
    ELECTRONICS = "electronics"  # 易损商品（精密/易碎/易损坏）
    PERISHABLE = "perishable"  # 生鲜特产


class SpecialtyScope(str, Enum):
    CITY = "city"        # 城市特产（一城一品）
    REGION = "region"    # 区域特产（区域内多城可采购）


CITY_REGIONS: dict[str, str] = {
    # 四大区域：东北 / 中原 / 南方 / 海岛
    "郑州": "中原",
    "石家庄": "中原",
    "太原": "中原",
    "北京": "中原",
    "沈阳": "东北",
    "长春": "东北",
    "哈尔滨": "东北",
    "广州": "南方",
    "深圳": "南方",
    "福州": "南方",
    "上海": "南方",
    "海南": "海岛",
    "台北": "海岛",
    "高雄": "海岛",
}


@dataclass(frozen=True, slots=True)
class Product:
    id: str
    name: str
    category: ProductCategory
    base_purchase_price: float  # 基础采购价（元/单位）
    profit_margin_rate: float  # 商品溢价率（固定，50%-200%，如0.5表示50%）
    origins: FrozenSet[str]  # 专属产地城市（一城一品）
    specialty_scope: SpecialtyScope = SpecialtyScope.CITY  # 城市特产 / 区域特产
    specialty_region: str | None = None  # 区域特产所属区域（仅 specialty_scope=REGION 时使用）
    perishable_shelf_life_days: int | None = None  # 生鲜保质期（游戏天）
    # λ波动参数
    lambda_min: float = -0.05  # λ下限
    lambda_max: float = 0.05  # λ上限
    lambda_alpha: float = 0.90  # 惯性系数α
    lambda_sigma: float = 0.01  # 扰动标准差σ
    # 运输损耗率（每趟运输的损耗百分比，0.0表示零损耗）
    transport_loss_rate: float = 0.0


def product_display_name(product: Product) -> str:
    """商品显示名：名称 + 属性标签 + 【城】或【区】。

    属性标签：
    - 易损类：【损】
    - 生鲜类：【鲜】
    - 基础民生品：【民】
    - 轻工消费品：【轻】
    """
    if product.category == ProductCategory.ELECTRONICS:
        attr = "【损】"
    elif product.category == ProductCategory.PERISHABLE:
        attr = "【鲜】"
    elif product.category == ProductCategory.BASE:
        attr = "【民】"
    elif product.category == ProductCategory.LIGHT_INDUSTRY:
        attr = "【轻】"
    else:
        attr = "【普】"

    if product.specialty_scope == SpecialtyScope.CITY:
        return f"{product.name}{attr}【城】"
    return f"{product.name}{attr}【区】"


def _fs(items: Iterable[str]) -> FrozenSet[str]:
    return frozenset(items)


# --- Cities (14个城市，一城一品) ---
# 根据新策划案：仅陆运、陆+海运、仅海运三类
CITIES: dict[str, City] = {
    # 中原地区（仅陆运）
    "郑州": City("郑州", frozenset({TransportMode.LAND}), has_bank=True, has_port=False, lat=34.7466, lon=113.6254),
    "石家庄": City("石家庄", frozenset({TransportMode.LAND}), has_bank=True, has_port=False, lat=38.0428, lon=114.5149),
    "太原": City("太原", frozenset({TransportMode.LAND}), has_bank=True, has_port=False, lat=37.8706, lon=112.5489),
    # 东北地区（仅陆运）
    "沈阳": City("沈阳", frozenset({TransportMode.LAND}), has_bank=True, has_port=False, lat=41.8057, lon=123.4315),
    "长春": City("长春", frozenset({TransportMode.LAND}), has_bank=True, has_port=False, lat=43.8171, lon=125.3235),
    "哈尔滨": City("哈尔滨", frozenset({TransportMode.LAND}), has_bank=True, has_port=False, lat=45.8038, lon=126.5349),
    # 经济中心（仅陆运）
    "北京": City("北京", frozenset({TransportMode.LAND}), has_bank=True, has_port=False, lat=39.9042, lon=116.4074),
    # 南方商贸圈（陆+海运）
    "广州": City("广州", frozenset({TransportMode.LAND, TransportMode.SEA}), has_bank=True, has_port=True, lat=23.1291, lon=113.2644),
    "深圳": City("深圳", frozenset({TransportMode.LAND, TransportMode.SEA}), has_bank=True, has_port=True, lat=22.5431, lon=114.0579),
    "福州": City("福州", frozenset({TransportMode.LAND, TransportMode.SEA}), has_bank=True, has_port=True, lat=26.0745, lon=119.2965),
    "上海": City("上海", frozenset({TransportMode.LAND, TransportMode.SEA}), has_bank=True, has_port=True, lat=31.2304, lon=121.4737),
    # 海岛地区（仅海运，无银行）
    "海南": City("海南", frozenset({TransportMode.SEA}), has_bank=False, has_port=True, lat=20.0200, lon=110.3300),  # 海口/三亚合并为"海南"
    "台北": City("台北", frozenset({TransportMode.SEA}), has_bank=False, has_port=True, lat=25.0330, lon=121.5654),
    "高雄": City("高雄", frozenset({TransportMode.SEA}), has_bank=False, has_port=True, lat=22.6273, lon=120.3014),
}

# 高消费区城市（K高消=1.5）
HIGH_CONSUMPTION_CITIES: FrozenSet[str] = frozenset({"北京", "上海", "广州", "深圳", "台北", "高雄"})

# --- Products ---
# 一城一品 + 地区专属特产
PRODUCTS: dict[str, Product] = {
    # 基础民生品（零滞销，轻微损耗）
    "zz_flour": Product(
        id="zz_flour",
        name="郑州特精粉",
        category=ProductCategory.BASE,
        base_purchase_price=2.5,
        profit_margin_rate=0.50,  # 50%
        origins=_fs(["郑州"]),
        lambda_min=-0.05,
        lambda_max=0.05,
        lambda_alpha=0.90,
        lambda_sigma=0.01,
        transport_loss_rate=0.005,  # 0.5%损耗
    ),
    "sjz_antibiotics": Product(
        id="sjz_antibiotics",
        name="石家庄抗生素",
        category=ProductCategory.BASE,
        base_purchase_price=15.0,
        profit_margin_rate=0.60,  # 60%
        origins=_fs(["石家庄"]),
        lambda_min=-0.08,
        lambda_max=0.08,
        lambda_alpha=0.90,
        lambda_sigma=0.02,
        transport_loss_rate=0.005,  # 0.5%损耗
    ),
    "ty_coal": Product(
        id="ty_coal",
        name="太原块煤",
        category=ProductCategory.BASE,
        base_purchase_price=0.8,
        profit_margin_rate=0.80,  # 80%
        origins=_fs(["太原"]),
        lambda_min=-0.10,
        lambda_max=0.10,
        lambda_alpha=0.85,
        lambda_sigma=0.02,
        transport_loss_rate=0.005,  # 0.5%损耗
    ),
    "cc_cornstarch": Product(
        id="cc_cornstarch",
        name="长春淀粉",
        category=ProductCategory.BASE,
        base_purchase_price=3.0,
        profit_margin_rate=0.55,  # 55%
        origins=_fs(["长春"]),
        lambda_min=-0.07,
        lambda_max=0.07,
        lambda_alpha=0.82,
        lambda_sigma=0.02,
        transport_loss_rate=0.005,  # 0.5%损耗
    ),
    "hrb_soybean_oil": Product(
        id="hrb_soybean_oil",
        name="哈尔滨大豆油",
        category=ProductCategory.BASE,
        base_purchase_price=7.5,
        profit_margin_rate=0.70,  # 70%
        origins=_fs(["哈尔滨"]),
        lambda_min=-0.07,
        lambda_max=0.07,
        lambda_alpha=0.80,
        lambda_sigma=0.02,
        transport_loss_rate=0.005,  # 0.5%损耗
    ),
    # 轻工消费品（有滞销阈值，3%损耗）
    "sy_hardware": Product(
        id="sy_hardware",
        name="沈阳五金",
        category=ProductCategory.LIGHT_INDUSTRY,
        base_purchase_price=18.0,
        profit_margin_rate=1.00,  # 100%
        origins=_fs(["沈阳"]),
        lambda_min=-0.19,
        lambda_max=0.19,
        lambda_alpha=0.74,
        lambda_sigma=0.045,
        transport_loss_rate=0.03,  # 3%损耗
    ),
    "gz_clothes": Product(
        id="gz_clothes",
        name="广州休闲服装",
        category=ProductCategory.LIGHT_INDUSTRY,
        base_purchase_price=60.0,
        profit_margin_rate=1.20,  # 120%
        origins=_fs(["广州"]),
        lambda_min=-0.25,
        lambda_max=0.25,
        lambda_alpha=0.70,
        lambda_sigma=0.06,
        transport_loss_rate=0.03,  # 3%损耗
    ),
    "sh_enamel": Product(
        id="sh_enamel",
        name="上海搪瓷",
        category=ProductCategory.ELECTRONICS,
        base_purchase_price=25.0,
        profit_margin_rate=0.90,  # 90%，略降
        origins=_fs(["上海"]),
        lambda_min=-0.22,
        lambda_max=0.22,
        lambda_alpha=0.73,
        lambda_sigma=0.05,
        transport_loss_rate=0.05,  # 5%损耗（易损）
    ),
    # 易损商品
    "sz_pager": Product(
        id="sz_pager",
        name="深圳传呼机",
        category=ProductCategory.ELECTRONICS,
        base_purchase_price=80.0,
        profit_margin_rate=1.20,  # 120%，略降
        origins=_fs(["深圳"]),
        lambda_min=-0.20,
        lambda_max=0.20,
        lambda_alpha=0.75,
        lambda_sigma=0.05,
        transport_loss_rate=0.05,  # 5%损耗（易损）
    ),
    "bj_computer": Product(
        id="bj_computer",
        name="北京电脑",
        category=ProductCategory.ELECTRONICS,
        base_purchase_price=3500.0,
        profit_margin_rate=1.50,  # 150%，略降
        origins=_fs(["北京"]),
        lambda_min=-0.25,
        lambda_max=0.25,
        lambda_alpha=0.80,
        lambda_sigma=0.06,
        transport_loss_rate=0.05,  # 5%损耗
    ),
    # 生鲜特产
    "fz_fish": Product(
        id="fz_fish",
        name="福州带鱼",
        category=ProductCategory.PERISHABLE,
        base_purchase_price=10.0,
        profit_margin_rate=1.30,  # 130%，略升
        origins=_fs(["福州"]),
        perishable_shelf_life_days=5,
        lambda_min=-0.35,
        lambda_max=0.35,
        lambda_alpha=0.40,
        lambda_sigma=0.12,
        transport_loss_rate=0.08,  # 8%损耗（冷链陆运）
    ),
    "hn_mango": Product(
        id="hn_mango",
        name="海南芒果",
        category=ProductCategory.PERISHABLE,
        base_purchase_price=8.0,
        profit_margin_rate=1.60,  # 160%，略升
        origins=_fs(["海南"]),
        perishable_shelf_life_days=6,
        lambda_min=-0.32,
        lambda_max=0.32,
        lambda_alpha=0.45,
        lambda_sigma=0.11,
        transport_loss_rate=0.05,  # 5%损耗（仅冷链海运）
    ),
    "tp_tea": Product(
        id="tp_tea",
        name="台北高山茶",
        category=ProductCategory.ELECTRONICS,  # 视作高价值易损商品
        base_purchase_price=50.0,
        profit_margin_rate=1.30,  # 130%，略降
        origins=_fs(["台北"]),
        perishable_shelf_life_days=None,  # 无保质期（通过易损运输损耗体现风险）
        lambda_min=-0.30,
        lambda_max=0.30,
        lambda_alpha=0.70,
        lambda_sigma=0.07,
        transport_loss_rate=0.05,  # 5%损耗（易损）+10%关税（在economy中处理）
    ),
    "ks_pineapple": Product(
        id="ks_pineapple",
        name="高雄凤梨酥",
        category=ProductCategory.PERISHABLE,
        base_purchase_price=12.0,
        profit_margin_rate=2.00,  # 200%
        origins=_fs(["高雄"]),
        perishable_shelf_life_days=10,
        lambda_min=-0.40,
        lambda_max=0.40,
        lambda_alpha=0.35,
        lambda_sigma=0.13,
        transport_loss_rate=0.03,  # 3%损耗+10%关税（在economy中处理）
    ),
    # --- 地区专属特产（地区内所有城市均可采购） ---
    # 中原地区特产：中原粮食产品（基础民生品）
    "zy_grain": Product(
        id="zy_grain",
        name="中原粮食",
        category=ProductCategory.BASE,
        base_purchase_price=3.0,
        profit_margin_rate=0.30,  # 30%
        origins=_fs(["郑州", "石家庄", "太原"]),
        specialty_scope=SpecialtyScope.REGION,
        specialty_region="中原",
        lambda_min=-0.06,
        lambda_max=0.06,
        lambda_alpha=0.88,
        lambda_sigma=0.015,
        transport_loss_rate=0.0,  # 零损耗
    ),
    # 东北地区特产：东北野生榛蘑（干货，生鲜特产）
    "db_mushroom": Product(
        id="db_mushroom",
        name="东北榛蘑",
        category=ProductCategory.PERISHABLE,  # 干货但沿用“生鲜特产”分类，无保质期
        base_purchase_price=15.0,
        profit_margin_rate=1.50,  # 150%
        origins=_fs(["沈阳", "长春", "哈尔滨"]),
        specialty_scope=SpecialtyScope.REGION,
        specialty_region="东北",
        perishable_shelf_life_days=None,  # 无保质期（临期机制已移除，仅通过运输损耗体现时效性）
        lambda_min=-0.25,
        lambda_max=0.25,
        lambda_alpha=0.70,
        lambda_sigma=0.06,
        transport_loss_rate=0.01,  # 1% 基础损耗，通过时间乘数体现缓慢变质
    ),
    # 南方商贸圈特产：南方轻工纺织品（轻工消费品）
    "sf_textile": Product(
        id="sf_textile",
        name="南方纺织品",
        category=ProductCategory.LIGHT_INDUSTRY,
        base_purchase_price=20.0,
        profit_margin_rate=0.70,  # 70%
        origins=_fs(["广州", "深圳", "福州"]),
        specialty_scope=SpecialtyScope.REGION,
        specialty_region="南方",
        lambda_min=-0.22,
        lambda_max=0.22,
        lambda_alpha=0.72,
        lambda_sigma=0.05,
        transport_loss_rate=0.03,  # 3%损耗/趟
    ),
    # 海岛地区特产：海岛椰子糖（混合装，生鲜特产加工品）
    "island_candy": Product(
        id="island_candy",
        name="海岛椰子糖",
        category=ProductCategory.PERISHABLE,
        base_purchase_price=8.0,
        profit_margin_rate=1.00,  # 100%
        origins=_fs(["海南", "台北", "高雄"]),
        specialty_scope=SpecialtyScope.REGION,
        specialty_region="海岛",
        perishable_shelf_life_days=15,
        lambda_min=-0.30,
        lambda_max=0.30,
        lambda_alpha=0.55,
        lambda_sigma=0.09,
        transport_loss_rate=0.02,  # 海运损耗2%/趟；台湾额外+1%和10%关税在其他模块处理
    ),
}


# --- 从 CSV 加载配置，并覆盖上面的硬编码默认值 ---

def _get_data_dir() -> Path:
    """获取数据文件目录（支持打包后的情况）"""
    import sys
    if getattr(sys, 'frozen', False):
        # PyInstaller 打包后：数据文件在临时目录
        return Path(sys._MEIPASS) / "trade_game"
    else:
        # 开发模式：使用当前文件所在目录
        return Path(__file__).resolve().parent

BASE_DIR = _get_data_dir()
CITIES_CSV = BASE_DIR / "cities.csv"
PRODUCTS_CSV = BASE_DIR / "products.csv"


def _parse_bool(value: str) -> bool:
    return value.strip() in {"1", "true", "True", "yes", "Y"}


def _load_cities_from_csv() -> tuple[dict[str, City], dict[str, str], FrozenSet[str]]:
    if not CITIES_CSV.exists():
        return CITIES, CITY_REGIONS, HIGH_CONSUMPTION_CITIES

    cities: dict[str, City] = {}
    regions: dict[str, str] = {}
    high_consumption: set[str] = set()

    with CITIES_CSV.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"].strip()
            region = row.get("region", "").strip()
            modes_str = row.get("modes", "land").strip()
            if modes_str == "land":
                modes = frozenset({TransportMode.LAND})
            elif modes_str == "sea":
                modes = frozenset({TransportMode.SEA})
            else:
                modes = frozenset({TransportMode.LAND, TransportMode.SEA})
            has_bank = _parse_bool(row.get("has_bank", "0"))
            has_port = _parse_bool(row.get("has_port", "0"))
            lat = float(row.get("lat", "0") or 0.0)
            lon = float(row.get("lon", "0") or 0.0)
            cities[name] = City(
                name=name,
                modes=modes,
                has_bank=has_bank,
                has_port=has_port,
                lat=lat,
                lon=lon,
            )
            if region:
                regions[name] = region
            if _parse_bool(row.get("is_high_consumption", "0")):
                high_consumption.add(name)

    return cities, regions, frozenset(high_consumption)


def _load_products_from_csv() -> dict[str, Product]:
    if not PRODUCTS_CSV.exists():
        return PRODUCTS

    products: dict[str, Product] = {}
    with PRODUCTS_CSV.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row["id"].strip()
            name = row["name"].strip()
            category = ProductCategory(row["category"].strip())
            base_purchase_price = float(row.get("base_purchase_price", "0") or 0.0)
            profit_margin_rate = float(row.get("profit_margin_rate", "0") or 0.0)
            origins_raw = row.get("origins", "").strip()
            origins_list = [s for s in origins_raw.split(";") if s] if origins_raw else []
            specialty_scope_str = row.get("specialty_scope", "").strip() or "city"
            specialty_scope = SpecialtyScope(specialty_scope_str)
            specialty_region = row.get("specialty_region", "").strip() or None
            shelf_raw = row.get("perishable_shelf_life_days", "").strip()
            perishable_shelf_life_days = int(shelf_raw) if shelf_raw else None
            lambda_min = float(row.get("lambda_min", "0") or 0.0)
            lambda_max = float(row.get("lambda_max", "0") or 0.0)
            lambda_alpha = float(row.get("lambda_alpha", "0") or 0.0)
            lambda_sigma = float(row.get("lambda_sigma", "0") or 0.0)
            transport_loss_rate = float(row.get("transport_loss_rate", "0") or 0.0)

            products[pid] = Product(
                id=pid,
                name=name,
                category=category,
                base_purchase_price=base_purchase_price,
                profit_margin_rate=profit_margin_rate,
                origins=_fs(origins_list),
                specialty_scope=specialty_scope,
                specialty_region=specialty_region,
                perishable_shelf_life_days=perishable_shelf_life_days,
                lambda_min=lambda_min,
                lambda_max=lambda_max,
                lambda_alpha=lambda_alpha,
                lambda_sigma=lambda_sigma,
                transport_loss_rate=transport_loss_rate,
            )

    return products


# 覆盖硬编码表，使运行时完全从 CSV 读取数据
CITIES, CITY_REGIONS, HIGH_CONSUMPTION_CITIES = _load_cities_from_csv()
PRODUCTS = _load_products_from_csv()

