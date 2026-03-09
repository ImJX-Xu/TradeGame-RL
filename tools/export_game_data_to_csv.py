from __future__ import annotations

"""
一次性工具：从当前硬编码的 trade_game.data 中导出城市 / 商品配置到 CSV。

运行方式（在项目根目录）：

    python -m tools.export_game_data_to_csv

生成：
    trade_game/cities.csv
    trade_game/products.csv
"""

import csv
from pathlib import Path

from trade_game.data import CITIES, CITY_REGIONS, HIGH_CONSUMPTION_CITIES, PRODUCTS


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "trade_game"


def export_cities() -> None:
    path = OUT_DIR / "cities.csv"
    fieldnames = [
        "name",
        "region",
        "modes",  # "land", "sea", "land+sea"
        "has_bank",
        "has_port",
        "lat",
        "lon",
        "is_high_consumption",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for name, city in CITIES.items():
            modes = city.modes
            if modes == {"land"}:
                mode_str = "land"
            elif modes == {"sea"}:
                mode_str = "sea"
            else:
                mode_str = "land+sea"
            writer.writerow(
                {
                    "name": name,
                    "region": CITY_REGIONS.get(name, ""),
                    "modes": mode_str,
                    "has_bank": int(bool(city.has_bank)),
                    "has_port": int(bool(city.has_port)),
                    "lat": city.lat,
                    "lon": city.lon,
                    "is_high_consumption": int(name in HIGH_CONSUMPTION_CITIES),
                }
            )
    print(f"Written {path}")


def export_products() -> None:
    path = OUT_DIR / "products.csv"
    fieldnames = [
        "id",
        "name",
        "category",
        "base_purchase_price",
        "profit_margin_rate",
        "origins",  # 以 ; 分隔的城市名
        "specialty_scope",
        "specialty_region",
        "perishable_shelf_life_days",
        "lambda_min",
        "lambda_max",
        "lambda_alpha",
        "lambda_sigma",
        "stale_threshold",
        "transport_loss_rate",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for pid, prod in PRODUCTS.items():
            writer.writerow(
                {
                    "id": prod.id,
                    "name": prod.name,
                    "category": prod.category.value,
                    "base_purchase_price": prod.base_purchase_price,
                    "profit_margin_rate": prod.profit_margin_rate,
                    "origins": ";".join(sorted(prod.origins)),
                    "specialty_scope": getattr(prod, "specialty_scope", None).value
                    if getattr(prod, "specialty_scope", None)
                    else "",
                    "specialty_region": getattr(prod, "specialty_region", "") or "",
                    "perishable_shelf_life_days": (
                        "" if prod.perishable_shelf_life_days is None else prod.perishable_shelf_life_days
                    ),
                    "lambda_min": prod.lambda_min,
                    "lambda_max": prod.lambda_max,
                    "lambda_alpha": prod.lambda_alpha,
                    "lambda_sigma": prod.lambda_sigma,
                    "stale_threshold": "" if prod.stale_threshold is None else prod.stale_threshold,
                    "transport_loss_rate": prod.transport_loss_rate,
                }
            )
    print(f"Written {path}")


def main() -> None:
    export_cities()
    export_products()


if __name__ == "__main__":
    main()

