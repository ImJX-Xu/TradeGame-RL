from __future__ import annotations

"""
从 CSV 加载城市 / 商品配置，构建运行时使用的 CITIES / PRODUCTS 等数据结构。
"""

import csv
from pathlib import Path
from typing import Dict, Tuple

from .data import (
    CITY_REGIONS,
    HIGH_CONSUMPTION_CITIES,
    CITIES,
    PRODUCTS,
)

# 注意：此文件目前只是占位，实际加载逻辑已经直接嵌入 data.py 中。
# 留空以便未来如需拆分 loader，可以在此实现。

