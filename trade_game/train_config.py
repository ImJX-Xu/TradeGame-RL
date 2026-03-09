"""
训练与 CLI 规则配置

与 game_config 分离：本文件管理「训练环境、CLI、录制模式」共用规则，
如采购/售出数量档位、借贷/还款比例、结算与评分、RL 奖励塑形等。
正常游戏平衡参数（运输成本、车辆、价格波动等）见 game_config。
"""

from __future__ import annotations

from typing import Tuple

# ============================================================================
# 一、采购/售出数量档位（CLI + 训练 + 录制模式 GUI 对齐）
# ============================================================================
# 市场采购、售卖时的数量规则：
# - 变体档位：按“当前上限”（可买上限 / 可卖上限）的 1/5、2/5、3/5、4/5、5/5
# - 固定档位：1、2、3
# 训练动作空间与录制模式下的 GUI 均使用同一套 8 档规则（动态数量，随上限变化）。
QTY_FRACTIONS: Tuple[float, ...] = (1.0 / 5.0, 2.0 / 5.0, 3.0 / 5.0, 4.0 / 5.0, 1.0)
QTY_FIXED: Tuple[int, ...] = (1, 2, 3)

# ============================================================================
# 二、借贷/还款金额比例（训练 + 录制模式 GUI 对齐）
# ============================================================================
# 按可借额度/可还额度的比例：1/3、2/3、3/3。借贷 actual = fraction * max_loan；还款 3/3 = 全部。
AMOUNT_FRACTIONS: Tuple[float, ...] = (1.0 / 3.0, 2.0 / 3.0, 1.0)

# ============================================================================
# 三、天数上限
# ============================================================================
# 挑战模式时间上限（天）
CHALLENGE_DAYS: int = 365
# 玩家演示/录制模式时间上限（天）
DEMO_MODE_DAYS: int = 90

# ============================================================================
# 四、结算与评分
# ============================================================================
# 结算公式：结算金额 = 现金 + 标准价格卖出货物 + 货车残值 - 应还借贷
# 货车残值：每辆额外货车（货车数-1）按此金额计入
SETTLEMENT_TRUCK_RESIDUAL: float = 40_000.0

# 挑战模式评分阈值：[(资产下限, 称号), ...]，按资产从高到低排列
CHALLENGE_RATING_THRESHOLDS: list[tuple[float, str]] = [
    (1000_000.0, "夯"),
    (500_000.0, "顶级"),
    (100_000.0, "人上人"),
    (10_000.0, "NPC"),
    (0.0, "拉完了"),
]


def compute_challenge_rating(total_assets: float, bankrupt: bool = False) -> str:
    """根据结算金额计算挑战模式评分。破产时一律返回「拉完了」。"""
    if bankrupt or total_assets < 0:
        return "拉完了"
    for threshold, rating in CHALLENGE_RATING_THRESHOLDS:
        if total_assets >= threshold:
            return rating
    return "拉完了"


def compute_settlement_amount(cash: float, cargo_lots: list, truck_count: int, loans: list) -> float:
    """
    结算金额 = 现金 + 按标准价格卖出所有货物 + 货车残值 - 应还借贷
    标准价格 = 基础采购价 × (1 + 溢价率)。
    """
    from .data import PRODUCTS

    goods_value = 0.0
    for lot in cargo_lots:
        prod = PRODUCTS.get(lot.product_id)
        if prod:
            standard_price = prod.base_purchase_price * (1.0 + prod.profit_margin_rate)
            goods_value += standard_price * lot.quantity

    truck_residual = SETTLEMENT_TRUCK_RESIDUAL * max(0, truck_count - 1)
    total_debt = sum(l.debt_total() for l in loans)
    return round(cash + goods_value + truck_residual - total_debt, 2)


def get_max_days(game_mode: str) -> int | None:
    """返回天数上限，None 表示无上限（自由模式）。"""
    if game_mode == "challenge":
        return CHALLENGE_DAYS
    if game_mode == "demo":
        return DEMO_MODE_DAYS
    return None


# ============================================================================
# 五、RL 奖励塑形（跨城/同城卖出）
# ============================================================================
CROSS_CITY_SELL_BONUS: float = 10.0
SAME_CITY_SELL_PENALTY: float = 10.0
