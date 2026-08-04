"""智能体训练使用的经营奖励函数。"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from math import log

from trade_game.core import Catalog, GameEndReason, GameRules, GameState, settlement_assets


@dataclass(frozen=True, slots=True)
class RewardV1Config:
    """以经营资产增长为目标的奖励参数。"""

    asset_floor: Decimal = Decimal("1")
    terminal_asset_weight: float = 0.25
    bankruptcy_penalty: float = 3.0


@dataclass(frozen=True, slots=True)
class RewardBreakdown:
    """单个训练转移的奖励及可用于记录的组成部分。"""

    reward: float
    asset_log_change: float
    terminal_asset_bonus: float
    bankruptcy_penalty: float
    assets_before: Decimal
    assets_after: Decimal


class RewardV1:
    """根据公开可清算资产变化计算密集训练奖励。"""

    def __init__(self, config: RewardV1Config | None = None) -> None:
        self.config = config or RewardV1Config()

    def evaluate(
        self,
        catalog: Catalog,
        rules: GameRules,
        before: GameState,
        after: GameState,
        *,
        initial_assets: Decimal,
    ) -> RewardBreakdown:
        """计算一次成功游戏命令前后的资产势能变化。"""

        assets_before = settlement_assets(catalog, rules, before)
        assets_after = settlement_assets(catalog, rules, after)
        asset_log_change = self._asset_log(assets_after) - self._asset_log(assets_before)
        terminal_asset_bonus = 0.0
        bankruptcy_penalty = 0.0
        if after.outcome is not None:
            if after.outcome.reason is GameEndReason.TIME_LIMIT:
                terminal_asset_bonus = self.config.terminal_asset_weight * (
                    self._asset_log(assets_after) - self._asset_log(initial_assets)
                )
            elif after.outcome.reason is GameEndReason.BANKRUPTCY:
                bankruptcy_penalty = -self.config.bankruptcy_penalty
        reward = asset_log_change + terminal_asset_bonus + bankruptcy_penalty
        return RewardBreakdown(
            reward=reward,
            asset_log_change=asset_log_change,
            terminal_asset_bonus=terminal_asset_bonus,
            bankruptcy_penalty=bankruptcy_penalty,
            assets_before=assets_before,
            assets_after=assets_after,
        )

    def _asset_log(self, assets: Decimal) -> float:
        return log(float(max(assets, self.config.asset_floor)))


__all__ = ["RewardBreakdown", "RewardV1", "RewardV1Config"]
