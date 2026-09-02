"""以智能体动作协议驱动核心游戏会话的训练回合接口。"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from trade_game.core import GameMode, GameSession, create_game_session, settlement_assets

from .actions import ActionHead, ActionVocabulary
from .decoder import decode_action
from .masks import ActionMask, build_action_mask
from .observation import AgentObservation, ObservationConfig, build_observation
from .rewards import RewardBreakdown, RewardV1


@dataclass(frozen=True, slots=True)
class EpisodeStart:
    """重置后可直接交给策略网络的初始观测与动作掩码。"""

    observation: AgentObservation
    action_mask: ActionMask
    initial_assets: Decimal


@dataclass(frozen=True, slots=True)
class EpisodeTransition:
    """一次智能体决策后的下一状态、奖励与终局标记。"""

    observation: AgentObservation
    action_mask: ActionMask
    reward: float
    terminated: bool
    final_assets: Decimal | None
    elapsed_days: int
    reward_breakdown: RewardBreakdown


class AgentEnvironment:
    """训练侧的 ``reset/step`` 包装，游戏规则仍由 ``GameSession`` 独占。"""

    def __init__(
        self,
        *,
        mode: GameMode = GameMode.CHALLENGE,
        reward: RewardV1 | None = None,
        observation_config: ObservationConfig | None = None,
    ) -> None:
        self.mode = mode
        self.reward = reward or RewardV1()
        self.observation_config = observation_config
        self.session: GameSession | None = None
        self.vocabulary: ActionVocabulary | None = None
        self.initial_assets: Decimal | None = None

    def reset(self, *, seed: int | None = None, mode: GameMode | None = None) -> EpisodeStart:
        """创建一局新游戏并返回首个智能体观测。"""

        if mode is not None:
            self.mode = mode
        self.session = create_game_session(seed=seed, mode=self.mode)
        self.vocabulary = ActionVocabulary.from_catalog(self.session.catalog)
        self.initial_assets = settlement_assets(self.session.catalog, self.session.rules, self.session.state)
        return EpisodeStart(
            observation=build_observation(
                self.session, self.vocabulary, config=self.observation_config
            ),
            action_mask=build_action_mask(self.session, self.vocabulary),
            initial_assets=self.initial_assets,
        )

    def step(self, action: ActionHead) -> EpisodeTransition:
        """执行一个合法动作，并返回以实际经过天数标记的训练转移。"""

        if self.session is None or self.vocabulary is None or self.initial_assets is None:
            raise RuntimeError("请先调用 reset")
        before = self.session.state
        if before.outcome is not None:
            raise RuntimeError("当前回合已经结束")
        result = self.session.dispatch(decode_action(self.session, action, self.vocabulary))
        if not result.accepted:
            raise RuntimeError("智能体动作被核心规则拒绝")
        after = self.session.state
        reward_breakdown = self.reward.evaluate(
            self.session.catalog,
            self.session.rules,
            before,
            after,
            initial_assets=self.initial_assets,
        )
        return EpisodeTransition(
            observation=build_observation(
                self.session, self.vocabulary, config=self.observation_config
            ),
            action_mask=build_action_mask(self.session, self.vocabulary),
            reward=reward_breakdown.reward,
            terminated=after.outcome is not None,
            final_assets=after.outcome.final_assets if after.outcome is not None else None,
            elapsed_days=after.day - before.day,
            reward_breakdown=reward_breakdown,
        )


__all__ = ["AgentEnvironment", "EpisodeStart", "EpisodeTransition"]
