"""原生 PPO 的轨迹记录与按游戏日折扣的 GAE 计算。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from trade_game.agent import ActionHead, ActionMask, AgentObservation

from .policy import ActionBatch


@dataclass(frozen=True, slots=True)
class RolloutStep:
    """一个动作决策时刻的完整训练记录。"""

    observation: AgentObservation
    action_mask: ActionMask
    action: ActionHead
    log_prob: float
    value: float
    reward: float
    terminated: bool
    final_assets: float | None
    elapsed_days: int
    environment_step: int


@dataclass(frozen=True, slots=True)
class RolloutBatch:
    """已计算回报和优势的 PPO 更新批次。"""

    observations: tuple[AgentObservation, ...]
    action_masks: tuple[ActionMask, ...]
    actions: ActionBatch
    old_log_probs: Tensor
    old_values: Tensor
    rewards: Tensor
    terminated: Tensor
    final_assets: tuple[float | None, ...]
    head_entropies: Tensor
    elapsed_days: Tensor
    environment_steps: Tensor
    advantages: Tensor
    returns: Tensor

    @property
    def size(self) -> int:
        return len(self.observations)

    @classmethod
    def concatenate(cls, batches: tuple["RolloutBatch", ...]) -> "RolloutBatch":
        """合并多个独立环境的完整轨迹，保留各轨迹内部的时间顺序。"""

        if not batches:
            raise ValueError("不能合并空轨迹集合")
        return cls(
            observations=tuple(
                observation for batch in batches for observation in batch.observations
            ),
            action_masks=tuple(action_mask for batch in batches for action_mask in batch.action_masks),
            actions=ActionBatch.from_tensor(
                torch.cat(tuple(batch.actions.as_tensor() for batch in batches), dim=0)
            ),
            old_log_probs=torch.cat(tuple(batch.old_log_probs for batch in batches), dim=0),
            old_values=torch.cat(tuple(batch.old_values for batch in batches), dim=0),
            rewards=torch.cat(tuple(batch.rewards for batch in batches), dim=0),
            terminated=torch.cat(tuple(batch.terminated for batch in batches), dim=0),
            final_assets=tuple(
                final_assets for batch in batches for final_assets in batch.final_assets
            ),
            head_entropies=torch.cat(tuple(batch.head_entropies for batch in batches), dim=0),
            elapsed_days=torch.cat(tuple(batch.elapsed_days for batch in batches), dim=0),
            environment_steps=torch.cat(
                tuple(batch.environment_steps for batch in batches), dim=0
            ),
            advantages=torch.cat(tuple(batch.advantages for batch in batches), dim=0),
            returns=torch.cat(tuple(batch.returns for batch in batches), dim=0),
        )


class RolloutBuffer:
    """以 Python 不可变观测保存交互，再在 PPO 更新前完成 GAE。"""

    def __init__(self) -> None:
        self._steps: list[RolloutStep] = []
        self._head_entropies: list[Tensor] = []

    def __len__(self) -> int:
        return len(self._steps)

    def append(
        self,
        *,
        observation: AgentObservation,
        action_mask: ActionMask,
        action: ActionHead,
        log_prob: float,
        value: float,
        reward: float,
        terminated: bool,
        final_assets: float | None,
        head_entropies: Tensor,
        elapsed_days: int,
        environment_step: int,
    ) -> None:
        """记录一次由当前策略采样并提交给游戏核心的转移。"""

        self._steps.append(
            RolloutStep(
                observation=observation,
                action_mask=action_mask,
                action=action,
                log_prob=log_prob,
                value=value,
                reward=reward,
                terminated=terminated,
                final_assets=final_assets,
                elapsed_days=elapsed_days,
                environment_step=environment_step,
            )
        )
        self._head_entropies.append(head_entropies.detach())

    def finish(self, *, next_value: float, gamma: float, gae_lambda: float) -> RolloutBatch:
        """使用实际经过天数计算 TD 残差、GAE 优势和回报。"""

        if not self._steps:
            raise ValueError("不能从空轨迹计算 GAE")
        advantages = [0.0] * len(self._steps)
        next_advantage = 0.0
        for index in range(len(self._steps) - 1, -1, -1):
            step = self._steps[index]
            continuation = 0.0 if step.terminated else 1.0
            discount = gamma**step.elapsed_days
            bootstrap_value = next_value if index == len(self._steps) - 1 else self._steps[index + 1].value
            delta = step.reward + discount * continuation * bootstrap_value - step.value
            next_advantage = delta + discount * gae_lambda * continuation * next_advantage
            advantages[index] = next_advantage
        actions = ActionBatch.from_actions(tuple(step.action for step in self._steps))
        old_values = torch.tensor([step.value for step in self._steps], dtype=torch.float32)
        advantages_tensor = torch.tensor(advantages, dtype=torch.float32)
        return RolloutBatch(
            observations=tuple(step.observation for step in self._steps),
            action_masks=tuple(step.action_mask for step in self._steps),
            actions=actions,
            old_log_probs=torch.tensor([step.log_prob for step in self._steps], dtype=torch.float32),
            old_values=old_values,
            rewards=torch.tensor([step.reward for step in self._steps], dtype=torch.float32),
            terminated=torch.tensor([step.terminated for step in self._steps], dtype=torch.bool),
            final_assets=tuple(step.final_assets for step in self._steps),
            head_entropies=torch.stack(self._head_entropies).to(dtype=torch.float32, device="cpu"),
            elapsed_days=torch.tensor([step.elapsed_days for step in self._steps], dtype=torch.long),
            environment_steps=torch.tensor(
                [step.environment_step for step in self._steps], dtype=torch.long
            ),
            advantages=advantages_tensor,
            returns=advantages_tensor + old_values,
        )


__all__ = ["RolloutBatch", "RolloutBuffer", "RolloutStep"]
