"""原生 PyTorch PPO 的采样、裁剪目标和参数更新。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam

from trade_game.agent import ActionHead, ActionMask, AgentEnvironment, AgentObservation

from .batching import ActionMaskBatch, ObservationBatch
from .policy import ActionBatch, ActorCritic
from .rollout import RolloutBatch, RolloutBuffer


@dataclass(frozen=True, slots=True)
class PPOConfig:
    """PPO 采样与优化的默认超参数。"""

    gamma: float = 0.995
    gae_lambda: float = 0.95
    rollout_steps: int = 512
    ppo_epochs: int = 4
    minibatch_size: int = 64
    learning_rate: float = 3e-4
    clip_range: float = 0.2
    value_clip_range: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    max_grad_norm: float = 0.5
    target_kl: float = 0.03
    normalize_advantages: bool = True


@dataclass(frozen=True, slots=True)
class PPOUpdateMetrics:
    """一次 PPO 更新的平均损失与策略健康指标。"""

    policy_loss: float
    value_loss: float
    entropy: float
    total_loss: float
    approx_kl: float
    clip_fraction: float
    grad_norm: float
    minibatches: int
    epochs: int
    early_stopped: bool


class PPOTrainer:
    """串联游戏采样、GAE 和 PPO 更新的单环境训练器。"""

    def __init__(
        self,
        model: ActorCritic,
        environment: AgentEnvironment,
        config: PPOConfig,
        *,
        device: torch.device | str = "cpu",
        seed: int | None = None,
    ) -> None:
        self.model = model.to(device)
        self.environment = environment
        self.config = config
        self.device = torch.device(device)
        self.seed = seed
        self.optimizer = Adam(self.model.parameters(), lr=config.learning_rate)
        self._observation: AgentObservation | None = None
        self._action_mask: ActionMask | None = None
        self._episode_index = 0
        self.environment_steps = 0

    def collect_rollout(self) -> RolloutBatch:
        """用当前策略采集固定数量的决策转移并计算 GAE。"""

        self._ensure_episode()
        buffer = RolloutBuffer()
        self.model.eval()
        for _ in range(self.config.rollout_steps):
            observation, action_mask = self._current_state()
            batch = ObservationBatch.from_observations(
                (observation,),
                spec=self.model.encoder.spec,
                device=self.device,
            )
            masks = ActionMaskBatch.from_masks((action_mask,), device=self.device)
            with torch.no_grad():
                sample = self.model.sample(batch, masks)
            action = _action_head(sample.action)
            transition = self.environment.step(action)
            buffer.append(
                observation=observation,
                action_mask=action_mask,
                action=action,
                log_prob=float(sample.log_prob.item()),
                value=float(sample.value.item()),
                reward=transition.reward,
                terminated=transition.terminated,
                final_assets=(
                    float(transition.final_assets) if transition.final_assets is not None else None
                ),
                head_entropies=sample.head_entropies.as_tensor()[0],
                elapsed_days=transition.elapsed_days,
            )
            self.environment_steps += 1
            self._observation = transition.observation
            self._action_mask = transition.action_mask
            if transition.terminated:
                self._start_episode()
        next_value = self._current_value()
        return buffer.finish(
            next_value=next_value,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
        )

    def update(self, rollout: RolloutBatch) -> PPOUpdateMetrics:
        """对已固定的采样轨迹执行多轮裁剪 PPO 更新。"""

        self.model.train()
        advantages = rollout.advantages.to(self.device)
        if self.config.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        policy_losses: list[float] = []
        value_losses: list[float] = []
        entropies: list[float] = []
        total_losses: list[float] = []
        kls: list[float] = []
        clip_fractions: list[float] = []
        grad_norms: list[float] = []
        completed_epochs = 0
        early_stopped = False
        for _ in range(self.config.ppo_epochs):
            completed_epochs += 1
            order = torch.randperm(rollout.size)
            for start in range(0, rollout.size, self.config.minibatch_size):
                indices = order[start : start + self.config.minibatch_size]
                metrics = self._update_minibatch(
                    rollout,
                    indices,
                    advantages[indices.to(self.device)],
                )
                policy_losses.append(metrics[0])
                value_losses.append(metrics[1])
                entropies.append(metrics[2])
                total_losses.append(metrics[3])
                kls.append(metrics[4])
                clip_fractions.append(metrics[5])
                grad_norms.append(metrics[6])
                if metrics[4] > self.config.target_kl:
                    early_stopped = True
                    break
            if early_stopped:
                break
        return PPOUpdateMetrics(
            policy_loss=_mean(policy_losses),
            value_loss=_mean(value_losses),
            entropy=_mean(entropies),
            total_loss=_mean(total_losses),
            approx_kl=_mean(kls),
            clip_fraction=_mean(clip_fractions),
            grad_norm=_mean(grad_norms),
            minibatches=len(policy_losses),
            epochs=completed_epochs,
            early_stopped=early_stopped,
        )

    def _update_minibatch(
        self,
        rollout: RolloutBatch,
        indices: Tensor,
        advantages: Tensor,
    ) -> tuple[float, float, float, float, float, float, float]:
        index_list = indices.tolist()
        batch = ObservationBatch.from_observations(
            tuple(rollout.observations[index] for index in index_list),
            spec=self.model.encoder.spec,
            device=self.device,
        )
        masks = ActionMaskBatch.from_masks(
            tuple(rollout.action_masks[index] for index in index_list),
            device=self.device,
        )
        actions = ActionBatch.from_tensor(rollout.actions.as_tensor()[indices].to(self.device))
        old_log_probs = rollout.old_log_probs[indices].to(self.device)
        old_values = rollout.old_values[indices].to(self.device)
        returns = rollout.returns[indices].to(self.device)
        evaluation = self.model.evaluate_actions(batch, masks, actions)
        log_ratio = evaluation.log_prob - old_log_probs
        ratio = log_ratio.exp()
        unclipped_objective = ratio * advantages
        clipped_objective = ratio.clamp(
            1.0 - self.config.clip_range,
            1.0 + self.config.clip_range,
        ) * advantages
        policy_loss = -torch.minimum(unclipped_objective, clipped_objective).mean()

        value_delta = evaluation.value - old_values
        clipped_values = old_values + value_delta.clamp(
            -self.config.value_clip_range,
            self.config.value_clip_range,
        )
        value_loss = 0.5 * torch.maximum(
            (evaluation.value - returns).square(),
            (clipped_values - returns).square(),
        ).mean()
        entropy = evaluation.entropy.mean()
        total_loss = (
            policy_loss
            + self.config.value_coefficient * value_loss
            - self.config.entropy_coefficient * entropy
        )
        self.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        grad_norm = clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        self.optimizer.step()

        approx_kl = ((ratio - 1.0) - log_ratio).mean()
        clip_fraction = (ratio.sub(1.0).abs() > self.config.clip_range).float().mean()
        return (
            float(policy_loss.detach()),
            float(value_loss.detach()),
            float(entropy.detach()),
            float(total_loss.detach()),
            float(approx_kl.detach()),
            float(clip_fraction.detach()),
            float(grad_norm),
        )

    def _ensure_episode(self) -> None:
        if self._observation is None or self._action_mask is None:
            self._start_episode()

    def _start_episode(self) -> None:
        episode_seed = None if self.seed is None else self.seed + self._episode_index
        start = self.environment.reset(seed=episode_seed)
        self._observation = start.observation
        self._action_mask = start.action_mask
        self._episode_index += 1

    def _current_state(self) -> tuple[AgentObservation, ActionMask]:
        if self._observation is None or self._action_mask is None:
            raise RuntimeError("训练回合尚未初始化")
        return self._observation, self._action_mask

    def _current_value(self) -> float:
        observation, _ = self._current_state()
        batch = ObservationBatch.from_observations(
            (observation,),
            spec=self.model.encoder.spec,
            device=self.device,
        )
        with torch.no_grad():
            return float(self.model(batch).value.item())


def _action_head(actions: ActionBatch) -> ActionHead:
    return ActionHead(*(int(value) for value in actions.as_tensor()[0]))


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


__all__ = ["PPOConfig", "PPOTrainer", "PPOUpdateMetrics"]
