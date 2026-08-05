"""原生 PyTorch PPO 的采样、裁剪目标和参数更新。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

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
    """协调多个独立游戏环境的采样、GAE 与 PPO 更新。"""

    def __init__(
        self,
        model: ActorCritic,
        environment: AgentEnvironment | Sequence[AgentEnvironment],
        config: PPOConfig,
        *,
        device: torch.device | str = "cpu",
        seed: int | None = None,
    ) -> None:
        self.model = model.to(device)
        self.environments = (
            (environment,) if isinstance(environment, AgentEnvironment) else tuple(environment)
        )
        if not self.environments:
            raise ValueError("至少需要一个训练环境")
        self.environment = self.environments[0]
        self.config = config
        self.device = torch.device(device)
        self.seed = seed
        self.optimizer = Adam(self.model.parameters(), lr=config.learning_rate)
        self._observations: list[AgentObservation | None] = [None] * len(self.environments)
        self._action_masks: list[ActionMask | None] = [None] * len(self.environments)
        self._episode_indices = [0] * len(self.environments)
        self.environment_steps = 0

    @property
    def environment_count(self) -> int:
        """返回本训练器批量采样的独立游戏环境数量。"""

        return len(self.environments)

    def collect_rollout(self) -> RolloutBatch:
        """用当前策略采集固定数量的决策转移并计算 GAE。"""

        self._ensure_episodes()
        buffers = tuple(RolloutBuffer() for _ in self.environments)
        self.model.eval()
        for _ in range(self.config.rollout_steps):
            observations, action_masks = self._current_states()
            batch = ObservationBatch.from_observations(
                observations,
                spec=self.model.encoder.spec,
                device=self.device,
            )
            masks = ActionMaskBatch.from_masks(action_masks, device=self.device)
            with torch.no_grad():
                sample = self.model.sample(batch, masks)
            actions = sample.action.as_tensor()
            head_entropies = sample.head_entropies.as_tensor()
            for index, environment in enumerate(self.environments):
                action = _action_head(actions[index])
                transition = environment.step(action)
                self.environment_steps += 1
                buffers[index].append(
                    observation=observations[index],
                    action_mask=action_masks[index],
                    action=action,
                    log_prob=float(sample.log_prob[index].item()),
                    value=float(sample.value[index].item()),
                    reward=transition.reward,
                    terminated=transition.terminated,
                    final_assets=(
                        float(transition.final_assets) if transition.final_assets is not None else None
                    ),
                    head_entropies=head_entropies[index],
                    elapsed_days=transition.elapsed_days,
                    environment_step=self.environment_steps,
                )
                self._observations[index] = transition.observation
                self._action_masks[index] = transition.action_mask
                if transition.terminated:
                    self._start_episode(index)
        next_values = self._current_values()
        return RolloutBatch.concatenate(
            tuple(
                buffer.finish(
                    next_value=next_values[index],
                    gamma=self.config.gamma,
                    gae_lambda=self.config.gae_lambda,
                )
                for index, buffer in enumerate(buffers)
            )
        )

    def update(self, rollout: RolloutBatch) -> PPOUpdateMetrics:
        """对已固定的采样轨迹执行多轮裁剪 PPO 更新。"""

        self.model.train()
        advantages = rollout.advantages.to(self.device)
        if self.config.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        observations = ObservationBatch.from_observations(
            rollout.observations,
            spec=self.model.encoder.spec,
            device=self.device,
        )
        action_masks = ActionMaskBatch.from_masks(
            rollout.action_masks,
            device=self.device,
        )
        actions = rollout.actions.to(self.device)
        old_log_probs = rollout.old_log_probs.to(self.device)
        old_values = rollout.old_values.to(self.device)
        returns = rollout.returns.to(self.device)

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
            order = torch.randperm(rollout.size, device=self.device)
            for start in range(0, rollout.size, self.config.minibatch_size):
                indices = order[start : start + self.config.minibatch_size]
                metrics = self._update_minibatch(
                    observations.index_select(indices),
                    action_masks.index_select(indices),
                    actions.index_select(indices),
                    old_log_probs.index_select(0, indices),
                    old_values.index_select(0, indices),
                    returns.index_select(0, indices),
                    advantages.index_select(0, indices),
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
        observations: ObservationBatch,
        action_masks: ActionMaskBatch,
        actions: ActionBatch,
        old_log_probs: Tensor,
        old_values: Tensor,
        returns: Tensor,
        advantages: Tensor,
    ) -> tuple[float, float, float, float, float, float, float]:
        evaluation = self.model.evaluate_actions(observations, action_masks, actions)
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

    def _ensure_episodes(self) -> None:
        for index, (observation, action_mask) in enumerate(
            zip(self._observations, self._action_masks, strict=True)
        ):
            if observation is None or action_mask is None:
                self._start_episode(index)

    def _start_episode(self, index: int) -> None:
        episode_seed = (
            None
            if self.seed is None
            else self.seed + index * 1_000_003 + self._episode_indices[index]
        )
        start = self.environments[index].reset(seed=episode_seed)
        self._observations[index] = start.observation
        self._action_masks[index] = start.action_mask
        self._episode_indices[index] += 1

    def _current_states(self) -> tuple[tuple[AgentObservation, ...], tuple[ActionMask, ...]]:
        observations = tuple(self._observations)
        action_masks = tuple(self._action_masks)
        if any(observation is None for observation in observations) or any(
            action_mask is None for action_mask in action_masks
        ):
            raise RuntimeError("训练回合尚未初始化")
        return (
            tuple(observation for observation in observations if observation is not None),
            tuple(action_mask for action_mask in action_masks if action_mask is not None),
        )

    def _current_values(self) -> tuple[float, ...]:
        observations, _ = self._current_states()
        batch = ObservationBatch.from_observations(
            observations,
            spec=self.model.encoder.spec,
            device=self.device,
        )
        with torch.no_grad():
            return tuple(float(value.item()) for value in self.model(batch).value)


def _action_head(values: Tensor) -> ActionHead:
    return ActionHead(*(int(value) for value in values.detach().cpu().tolist()))


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


__all__ = ["PPOConfig", "PPOTrainer", "PPOUpdateMetrics"]
