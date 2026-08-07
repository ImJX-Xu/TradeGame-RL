"""使用确定性策略游玩挑战回合并汇总评估指标。"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

import torch

from trade_game.agent import ActionHead, AgentEnvironment
from trade_game.core import GameEndReason

from .batching import ActionMaskBatch, ObservationBatch
from .policy import ActorCritic


@dataclass(frozen=True, slots=True)
class PolicyTraceStep:
    """评估游玩轨迹中的一个动作与其经营结果。"""

    day: int
    action: ActionHead
    reward: float
    assets_after: Decimal


@dataclass(frozen=True, slots=True)
class PolicyEpisode:
    """一局确定性策略游玩的终局摘要与完整动作轨迹。"""

    seed: int
    final_assets: Decimal
    total_reward: float
    elapsed_days: int
    end_reason: GameEndReason
    trace: tuple[PolicyTraceStep, ...]


@dataclass(frozen=True, slots=True)
class EvaluationSummary:
    """多个固定随机种子上的策略评估结果。"""

    episodes: tuple[PolicyEpisode, ...]
    mean_final_assets: Decimal
    median_final_assets: Decimal
    mean_total_reward: float
    mean_elapsed_days: float
    bankruptcy_rate: float


def play_policy(
    model: ActorCritic,
    *,
    seed: int,
    device: torch.device | str = "cpu",
    capture_trace: bool = True,
) -> PolicyEpisode:
    """以确定性条件策略游玩一局挑战模式，并保留完整动作轨迹。"""

    model = model.to(device)
    environment = AgentEnvironment()
    start = environment.reset(seed=seed)
    observation = start.observation
    action_mask = start.action_mask
    trace: list[PolicyTraceStep] = []
    total_reward = 0.0
    total_days = 0
    was_training = model.training
    model.eval()
    try:
        while True:
            batch = ObservationBatch.from_observations(
                (observation,),
                spec=model.encoder.spec,
                device=device,
            )
            masks = ActionMaskBatch.from_masks((action_mask,), device=device)
            with torch.no_grad():
                sample = model.sample(batch, masks, deterministic=True)
            action = ActionHead(*(int(value) for value in sample.action.as_tensor()[0]))
            if environment.session is None:
                raise RuntimeError("评估回合尚未初始化")
            day = environment.session.state.day
            transition = environment.step(action)
            if capture_trace:
                trace.append(
                    PolicyTraceStep(
                        day=day,
                        action=action,
                        reward=transition.reward,
                        assets_after=transition.reward_breakdown.assets_after,
                    )
                )
            total_reward += transition.reward
            total_days += transition.elapsed_days
            observation = transition.observation
            action_mask = transition.action_mask
            if transition.terminated:
                break
    finally:
        model.train(was_training)
    if environment.session is None or environment.session.state.outcome is None:
        raise RuntimeError("挑战回合没有形成终局结果")
    outcome = environment.session.state.outcome
    return PolicyEpisode(
        seed=seed,
        final_assets=outcome.final_assets,
        total_reward=total_reward,
        elapsed_days=total_days,
        end_reason=outcome.reason,
        trace=tuple(trace),
    )


def evaluate_policy(
    model: ActorCritic,
    *,
    seeds: tuple[int, ...],
    device: torch.device | str = "cpu",
    capture_trace: bool = True,
) -> EvaluationSummary:
    """在固定种子集合上执行确定性挑战评估。"""

    episodes = tuple(
        play_policy(model, seed=seed, device=device, capture_trace=capture_trace)
        for seed in seeds
    )
    final_assets = tuple(sorted(episode.final_assets for episode in episodes))
    count = len(episodes)
    if not count:
        raise ValueError("评估种子不能为空")
    midpoint = count // 2
    median = (
        final_assets[midpoint]
        if count % 2
        else (final_assets[midpoint - 1] + final_assets[midpoint]) / Decimal("2")
    )
    return EvaluationSummary(
        episodes=episodes,
        mean_final_assets=sum(final_assets, start=Decimal("0")) / count,
        median_final_assets=median,
        mean_total_reward=sum(episode.total_reward for episode in episodes) / count,
        mean_elapsed_days=sum(episode.elapsed_days for episode in episodes) / count,
        bankruptcy_rate=sum(
            episode.end_reason is GameEndReason.BANKRUPTCY for episode in episodes
        )
        / count,
    )


__all__ = [
    "EvaluationSummary",
    "PolicyEpisode",
    "PolicyTraceStep",
    "evaluate_policy",
    "play_policy",
]
