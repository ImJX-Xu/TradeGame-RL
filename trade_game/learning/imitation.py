"""贪心教师、行为克隆和 DAgger 训练流程。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from random import Random
import tomllib
from typing import Sequence

import torch
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from trade_game.agent import (
    ActionHead,
    ActionMask,
    AgentEnvironment,
    AgentObservation,
    ObservationConfig,
    encode_command,
)
from trade_game.analysis import GreedyPolicy

from .batching import ActionMaskBatch, ObservationBatch, ObservationSpec
from .encoder import StateEncoderConfig
from .evaluate import EvaluationSummary, evaluate_policy
from .policy import ActionBatch, ActorCritic
from .ppo import PPOConfig
from .train import PPOTrainingConfig, TrainingResult, train_ppo


@dataclass(frozen=True, slots=True)
class ImitationExample:
    """一个状态快照及贪心教师给出的动作标签。"""

    observation: AgentObservation
    action_mask: ActionMask
    action: ActionHead


class DemonstrationBuffer:
    """保存 DAgger 聚合数据集，不复制或修改游戏状态。"""

    def __init__(self) -> None:
        self._examples: list[ImitationExample] = []

    def __len__(self) -> int:
        return len(self._examples)

    @property
    def examples(self) -> tuple[ImitationExample, ...]:
        return tuple(self._examples)

    def extend(self, examples: Sequence[ImitationExample]) -> None:
        items = tuple(examples)
        if not items:
            return
        self._examples.extend(items)


@dataclass(frozen=True, slots=True)
class BehaviorCloningConfig:
    """行为克隆的监督优化参数。"""

    initial_epochs: int = 8
    round_epochs: int = 4
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    max_grad_norm: float = 1.0


@dataclass(frozen=True, slots=True)
class DAggerConfig:
    """DAgger 的教师数据量和教师强制概率。"""

    expert_episodes: int = 16
    rounds: int = 2
    episodes_per_round: int = 8
    beta_start: float = 0.5
    beta_final: float = 0.0
    seed_offset: int = 100_000
    max_decisions_per_episode: int = 512


@dataclass(frozen=True, slots=True)
class BehaviorCloningMetrics:
    """一次行为克隆阶段的聚合指标。"""

    epochs: int
    examples: int
    loss: float
    action_accuracy: float
    action_type_accuracy: float
    grad_norm: float


@dataclass(frozen=True, slots=True)
class DAggerRoundMetrics:
    """一次 DAgger 状态访问、标签聚合和再训练的结果。"""

    round_index: int
    beta: float
    added_examples: int
    dataset_size: int
    learner_action_accuracy: float
    behavior_cloning: BehaviorCloningMetrics
    evaluation: EvaluationSummary


@dataclass(frozen=True, slots=True)
class DAggerTrainingResult:
    """DAgger 阶段的模型和可复现实验记录。"""

    model: ActorCritic
    dataset_size: int
    initial_behavior_cloning: BehaviorCloningMetrics
    initial_evaluation: EvaluationSummary
    rounds: tuple[DAggerRoundMetrics, ...]


@dataclass(frozen=True, slots=True)
class DAggerPPOTrainingConfig:
    """完整的贪心教师、DAgger、BC 和 PPO 配置。"""

    ppo: PPOTrainingConfig = field(default_factory=PPOTrainingConfig)
    dagger: DAggerConfig = field(default_factory=DAggerConfig)
    behavior_cloning: BehaviorCloningConfig = field(default_factory=BehaviorCloningConfig)


@dataclass(frozen=True, slots=True)
class DAggerPPOTrainingResult:
    """完整 DAgger-PPO 运行的中间结果和最终 PPO 结果。"""

    dagger: DAggerTrainingResult
    ppo: TrainingResult


@dataclass(frozen=True, slots=True)
class _CollectionResult:
    examples: tuple[ImitationExample, ...]
    learner_action_accuracy: float


def load_dagger_ppo_config(path: Path) -> DAggerPPOTrainingConfig:
    """从 TOML 文件读取 DAgger、BC 和 PPO 配置。"""

    with path.open("rb") as source:
        document = tomllib.load(source)
    training_values = dict(document.get("training", {}))
    ppo_values = dict(document.get("ppo", {}))
    encoder_values = dict(document.get("encoder", {}))
    observation_values = dict(document.get("observation", {}))
    if "checkpoint_path" in training_values:
        training_values["checkpoint_path"] = Path(training_values["checkpoint_path"])
    if "tensorboard_log_dir" in training_values:
        training_values["tensorboard_log_dir"] = Path(training_values["tensorboard_log_dir"])
    if "evaluation_seeds" in training_values:
        training_values["evaluation_seeds"] = tuple(training_values["evaluation_seeds"])
    if "market_history_offsets" in observation_values:
        observation_values["market_history_offsets"] = tuple(
            observation_values["market_history_offsets"]
        )
    ppo_training = PPOTrainingConfig(
        **training_values,
        ppo=PPOConfig(**ppo_values),
        encoder=StateEncoderConfig(**encoder_values),
        observation=ObservationConfig(**observation_values),
    )
    return DAggerPPOTrainingConfig(
        ppo=ppo_training,
        dagger=DAggerConfig(**dict(document.get("dagger", {}))),
        behavior_cloning=BehaviorCloningConfig(**dict(document.get("bc", {}))),
    )


def train_behavior_cloning(
    model: ActorCritic,
    dataset: DemonstrationBuffer,
    config: BehaviorCloningConfig,
    *,
    epochs: int,
    device: torch.device | str,
) -> BehaviorCloningMetrics:
    """用教师动作的联合对数概率训练策略和共享编码器。"""

    if len(dataset) == 0:
        raise ValueError("行为克隆数据集不能为空")
    if epochs <= 0 or config.batch_size <= 0:
        raise ValueError("行为克隆的 epochs 和 batch_size 必须为正")

    model = model.to(device)
    optimizer = Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    examples = dataset.examples
    losses: list[float] = []
    action_accuracies: list[float] = []
    action_type_accuracies: list[float] = []
    grad_norms: list[float] = []
    was_training = model.training
    model.train()
    try:
        for _ in range(epochs):
            order = torch.randperm(len(examples))
            for start in range(0, len(examples), config.batch_size):
                indices = order[start : start + config.batch_size].tolist()
                batch_examples = tuple(examples[index] for index in indices)
                observations = ObservationBatch.from_observations(
                    tuple(example.observation for example in batch_examples),
                    spec=model.encoder.spec,
                    device=device,
                )
                masks = ActionMaskBatch.from_masks(
                    tuple(example.action_mask for example in batch_examples),
                    device=device,
                )
                targets = ActionBatch.from_actions(
                    tuple(example.action for example in batch_examples),
                    device=device,
                )
                evaluation = model.evaluate_actions(observations, masks, targets)
                loss = -evaluation.log_prob.mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()

                with torch.no_grad():
                    predictions = model.sample(observations, masks, deterministic=True).action
                    target_values = targets.as_tensor()
                    predicted_values = predictions.as_tensor()
                    action_accuracies.append(
                        float(
                            predicted_values.eq(target_values).all(dim=1).float().mean().item()
                        )
                    )
                    action_type_accuracies.append(
                        float(
                            predicted_values[:, 0]
                            .eq(target_values[:, 0])
                            .float()
                            .mean()
                            .item()
                        )
                    )
                losses.append(float(loss.detach()))
                grad_norms.append(float(grad_norm))
    finally:
        model.train(was_training)
    return BehaviorCloningMetrics(
        epochs=epochs,
        examples=len(examples),
        loss=_mean(losses),
        action_accuracy=_mean(action_accuracies),
        action_type_accuracy=_mean(action_type_accuracies),
        grad_norm=_mean(grad_norms),
    )


def train_dagger_bc(config: DAggerPPOTrainingConfig) -> DAggerTrainingResult:
    """独立执行贪心教师、DAgger 和行为克隆，返回 BC 模型。"""

    ppo_config = config.ppo
    if config.dagger.expert_episodes <= 0:
        raise ValueError("expert_episodes 必须为正")
    if config.dagger.rounds < 0 or config.dagger.episodes_per_round <= 0:
        raise ValueError("DAgger rounds 和 episodes_per_round 不合法")
    torch.manual_seed(ppo_config.seed)
    environment = AgentEnvironment(observation_config=ppo_config.observation)
    start = environment.reset(seed=ppo_config.seed)
    model = ActorCritic(
        ObservationSpec.from_observation(start.observation),
        encoder_config=ppo_config.encoder,
    ).to(ppo_config.device)
    teacher = GreedyPolicy()
    dataset = DemonstrationBuffer()
    writer = (
        SummaryWriter(log_dir=str(ppo_config.tensorboard_log_dir))
        if ppo_config.tensorboard_log_dir is not None
        else None
    )
    try:
        expert_seeds = _episode_seeds(
            ppo_config.seed + config.dagger.seed_offset,
            config.dagger.expert_episodes,
        )
        expert_collection = _collect_examples(
            teacher,
            dataset,
            seeds=expert_seeds,
            model=None,
            beta=1.0,
            device=ppo_config.device,
            max_decisions=config.dagger.max_decisions_per_episode,
            observation_config=ppo_config.observation,
        )
        _log_collection(writer, 0, dataset_size=len(dataset), collection=expert_collection)
        initial_bc = train_behavior_cloning(
            model,
            dataset,
            config.behavior_cloning,
            epochs=config.behavior_cloning.initial_epochs,
            device=ppo_config.device,
        )
        initial_evaluation = evaluate_policy(
            model,
            seeds=ppo_config.evaluation_seeds,
            device=ppo_config.device,
            capture_trace=False,
        )
        _log_bc(writer, 0, initial_bc, initial_evaluation)

        rounds: list[DAggerRoundMetrics] = []
        for round_index in range(config.dagger.rounds):
            beta = _interpolate_beta(
                config.dagger.beta_start,
                config.dagger.beta_final,
                round_index / max(config.dagger.rounds - 1, 1),
            )
            round_seeds = _episode_seeds(
                ppo_config.seed + config.dagger.seed_offset + 10_000 * (round_index + 1),
                config.dagger.episodes_per_round,
            )
            before = len(dataset)
            collection = _collect_examples(
                teacher,
                dataset,
                seeds=round_seeds,
                model=model,
                beta=beta,
                device=ppo_config.device,
                max_decisions=config.dagger.max_decisions_per_episode,
                observation_config=ppo_config.observation,
            )
            bc_metrics = train_behavior_cloning(
                model,
                dataset,
                config.behavior_cloning,
                epochs=config.behavior_cloning.round_epochs,
                device=ppo_config.device,
            )
            evaluation = evaluate_policy(
                model,
                seeds=ppo_config.evaluation_seeds,
                device=ppo_config.device,
                capture_trace=False,
            )
            metrics = DAggerRoundMetrics(
                round_index=round_index + 1,
                beta=beta,
                added_examples=len(dataset) - before,
                dataset_size=len(dataset),
                learner_action_accuracy=collection.learner_action_accuracy,
                behavior_cloning=bc_metrics,
                evaluation=evaluation,
            )
            rounds.append(metrics)
            _log_collection(
                writer,
                round_index + 1,
                dataset_size=len(dataset),
                collection=collection,
                beta=beta,
            )
            _log_bc(writer, round_index + 1, bc_metrics, evaluation)
            if writer is not None:
                writer.flush()
        if writer is not None:
            writer.flush()
    finally:
        if writer is not None:
            writer.close()

    return DAggerTrainingResult(
        model=model,
        dataset_size=len(dataset),
        initial_behavior_cloning=initial_bc,
        initial_evaluation=initial_evaluation,
        rounds=tuple(rounds),
    )


def train_dagger_ppo(config: DAggerPPOTrainingConfig) -> DAggerPPOTrainingResult:
    """兼容旧入口：执行 DAgger+BC 后继续 PPO。"""

    dagger_result = train_dagger_bc(config)
    ppo_result = train_ppo(config.ppo, initial_model=dagger_result.model)
    return DAggerPPOTrainingResult(dagger=dagger_result, ppo=ppo_result)


def _collect_examples(
    teacher: GreedyPolicy,
    dataset: DemonstrationBuffer,
    *,
    seeds: tuple[int, ...],
    model: ActorCritic | None,
    beta: float,
    device: torch.device | str,
    max_decisions: int,
    observation_config: ObservationConfig,
) -> _CollectionResult:
    if not 0.0 <= beta <= 1.0:
        raise ValueError("DAgger beta 必须位于 [0, 1]")
    if max_decisions <= 0:
        raise ValueError("max_decisions 必须为正")
    chooser = Random(sum(seeds) + int(beta * 10_000))
    learner_matches = 0
    learner_decisions = 0
    collected: list[ImitationExample] = []
    was_training = model.training if model is not None else False
    if model is not None:
        model.eval()
    try:
        for seed in seeds:
            environment = AgentEnvironment(observation_config=observation_config)
            start = environment.reset(seed=seed)
            observation = start.observation
            action_mask = start.action_mask
            for _ in range(max_decisions):
                if environment.session is None or environment.vocabulary is None:
                    raise RuntimeError("DAgger 回合尚未初始化")
                expert_command = teacher.choose(environment.session)
                expert_action = encode_command(
                    environment.session,
                    expert_command,
                    environment.vocabulary,
                )
                learner_action = expert_action
                if model is not None:
                    batch = ObservationBatch.from_observations(
                        (observation,),
                        spec=model.encoder.spec,
                        device=device,
                    )
                    masks = ActionMaskBatch.from_masks((action_mask,), device=device)
                    with torch.no_grad():
                        learner_action = _action_from_tensor(
                            model.sample(batch, masks, deterministic=True).action.as_tensor()[0]
                        )
                    learner_decisions += 1
                    learner_matches += int(learner_action.as_tuple() == expert_action.as_tuple())
                collected.append(
                    ImitationExample(
                        observation=observation,
                        action_mask=action_mask,
                        action=expert_action,
                    )
                )
                chosen_action = (
                    expert_action
                    if model is None or chooser.random() < beta
                    else learner_action
                )
                transition = environment.step(chosen_action)
                observation = transition.observation
                action_mask = transition.action_mask
                if transition.terminated:
                    break
            else:
                raise RuntimeError("DAgger 回合超过最大决策数")
    finally:
        if model is not None:
            model.train(was_training)
    dataset.extend(collected)
    return _CollectionResult(
        examples=tuple(collected),
        learner_action_accuracy=(
            learner_matches / learner_decisions if learner_decisions else 1.0
        ),
    )


def _action_from_tensor(values: Tensor) -> ActionHead:
    return ActionHead(*(int(value) for value in values.detach().cpu().tolist()))


def _episode_seeds(start: int, count: int) -> tuple[int, ...]:
    return tuple(start + index for index in range(count))


def _interpolate_beta(start: float, end: float, progress: float) -> float:
    return start + (end - start) * min(max(progress, 0.0), 1.0)


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("指标序列不能为空")
    return sum(values) / len(values)


def _log_collection(
    writer: SummaryWriter | None,
    step: int,
    *,
    dataset_size: int,
    collection: _CollectionResult,
    beta: float | None = None,
) -> None:
    if writer is None:
        return
    writer.add_scalar("imitation/dataset_size", dataset_size, step)
    writer.add_scalar(
        "imitation/learner_action_accuracy",
        collection.learner_action_accuracy,
        step,
    )
    if beta is not None:
        writer.add_scalar("imitation/beta", beta, step)


def _log_bc(
    writer: SummaryWriter | None,
    step: int,
    metrics: BehaviorCloningMetrics,
    evaluation: EvaluationSummary,
) -> None:
    if writer is None:
        return
    writer.add_scalar("imitation/bc_loss", metrics.loss, step)
    writer.add_scalar("imitation/bc_action_accuracy", metrics.action_accuracy, step)
    writer.add_scalar(
        "imitation/bc_action_type_accuracy",
        metrics.action_type_accuracy,
        step,
    )
    writer.add_scalar("imitation/bc_grad_norm", metrics.grad_norm, step)
    writer.add_scalar("imitation/evaluation_final_assets_mean", float(evaluation.mean_final_assets), step)
    writer.add_scalar("imitation/evaluation_bankruptcy_rate", evaluation.bankruptcy_rate, step)


__all__ = [
    "BehaviorCloningConfig",
    "BehaviorCloningMetrics",
    "DAggerConfig",
    "DAggerPPOTrainingConfig",
    "DAggerPPOTrainingResult",
    "DAggerRoundMetrics",
    "DAggerTrainingResult",
    "DemonstrationBuffer",
    "ImitationExample",
    "load_dagger_ppo_config",
    "train_behavior_cloning",
    "train_dagger_bc",
    "train_dagger_ppo",
]
