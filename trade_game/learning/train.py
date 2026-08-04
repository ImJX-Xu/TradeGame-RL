"""PPO 默认配置、训练入口和检查点保存。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import tomllib

import torch

from trade_game.agent import AgentEnvironment

from .batching import ObservationSpec
from .encoder import StateEncoderConfig
from .evaluate import EvaluationSummary, evaluate_policy
from .ppo import PPOConfig, PPOTrainer, PPOUpdateMetrics
from .policy import ActorCritic


@dataclass(frozen=True, slots=True)
class PPOTrainingConfig:
    """一次完整 PPO 训练运行的配置。"""

    updates: int = 100
    seed: int = 7
    device: str = "cpu"
    evaluation_interval: int = 10
    evaluation_seeds: tuple[int, ...] = (101, 103, 107, 109, 113)
    checkpoint_path: Path | None = None
    ppo: PPOConfig = field(default_factory=PPOConfig)
    encoder: StateEncoderConfig = field(
        default_factory=lambda: StateEncoderConfig(
            embedding_dim=16,
            entity_dim=64,
            state_dim=128,
            hidden_dim=128,
        )
    )


@dataclass(frozen=True, slots=True)
class TrainingSnapshot:
    """单次 PPO 更新后的训练和评估指标。"""

    update: int
    environment_steps: int
    metrics: PPOUpdateMetrics
    evaluation: EvaluationSummary | None


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """训练完成后返回模型和逐次更新记录。"""

    model: ActorCritic
    snapshots: tuple[TrainingSnapshot, ...]


def load_training_config(path: Path) -> PPOTrainingConfig:
    """从 TOML 文件读取训练、PPO 与编码器参数。"""

    with path.open("rb") as source:
        document = tomllib.load(source)
    training_values = dict(document.get("training", {}))
    ppo_values = dict(document.get("ppo", {}))
    encoder_values = dict(document.get("encoder", {}))
    if "checkpoint_path" in training_values:
        training_values["checkpoint_path"] = Path(training_values["checkpoint_path"])
    if "evaluation_seeds" in training_values:
        training_values["evaluation_seeds"] = tuple(training_values["evaluation_seeds"])
    return PPOTrainingConfig(
        **training_values,
        ppo=PPOConfig(**ppo_values),
        encoder=StateEncoderConfig(**encoder_values),
    )


def train_ppo(config: PPOTrainingConfig) -> TrainingResult:
    """执行配置指定次数的采样、PPO 更新和周期性确定性评估。"""

    torch.manual_seed(config.seed)
    environment = AgentEnvironment()
    start = environment.reset(seed=config.seed)
    model = ActorCritic(
        ObservationSpec.from_observation(start.observation),
        encoder_config=config.encoder,
    )
    trainer = PPOTrainer(
        model,
        environment,
        config.ppo,
        device=config.device,
        seed=config.seed,
    )
    snapshots: list[TrainingSnapshot] = []
    for update in range(1, config.updates + 1):
        rollout = trainer.collect_rollout()
        metrics = trainer.update(rollout)
        evaluation = (
            evaluate_policy(model, seeds=config.evaluation_seeds, device=config.device)
            if update % config.evaluation_interval == 0 or update == config.updates
            else None
        )
        snapshots.append(
            TrainingSnapshot(
                update=update,
                environment_steps=trainer.environment_steps,
                metrics=metrics,
                evaluation=evaluation,
            )
        )
    result = TrainingResult(model=model, snapshots=tuple(snapshots))
    if config.checkpoint_path is not None:
        save_checkpoint(config.checkpoint_path, result, config, trainer)
    return result


def save_checkpoint(
    path: Path,
    result: TrainingResult,
    config: PPOTrainingConfig,
    trainer: PPOTrainer,
) -> None:
    """保存模型、优化器、训练配置和已记录指标。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": result.model.state_dict(),
            "optimizer": trainer.optimizer.state_dict(),
            "observation_spec": result.model.encoder.spec,
            "training_config": asdict(config),
            "snapshots": result.snapshots,
        },
        path,
    )


__all__ = [
    "PPOTrainingConfig",
    "TrainingResult",
    "TrainingSnapshot",
    "load_training_config",
    "save_checkpoint",
    "train_ppo",
]
