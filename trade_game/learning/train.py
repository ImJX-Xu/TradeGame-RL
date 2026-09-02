"""PPO 默认配置、训练入口和检查点保存。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import tomllib

import torch

from trade_game.agent import AgentEnvironment, ObservationConfig

from .batching import ObservationSpec
from .encoder import StateEncoderConfig
from .evaluate import EvaluationSummary, evaluate_policy
from .ppo import PPOConfig, PPOTrainer, PPOUpdateMetrics
from .policy import ActorCritic
from .tensorboard import TensorBoardLogger


@dataclass(frozen=True, slots=True)
class PPOTrainingConfig:
    """一次完整 PPO 训练运行的配置。"""

    updates: int = 100
    seed: int = 7
    device: str = "cpu"
    environment_count: int = 1
    evaluation_interval: int = 10
    evaluation_seeds: tuple[int, ...] = (101, 103, 107, 109, 113)
    tensorboard_flush_interval: int = 10
    checkpoint_path: Path | None = None
    tensorboard_log_dir: Path | None = Path("runs/tensorboard")
    observation: ObservationConfig = field(default_factory=ObservationConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    encoder: StateEncoderConfig = field(
        default_factory=lambda: StateEncoderConfig(
            row_dim=64,
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
    return PPOTrainingConfig(
        **training_values,
        ppo=PPOConfig(**ppo_values),
        encoder=StateEncoderConfig(**encoder_values),
        observation=ObservationConfig(**observation_values),
    )


def train_ppo(
    config: PPOTrainingConfig,
    *,
    initial_model: ActorCritic | None = None,
) -> TrainingResult:
    """执行 PPO；可从行为克隆或 DAgger 产生的模型继续微调。"""

    if config.environment_count <= 0:
        raise ValueError("训练环境数量必须为正")
    torch.manual_seed(config.seed)
    environments = tuple(
        AgentEnvironment(observation_config=config.observation)
        for _ in range(config.environment_count)
    )
    start = environments[0].reset(seed=config.seed)
    spec = ObservationSpec.from_observation(start.observation)
    model = initial_model if initial_model is not None else ActorCritic(spec, encoder_config=config.encoder)
    if model.encoder.spec != spec:
        raise ValueError("初始模型的观测规格与当前游戏目录不一致")
    trainer = PPOTrainer(
        model,
        environments,
        config.ppo,
        device=config.device,
        seed=config.seed,
    )
    snapshots: list[TrainingSnapshot] = []
    logger = (
        TensorBoardLogger(config.tensorboard_log_dir, config)
        if config.tensorboard_log_dir is not None
        else None
    )
    try:
        for update in range(1, config.updates + 1):
            trainer.set_training_progress((update - 1) / max(config.updates - 1, 1))
            rollout = trainer.collect_rollout()
            metrics = trainer.update(rollout)
            evaluation = (
                evaluate_policy(
                    model,
                    seeds=config.evaluation_seeds,
                    device=config.device,
                    capture_trace=False,
                )
                if update % config.evaluation_interval == 0 or update == config.updates
                else None
            )
            snapshot = TrainingSnapshot(
                update=update,
                environment_steps=trainer.environment_steps,
                metrics=metrics,
                evaluation=evaluation,
            )
            snapshots.append(snapshot)
            if logger is not None:
                logger.log_rollout(rollout, environment_steps=trainer.environment_steps)
                logger.log_update(metrics, environment_steps=trainer.environment_steps)
                if evaluation is not None:
                    logger.log_evaluation(evaluation, environment_steps=trainer.environment_steps)
                if (
                    update % config.tensorboard_flush_interval == 0
                    or evaluation is not None
                    or update == config.updates
                ):
                    logger.flush()
            del rollout
    finally:
        if logger is not None:
            logger.close()
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
