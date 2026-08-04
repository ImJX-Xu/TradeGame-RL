"""原生 PPO 训练命令的终端输出。"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from .train import PPOTrainingConfig, load_training_config, train_ppo


def run_training(
    *,
    config_path: Path | None,
    updates: int | None,
    rollout_steps: int | None,
    checkpoint_path: Path | None,
    tensorboard_log_dir: Path | None,
) -> None:
    """加载默认或 TOML 配置，执行训练并输出每次更新的主要指标。"""

    config = load_training_config(config_path) if config_path else PPOTrainingConfig()
    if updates is not None:
        config = replace(config, updates=updates)
    if rollout_steps is not None:
        config = replace(config, ppo=replace(config.ppo, rollout_steps=rollout_steps))
    if checkpoint_path is not None:
        config = replace(config, checkpoint_path=checkpoint_path)
    if tensorboard_log_dir is not None:
        config = replace(config, tensorboard_log_dir=tensorboard_log_dir)
    result = train_ppo(config)
    for snapshot in result.snapshots:
        metrics = snapshot.metrics
        line = (
            f"更新 {snapshot.update}: steps={snapshot.environment_steps} "
            f"policy_loss={metrics.policy_loss:.4f} value_loss={metrics.value_loss:.4f} "
            f"entropy={metrics.entropy:.4f} kl={metrics.approx_kl:.5f}"
        )
        if snapshot.evaluation is not None:
            line += f" mean_assets={snapshot.evaluation.mean_final_assets}"
        print(line)


__all__ = ["run_training"]
