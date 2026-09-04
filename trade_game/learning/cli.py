"""原生 PPO 训练命令的终端输出。"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from .imitation import (
    DAggerPPOTrainingConfig,
    load_dagger_ppo_config,
    train_dagger_ppo,
    train_dagger_bc,
)
from .train import (
    PPOTrainingConfig,
    load_training_config,
    save_model_checkpoint,
    train_ppo,
    train_ppo_finetune,
)


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


def run_dagger_ppo(
    *,
    config_path: Path | None,
    updates: int | None,
    rollout_steps: int | None,
    expert_episodes: int | None,
    dagger_rounds: int | None,
    checkpoint_path: Path | None,
    tensorboard_log_dir: Path | None,
) -> None:
    """运行贪心教师、DAgger、BC 和 PPO，并输出每个阶段的关键结果。"""

    config = (
        load_dagger_ppo_config(config_path)
        if config_path
        else DAggerPPOTrainingConfig()
    )
    ppo_config = config.ppo
    dagger_config = config.dagger
    if updates is not None:
        ppo_config = replace(ppo_config, updates=updates)
    if rollout_steps is not None:
        ppo_config = replace(ppo_config, ppo=replace(ppo_config.ppo, rollout_steps=rollout_steps))
    if checkpoint_path is not None:
        ppo_config = replace(ppo_config, checkpoint_path=checkpoint_path)
    if tensorboard_log_dir is not None:
        ppo_config = replace(ppo_config, tensorboard_log_dir=tensorboard_log_dir)
    if expert_episodes is not None:
        dagger_config = replace(dagger_config, expert_episodes=expert_episodes)
    if dagger_rounds is not None:
        dagger_config = replace(dagger_config, rounds=dagger_rounds)
    config = replace(config, ppo=ppo_config, dagger=dagger_config)

    result = train_dagger_ppo(config)
    initial = result.dagger.initial_behavior_cloning
    initial_eval = result.dagger.initial_evaluation
    print(
        "BC 初始阶段："
        f"examples={initial.examples} loss={initial.loss:.4f} "
        f"action_acc={initial.action_accuracy:.2%} "
        f"mean_assets={initial_eval.mean_final_assets}"
    )
    for round_metrics in result.dagger.rounds:
        bc = round_metrics.behavior_cloning
        print(
            f"DAgger 第 {round_metrics.round_index} 轮："
            f"beta={round_metrics.beta:.2f} added={round_metrics.added_examples} "
            f"dataset={round_metrics.dataset_size} "
            f"learner_acc={round_metrics.learner_action_accuracy:.2%} "
            f"bc_loss={bc.loss:.4f} mean_assets={round_metrics.evaluation.mean_final_assets}"
        )
    for snapshot in result.ppo.snapshots:
        metrics = snapshot.metrics
        line = (
            f"PPO 更新 {snapshot.update}: steps={snapshot.environment_steps} "
            f"policy_loss={metrics.policy_loss:.4f} value_loss={metrics.value_loss:.4f} "
            f"entropy={metrics.entropy:.4f} kl={metrics.approx_kl:.5f}"
        )
        if snapshot.evaluation is not None:
            line += f" mean_assets={snapshot.evaluation.mean_final_assets}"
        print(line)


def run_dagger_bc(
    *,
    config_path: Path | None,
    expert_episodes: int | None,
    dagger_rounds: int | None,
    checkpoint_path: Path | None,
    tensorboard_log_dir: Path | None,
) -> None:
    """独立运行 DAgger+BC，并保存 BC 模型。"""

    config = load_dagger_ppo_config(config_path) if config_path else DAggerPPOTrainingConfig()
    ppo_config = config.ppo
    dagger_config = config.dagger
    if expert_episodes is not None:
        dagger_config = replace(dagger_config, expert_episodes=expert_episodes)
    if dagger_rounds is not None:
        dagger_config = replace(dagger_config, rounds=dagger_rounds)
    if tensorboard_log_dir is not None:
        ppo_config = replace(ppo_config, tensorboard_log_dir=tensorboard_log_dir)
    config = replace(config, ppo=ppo_config, dagger=dagger_config)
    result = train_dagger_bc(config)
    if checkpoint_path is not None:
        save_model_checkpoint(checkpoint_path, result.model, config=ppo_config)
        print(f"BC 模型已保存：{checkpoint_path}")
    initial = result.initial_behavior_cloning
    print(
        "DAgger+BC 完成："
        f" dataset={result.dataset_size} examples={initial.examples} "
        f"loss={initial.loss:.4f} action_acc={initial.action_accuracy:.2%}"
    )


def run_ppo_finetune(
    *,
    model_path: Path,
    config_path: Path | None,
    checkpoint_path: Path | None,
    tensorboard_log_dir: Path | None,
    updates: int | None,
    rollout_steps: int | None,
) -> None:
    """加载已有模型并继续执行 PPO。"""

    if checkpoint_path is not None and checkpoint_path.resolve() == model_path.resolve():
        raise ValueError("finetune 输出 checkpoint 不能覆盖输入模型")
    config = load_training_config(config_path) if config_path else PPOTrainingConfig()
    if updates is not None:
        config = replace(config, updates=updates)
    if rollout_steps is not None:
        config = replace(config, ppo=replace(config.ppo, rollout_steps=rollout_steps))
    if checkpoint_path is not None:
        config = replace(config, checkpoint_path=checkpoint_path)
    if tensorboard_log_dir is not None:
        config = replace(config, tensorboard_log_dir=tensorboard_log_dir)
    result = train_ppo_finetune(config, model_path)
    for snapshot in result.snapshots:
        metrics = snapshot.metrics
        print(
            f"PPO finetune 更新 {snapshot.update}: steps={snapshot.environment_steps} "
            f"policy_loss={metrics.policy_loss:.4f} value_loss={metrics.value_loss:.4f} "
            f"entropy={metrics.entropy:.4f} kl={metrics.approx_kl:.5f}"
        )


__all__ = ["run_dagger_bc", "run_dagger_ppo", "run_ppo_finetune", "run_training"]
