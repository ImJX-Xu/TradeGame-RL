"""PPO 训练的 TensorBoard 指标写入。"""

from __future__ import annotations

from collections import deque
import json
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from torch.utils.tensorboard import SummaryWriter

from trade_game.agent import ACTION_TYPES
from trade_game.core import CommandType

from .evaluate import EvaluationSummary
from .ppo import PPOUpdateMetrics
from .rollout import RolloutBatch

if TYPE_CHECKING:
    from .train import PPOTrainingConfig


_HEAD_ENTROPY_ACTIONS: tuple[tuple[str, CommandType | None], ...] = (
    ("action", None),
    ("buy", CommandType.BUY),
    ("sell", CommandType.SELL),
    ("travel", CommandType.TRAVEL),
    ("borrow_quantity", CommandType.BORROW),
    ("repay_quantity", CommandType.REPAY),
    ("buy_truck_quantity", CommandType.BUY_TRUCK),
)


class TensorBoardLogger:
    """集中记录一次 PPO 训练运行的标量、配置与模型分布。"""

    def __init__(self, log_dir: Path, config: PPOTrainingConfig) -> None:
        self.writer = SummaryWriter(log_dir=str(log_dir))
        self._recent_final_assets: deque[float] = deque(maxlen=20)
        self.writer.add_text(
            "config/training",
            f"```json\n{json.dumps(_config_values(config), ensure_ascii=False, indent=2)}\n```",
            global_step=0,
        )

    def log_rollout(self, rollout: RolloutBatch, *, environment_steps: int) -> None:
        """记录当前采样段的奖励，以及其中结束回合的最终资产。"""

        self.writer.add_scalar("rollout/reward_mean", rollout.rewards.mean(), environment_steps)
        for action_index, action_type in enumerate(ACTION_TYPES):
            fraction = (rollout.actions.action_index == action_index).float().mean()
            self.writer.add_scalar(
                f"actions/fraction/{action_type.value}",
                fraction,
                environment_steps,
            )
        for entropy_index, (head_name, action_type) in enumerate(_HEAD_ENTROPY_ACTIONS):
            if action_type is None:
                mean_entropy = rollout.head_entropies[:, entropy_index].mean()
            else:
                action_index = ACTION_TYPES.index(action_type)
                selected = rollout.actions.action_index == action_index
                if not bool(selected.any()):
                    continue
                mean_entropy = rollout.head_entropies[selected, entropy_index].mean()
            self.writer.add_scalar(
                f"actions/entropy/{head_name}",
                mean_entropy,
                environment_steps,
            )
        for index, final_assets in enumerate(rollout.final_assets):
            if final_assets is None:
                continue
            terminal_step = int(rollout.environment_steps[index])
            self._recent_final_assets.append(final_assets)
            self.writer.add_scalar("episode/final_assets", final_assets, terminal_step)
            self.writer.add_scalar(
                "episode/final_assets_mean_20",
                sum(self._recent_final_assets) / len(self._recent_final_assets),
                terminal_step,
            )

    def log_update(self, metrics: PPOUpdateMetrics, *, environment_steps: int) -> None:
        """记录判断 PPO 更新是否稳定所需的核心指标。"""

        values = {
            "ppo/policy_loss": metrics.policy_loss,
            "ppo/value_loss": metrics.value_loss,
            "ppo/total_loss": metrics.total_loss,
            "ppo/entropy": metrics.entropy,
            "ppo/approx_kl": metrics.approx_kl,
            "ppo/clip_fraction": metrics.clip_fraction,
            "ppo/grad_norm": metrics.grad_norm,
            "ppo/minibatches": metrics.minibatches,
            "ppo/epochs": metrics.epochs,
            "ppo/early_stopped": float(metrics.early_stopped),
            "schedule/learning_rate": metrics.learning_rate,
            "schedule/entropy_coefficient": metrics.entropy_coefficient,
            "schedule/target_kl": metrics.target_kl,
        }
        for name, value in values.items():
            self.writer.add_scalar(name, value, environment_steps)

    def log_evaluation(self, summary: EvaluationSummary, *, environment_steps: int) -> None:
        """记录固定种子确定性游玩的经营结果。"""

        self.writer.add_scalar(
            "evaluation/final_assets_mean",
            float(summary.mean_final_assets),
            environment_steps,
        )
        self.writer.add_scalar(
            "evaluation/final_assets_median",
            float(summary.median_final_assets),
            environment_steps,
        )
        self.writer.add_scalar(
            "evaluation/bankruptcy_rate",
            summary.bankruptcy_rate,
            environment_steps,
        )

    def close(self) -> None:
        """刷新并关闭事件文件。"""

        self.writer.close()

    def flush(self) -> None:
        """将当前更新的事件立即写入磁盘，便于训练中查看。"""

        self.writer.flush()


def _config_values(config: PPOTrainingConfig) -> Mapping[str, Any]:
    return {
        "updates": config.updates,
        "seed": config.seed,
        "device": config.device,
        "environment_count": config.environment_count,
        "evaluation_interval": config.evaluation_interval,
        "evaluation_seeds": config.evaluation_seeds,
        "tensorboard_flush_interval": config.tensorboard_flush_interval,
        "checkpoint_path": str(config.checkpoint_path) if config.checkpoint_path else None,
        "tensorboard_log_dir": str(config.tensorboard_log_dir)
        if config.tensorboard_log_dir
        else None,
        "ppo": asdict(config.ppo),
        "encoder": asdict(config.encoder),
    }


__all__ = ["TensorBoardLogger"]
