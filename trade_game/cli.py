"""项目级命令入口，分派人类游玩和智能体训练功能。"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from trade_game.core import GameMode


def main(argv: Sequence[str] | None = None) -> int:
    """解析顶层命令；具体界面和训练实现按需导入。"""

    parser = argparse.ArgumentParser(prog="trade-game")
    subcommands = parser.add_subparsers(dest="command", required=True)
    play = subcommands.add_parser("play", help="开始一局人类游玩")
    play.add_argument("--seed", type=int, default=None)
    play.add_argument("--mode", type=GameMode, choices=tuple(GameMode), default=GameMode.FREE)
    play.add_argument("--terminal", action="store_true", help="使用终端界面")
    train = subcommands.add_parser("train", help="使用原生 PyTorch PPO 训练智能体")
    train.add_argument("--config", type=Path, default=None, help="训练 TOML 配置路径")
    train.add_argument("--updates", type=int, default=None, help="覆盖配置中的 PPO 更新次数")
    train.add_argument("--rollout-steps", type=int, default=None, help="覆盖每次采样决策数")
    train.add_argument("--checkpoint", type=Path, default=None, help="训练完成后保存检查点")
    train.add_argument("--tensorboard-logdir", type=Path, default=None, help="TensorBoard 事件目录")
    arguments = parser.parse_args(argv)
    if arguments.command == "play":
        _run_play(parser, arguments)
        return 0
    if arguments.command == "train":
        _run_train(parser, arguments)
        return 0
    raise RuntimeError(f"未处理的子命令：{arguments.command}")


def _run_play(parser: argparse.ArgumentParser, arguments: argparse.Namespace) -> None:
    if arguments.terminal:
        from trade_game.ui.terminal import run_terminal_game

        run_terminal_game(seed=arguments.seed, mode=arguments.mode)
        return
    try:
        from trade_game.ui.arcade import run_arcade_game
    except ModuleNotFoundError as error:
        if error.name == "arcade":
            parser.error("图形界面需要安装可选依赖：pip install -e \".[ui]\"")
        raise
    run_arcade_game(seed=arguments.seed, mode=arguments.mode)


def _run_train(parser: argparse.ArgumentParser, arguments: argparse.Namespace) -> None:
    try:
        from trade_game.learning.cli import run_training
    except ModuleNotFoundError as error:
        if error.name == "torch":
            parser.error("PPO 训练需要安装可选依赖：pip install -e \".[learning]\"")
        raise
    run_training(
        config_path=arguments.config,
        updates=arguments.updates,
        rollout_steps=arguments.rollout_steps,
        checkpoint_path=arguments.checkpoint,
        tensorboard_log_dir=arguments.tensorboard_logdir,
    )


__all__ = ["main"]
