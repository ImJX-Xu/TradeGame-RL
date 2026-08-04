"""TradeGame-RL 的命令行入口。"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from trade_game.core import GameMode

from .terminal import run_terminal_game


def main(argv: Sequence[str] | None = None) -> int:
    """解析启动选项并进入指定游玩模式。"""

    parser = argparse.ArgumentParser(prog="trade-game")
    subcommands = parser.add_subparsers(dest="command", required=True)
    play = subcommands.add_parser("play", help="开始一局人类游玩")
    play.add_argument("--seed", type=int, default=None)
    play.add_argument("--mode", type=GameMode, choices=tuple(GameMode), default=GameMode.FREE)
    play.add_argument("--terminal", action="store_true", help="使用终端界面")
    arguments = parser.parse_args(argv)
    if arguments.command == "play":
        if arguments.terminal:
            run_terminal_game(seed=arguments.seed, mode=arguments.mode)
        else:
            try:
                from .arcade import run_arcade_game
            except ModuleNotFoundError as error:
                if error.name == "arcade":
                    parser.error("图形界面需要安装可选依赖：pip install -e \".[ui]\"")
                raise
            run_arcade_game(seed=arguments.seed, mode=arguments.mode)
        return 0
    raise RuntimeError(f"未处理的子命令：{arguments.command}")
