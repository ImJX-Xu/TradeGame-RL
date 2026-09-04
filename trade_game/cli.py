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
    train.add_argument(
        "--rollout-steps",
        type=int,
        default=None,
        help="覆盖每个训练环境的单次采样决策数",
    )
    train.add_argument("--checkpoint", type=Path, default=None, help="训练完成后保存检查点")
    train.add_argument("--tensorboard-logdir", type=Path, default=None, help="TensorBoard 事件目录")
    dagger_ppo = subcommands.add_parser(
        "dagger-ppo",
        help="用贪心教师、DAgger 和行为克隆初始化 PPO",
    )
    dagger_ppo.add_argument("--config", type=Path, default=None, help="DAgger-PPO TOML 配置路径")
    dagger_ppo.add_argument("--updates", type=int, default=None, help="覆盖 PPO 更新次数")
    dagger_ppo.add_argument(
        "--rollout-steps",
        type=int,
        default=None,
        help="覆盖每个训练环境的单次采样决策数",
    )
    dagger_ppo.add_argument("--expert-episodes", type=int, default=None, help="覆盖初始贪心教师局数")
    dagger_ppo.add_argument("--dagger-rounds", type=int, default=None, help="覆盖 DAgger 轮数")
    dagger_ppo.add_argument("--checkpoint", type=Path, default=None, help="训练完成后保存检查点")
    dagger_ppo.add_argument(
        "--tensorboard-logdir",
        type=Path,
        default=None,
        help="覆盖 TensorBoard 事件目录",
    )
    dagger_bc = subcommands.add_parser(
        "dagger-bc",
        help="运行贪心教师、DAgger 和行为克隆，输出 BC 模型",
    )
    dagger_bc.add_argument("--config", type=Path, default=None, help="DAgger+BC TOML 配置路径")
    dagger_bc.add_argument("--expert-episodes", type=int, default=None)
    dagger_bc.add_argument("--dagger-rounds", type=int, default=None)
    dagger_bc.add_argument("--checkpoint", type=Path, default=None, help="BC 模型输出路径")
    dagger_bc.add_argument("--tensorboard-logdir", type=Path, default=None)
    finetune = subcommands.add_parser(
        "ppo-finetune",
        help="从已有 Actor-Critic 模型继续 PPO 训练",
    )
    finetune.add_argument("--model", type=Path, required=True, help="输入模型 checkpoint")
    finetune.add_argument("--config", type=Path, default=None, help="PPO TOML 配置路径")
    finetune.add_argument("--updates", type=int, default=None)
    finetune.add_argument("--rollout-steps", type=int, default=None)
    finetune.add_argument("--checkpoint", type=Path, default=None, help="finetune 后输出路径")
    finetune.add_argument("--tensorboard-logdir", type=Path, default=None)
    greedy = subcommands.add_parser("greedy", help="运行市场感知的贪心经营基准")
    greedy.add_argument("--seed", dest="seeds", type=int, action="append", default=None)
    greedy.add_argument("--episodes", type=int, default=16, help="未指定 --seed 时运行的局数")
    greedy.add_argument("--start-seed", type=int, default=101, help="未指定 --seed 时的起始种子")
    greedy.add_argument("--trace", action="store_true", help="输出每局完整命令轨迹")
    arguments = parser.parse_args(argv)
    if arguments.command == "play":
        _run_play(parser, arguments)
        return 0
    if arguments.command == "train":
        _run_train(parser, arguments)
        return 0
    if arguments.command == "dagger-ppo":
        _run_dagger_ppo(parser, arguments)
        return 0
    if arguments.command == "dagger-bc":
        _run_dagger_bc(parser, arguments)
        return 0
    if arguments.command == "ppo-finetune":
        _run_ppo_finetune(parser, arguments)
        return 0
    if arguments.command == "greedy":
        _run_greedy(arguments)
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


def _run_dagger_ppo(parser: argparse.ArgumentParser, arguments: argparse.Namespace) -> None:
    try:
        from trade_game.learning.cli import run_dagger_ppo
    except ModuleNotFoundError as error:
        if error.name in {"torch", "tensorboard"}:
            parser.error("DAgger-PPO 训练需要安装可选依赖：pip install -e \".[learning]\"")
        raise
    run_dagger_ppo(
        config_path=arguments.config,
        updates=arguments.updates,
        rollout_steps=arguments.rollout_steps,
        expert_episodes=arguments.expert_episodes,
        dagger_rounds=arguments.dagger_rounds,
        checkpoint_path=arguments.checkpoint,
        tensorboard_log_dir=arguments.tensorboard_logdir,
    )


def _run_dagger_bc(parser: argparse.ArgumentParser, arguments: argparse.Namespace) -> None:
    try:
        from trade_game.learning.cli import run_dagger_bc
    except ModuleNotFoundError as error:
        if error.name in {"torch", "tensorboard"}:
            parser.error("DAgger+BC 训练需要安装可选依赖：pip install -e \".[learning]\"")
        raise
    run_dagger_bc(
        config_path=arguments.config,
        expert_episodes=arguments.expert_episodes,
        dagger_rounds=arguments.dagger_rounds,
        checkpoint_path=arguments.checkpoint,
        tensorboard_log_dir=arguments.tensorboard_logdir,
    )


def _run_ppo_finetune(parser: argparse.ArgumentParser, arguments: argparse.Namespace) -> None:
    try:
        from trade_game.learning.cli import run_ppo_finetune
    except ModuleNotFoundError as error:
        if error.name in {"torch", "tensorboard"}:
            parser.error("PPO finetune 需要安装可选依赖：pip install -e \".[learning]\"")
        raise
    run_ppo_finetune(
        model_path=arguments.model,
        config_path=arguments.config,
        checkpoint_path=arguments.checkpoint,
        tensorboard_log_dir=arguments.tensorboard_logdir,
        updates=arguments.updates,
        rollout_steps=arguments.rollout_steps,
    )


def _run_greedy(arguments: argparse.Namespace) -> None:
    """执行不依赖学习框架的贪心经营基准，并以终端表格汇总结果。"""

    from collections import Counter

    from trade_game.analysis import evaluate_greedy

    seeds = (
        tuple(arguments.seeds)
        if arguments.seeds is not None
        else tuple(range(arguments.start_seed, arguments.start_seed + arguments.episodes))
    )
    evaluation = evaluate_greedy(seeds=seeds, capture_trace=arguments.trace)
    print(
        "贪心基准："
        f"平均资产={evaluation.mean_final_assets} "
        f"中位数={evaluation.median_final_assets} "
        f"最低={evaluation.minimum_final_assets} "
        f"最高={evaluation.maximum_final_assets} "
        f"破产率={evaluation.bankruptcy_rate:.2%}"
    )
    counts: Counter[str] = Counter()
    for episode in evaluation.episodes:
        counts.update(dict(episode.command_counts))
        print(
            f"seed={episode.seed} assets={episode.final_assets} days={episode.elapsed_days} "
            f"trucks={episode.truck_count} end={episode.end_reason.value}"
        )
        if arguments.trace:
            for step in episode.trace:
                print(
                    f"  day={step.day} city={step.location} "
                    f"command={step.command!r} assets={step.assets_after}"
                )
    print("操作汇总：" + " ".join(f"{name}={count}" for name, count in sorted(counts.items())))


__all__ = ["main"]
