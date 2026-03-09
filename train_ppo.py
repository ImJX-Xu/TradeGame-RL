from __future__ import annotations

from pathlib import Path
from typing import Optional

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env

from trade_game.sb3_env import TradeGameMaskedEnv, EnvConfig
from trade_game.ppo_warmstart import WarmstartConfig, pretrain_policy_from_baseline


def _ask_int(prompt: str, default: int) -> int:
    s = input(f"{prompt} [默认 {default}]: ").strip()
    if not s:
        return default
    try:
        return int(s)
    except ValueError:
        return default


def _choose_demo_file() -> Optional[Path]:
    base = Path("runs") / "demos"
    if not base.exists():
        print("未找到 runs/demos 目录，跳过 demo 预热。")
        return None
    demos = sorted(base.glob("*.npz"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not demos:
        print("runs/demos 目录为空，跳过 demo 预热。")
        return None
    print("\n可用的人类 demo 文件（按时间倒序）:")
    for i, p in enumerate(demos):
        print(f"  [{i}] {p.name} (steps≈{p.stat().st_size} bytes)")
    s = input(f"请选择用于行为克隆预热的 demo 索引，或直接回车使用最新 [{demos[0].name}]: ").strip()
    if not s:
        return demos[0]
    try:
        idx = int(s)
    except ValueError:
        print("输入无效，使用最新 demo。")
        return demos[0]
    if 0 <= idx < len(demos):
        return demos[idx]
    print("索引越界，使用最新 demo。")
    return demos[0]


def _play_one_episode(model: MaskablePPO, cfg: EnvConfig) -> None:
    print("\n=== 使用训练好的策略自动游玩一局 ===")
    base_env = TradeGameMaskedEnv(cfg)
    env = ActionMasker(base_env, lambda e: e.action_mask())
    obs, info = env.reset()
    total_reward = 0.0
    for step in range(int(cfg.max_steps)):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        day = int(info.get("day", 0))
        cash = float(info.get("cash", 0.0))
        print(f"step={step+1:3d} day={day:3d} cash={cash:,.0f} reward={reward:8.2f}")
        if terminated or truncated:
            print("episode finished:", info.get("api", {}).get("done_reason", "terminated or truncated"))
            break
    print(f"总 reward: {total_reward:.2f}")
    env.close()


def main() -> None:
    out_dir = Path("runs") / "ppo_trade_game"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== PPO 训练配置 ===")
    total_timesteps = _ask_int("输入训练总步数 total_timesteps", 300_000)

    cfg = EnvConfig(
        game_mode="free",
        max_days=90,
        max_steps=90,  # 与 max_days 一致，一局最多 90 步/天
    )

    def env_fn():
        base_env = TradeGameMaskedEnv(cfg)
        return ActionMasker(base_env, lambda env: env.action_mask())

    vec_env = make_vec_env(env_fn, n_envs=4, seed=42)

    # tensorboard 可选：未安装时不启用
    tb_log = None
    try:
        import tensorboard  # noqa: F401

        tb_log = str(out_dir / "tb")
    except Exception:
        tb_log = None

    # 是否从已有模型启动（例如 train_bc_from_demo.py 产生的 BC 模型）
    load_pretrained = input("是否从已有模型（如 BC 预训练好的模型）继续 PPO 训练？[y/N]: ").strip().lower()
    if load_pretrained.startswith("y"):
        default_path = Path("runs") / "bc_trade_game" / "bc_model.zip"
        s = input(f"请输入模型路径，或直接回车使用默认 [{default_path}]: ").strip()
        model_path = Path(s) if s else default_path
        if not model_path.exists():
            print(f"找不到模型 {model_path}，改为从头初始化 PPO。")
            model = MaskablePPO(
                policy="MlpPolicy",
                env=vec_env,
                verbose=1,
                n_steps=1024,
                batch_size=256,
                gamma=0.995,
                learning_rate=3e-4,
                ent_coef=0.01,
                tensorboard_log=tb_log,
            )
        else:
            print(f"从 {model_path} 加载模型并继续 PPO 训练。")
            model = MaskablePPO.load(str(model_path), env=vec_env, tensorboard_log=tb_log)
    else:
        model = MaskablePPO(
            policy="MlpPolicy",
            env=vec_env,
            verbose=1,
            n_steps=1024,
            batch_size=256,
            gamma=0.995,
            learning_rate=3e-4,
            ent_coef=0.01,
            tensorboard_log=tb_log,
        )

        # 可选：从基线脚本小幅预热，再进入 PPO
        use_baseline = input("是否用基线脚本策略生成 demo 做少量 BC 预热？[y/N]: ").strip().lower()
        if use_baseline.startswith("y"):
            print("\n使用基线策略采样 demo 做行为克隆预热。")
            ws_cfg = WarmstartConfig(
                demo_steps=_ask_int("基线 demo 步数 demo_steps", 6000),
                epochs=_ask_int("BC 预热 epochs", 3),
                batch_size=_ask_int("BC batch_size", 256),
                learning_rate=3e-4,
            )
            from trade_game.ppo_warmstart import make_single_masked_env

            pretrain_policy_from_baseline(model, single_env_fn=lambda: make_single_masked_env(env_fn), cfg=ws_cfg, seed=42)

    ckpt = CheckpointCallback(
        save_freq=50_000,
        save_path=str(out_dir),
        name_prefix="ppo_ckpt",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    print(f"\n开始 PPO 训练，总步数: {total_timesteps}")
    model.learn(total_timesteps=total_timesteps, callback=ckpt, progress_bar=True)

    model_path = out_dir / "ppo_trade_game.zip"
    model.save(str(model_path))
    print(f"\nSaved model to: {model_path}")

    # 训练后可选：自动游玩一局并在终端打印进度
    play = input("\n是否使用训练成果自动玩一局并打印进度？[y/N]: ").strip().lower()
    if play.startswith("y"):
        _play_one_episode(model, cfg)


if __name__ == "__main__":
    main()

