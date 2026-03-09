from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.env_util import make_vec_env

from trade_game.sb3_env import EnvConfig, TradeGameMaskedEnv
from trade_game.ppo_warmstart import WarmstartConfig


def _ask_int(prompt: str, default: int) -> int:
    s = input(f"{prompt} [默认 {default}]: ").strip()
    if not s:
        return default
    try:
        return int(s)
    except ValueError:
        return default


def _load_all_demos() -> tuple[np.ndarray, np.ndarray]:
    base = Path("runs") / "demos"
    if not base.exists():
        print("未找到 runs/demos 目录。")
        return np.zeros((0, 1), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    demos = sorted(base.glob("*.npz"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not demos:
        print("runs/demos 目录为空。")
        return np.zeros((0, 1), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    print("\n将使用以下 demo 文件进行行为克隆（全部合并在一起）：")
    for p in demos:
        print(f"  - {p.name}")

    obs_list: list[np.ndarray] = []
    act_list: list[np.ndarray] = []
    for p in demos:
        data = np.load(p, allow_pickle=True)
        obs = np.asarray(data["obs"], dtype=np.float32)
        act = np.asarray(data["action"], dtype=np.int64)
        if obs.size == 0 or act.size == 0:
            continue
        if len(obs) != len(act):
            print(f"  * 跳过 {p.name}（obs/act 长度不一致: {len(obs)} vs {len(act)}）")
            continue
        obs_list.append(obs)
        act_list.append(act)

    if not obs_list:
        print("所有 demo 都为空或无效，无法进行 BC。")
        return np.zeros((0, 1), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    obs_np = np.concatenate(obs_list, axis=0)
    act_np = np.concatenate(act_list, axis=0)
    print(f"\n共加载 {len(demos)} 个 demo，总条数 {obs_np.shape[0]}。")
    return obs_np, act_np


def _play_one_episode(model: MaskablePPO, cfg: EnvConfig) -> None:
    print("\n=== 使用行为克隆后的策略自动游玩一局 ===")
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
    out_dir = Path("runs") / "bc_trade_game"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== 行为克隆（仅 BC）训练 ===")
    obs_np, act_np = _load_all_demos()
    if obs_np.size == 0 or act_np.size == 0:
        print("没有可用 demo，退出。")
        return

    cfg = EnvConfig(
        game_mode="free",
        max_days=90,
        max_steps=90,
    )

    def env_fn():
        base_env = TradeGameMaskedEnv(cfg)
        return ActionMasker(base_env, lambda env: env.action_mask())

    # 这里只是为了初始化 policy 结构；不会进行 PPO 训练
    vec_env = make_vec_env(env_fn, n_envs=1, seed=42)

    model = MaskablePPO(
        policy="MlpPolicy",
        env=vec_env,
        verbose=1,
        n_steps=128,
        batch_size=64,
        gamma=0.995,
        learning_rate=3e-4,
        ent_coef=0.01,
        tensorboard_log=None,
    )

    print("\n开始基于全部 demo 的 BC 预热。")
    ws_cfg = WarmstartConfig(
        demo_steps=0,
        epochs=_ask_int("BC 预热 epochs", 3),
        batch_size=_ask_int("BC batch_size", 256),
        learning_rate=3e-4,
    )

    device = model.device
    policy = model.policy
    policy.set_training_mode(True)

    opt = torch.optim.Adam(policy.parameters(), lr=float(ws_cfg.learning_rate))

    n = int(obs_np.shape[0])
    bs = max(32, int(ws_cfg.batch_size))
    for _epoch in range(int(ws_cfg.epochs)):
        idx = np.random.permutation(n)
        for start in range(0, n, bs):
            j = idx[start : start + bs]
            obs_t = torch.as_tensor(obs_np[j], device=device)
            act_t = torch.as_tensor(act_np[j], device=device)
            dist = policy.get_distribution(obs_t)
            loss = (-dist.log_prob(act_t)).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    policy.set_training_mode(False)

    model_path = out_dir / "bc_model.zip"
    model.save(str(model_path))
    print(f"\n已保存 BC 预训练模型到: {model_path}")

    play = input("\n是否使用该模型自动玩一局并打印进度？[y/N]: ").strip().lower()
    if play.startswith("y"):
        _play_one_episode(model, cfg)


if __name__ == "__main__":
    main()

