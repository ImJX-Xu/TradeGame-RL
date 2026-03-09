from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv


@dataclass(frozen=True)
class WarmstartConfig:
    demo_steps: int = 4000
    epochs: int = 3
    batch_size: int = 256
    learning_rate: float = 3e-4


def _collect_demonstrations(env: ActionMasker, demo_steps: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    用基线策略在单环境上采样 (obs, action)。
    env 必须是 ActionMasker(TradeGameMaskedEnv, mask_fn)。
    """
    from .baseline_policy import BaselinePolicyConfig, choose_action

    base_env = env.env  # underlying TradeGameMaskedEnv
    obs, _ = env.reset(seed=seed)

    obs_buf: List[np.ndarray] = []
    act_buf: List[int] = []

    pol_cfg = BaselinePolicyConfig()
    for _ in range(int(demo_steps)):
        a = int(choose_action(base_env, pol_cfg))
        obs_buf.append(np.asarray(obs, dtype=np.float32))
        act_buf.append(a)
        obs, _, terminated, truncated, _ = env.step(a)
        if terminated or truncated:
            obs, _ = env.reset()

    return np.stack(obs_buf, axis=0), np.asarray(act_buf, dtype=np.int64)

def load_demo_npz(path: str) -> Tuple[np.ndarray, np.ndarray]:
    import numpy as np

    data = np.load(path, allow_pickle=True)
    obs = np.asarray(data["obs"], dtype=np.float32)
    act = np.asarray(data["action"], dtype=np.int64)
    return obs, act


def pretrain_policy_from_baseline(
    model: MaskablePPO,
    single_env_fn,
    cfg: WarmstartConfig,
    seed: int = 123,
) -> None:
    """
    行为克隆预热：用基线策略生成演示数据，然后对 PPO policy 做监督预训练。

    - model: MaskablePPO（已创建、绑定 vec env）
    - single_env_fn: 返回单环境（ActionMasker 包装）用于采样演示
    """
    # 1) 采样 demo
    env = single_env_fn()
    try:
        obs_np, act_np = _collect_demonstrations(env, demo_steps=cfg.demo_steps, seed=seed)
    finally:
        try:
            env.close()
        except Exception:
            pass

    # 2) 监督训练 policy：最小化 -log π(a|s)
    import torch
    import torch.nn.functional as F

    device = model.device
    policy = model.policy
    policy.set_training_mode(True)

    opt = torch.optim.Adam(policy.parameters(), lr=float(cfg.learning_rate))

    n = int(obs_np.shape[0])
    bs = max(32, int(cfg.batch_size))
    for _epoch in range(int(cfg.epochs)):
        idx = np.random.permutation(n)
        for start in range(0, n, bs):
            j = idx[start : start + bs]
            obs_t = torch.as_tensor(obs_np[j], device=device)
            act_t = torch.as_tensor(act_np[j], device=device)

            dist = policy.get_distribution(obs_t)
            # Discrete 分布：用 log_prob 做交叉熵等价目标
            logp = dist.log_prob(act_t)
            loss = (-logp).mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    policy.set_training_mode(False)

def pretrain_policy_from_demo_file(model: MaskablePPO, demo_path: str, cfg: WarmstartConfig) -> None:
    """
    行为克隆预热：从人类录制（或其它来源）的 demo npz 文件中加载 (obs, action)。
    """
    obs_np, act_np = load_demo_npz(demo_path)
    if obs_np.size == 0 or act_np.size == 0:
        return

    import torch

    device = model.device
    policy = model.policy
    policy.set_training_mode(True)

    opt = torch.optim.Adam(policy.parameters(), lr=float(cfg.learning_rate))

    n = int(obs_np.shape[0])
    bs = max(32, int(cfg.batch_size))
    for _epoch in range(int(cfg.epochs)):
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

def make_single_masked_env(env_fn):
    """
    将一个返回 TradeGameMaskedEnv 的 env_fn 包装成 ActionMasker(单环境)，便于 demo 采样。
    """
    base_env = env_fn()
    return ActionMasker(base_env, lambda e: e.action_mask())

