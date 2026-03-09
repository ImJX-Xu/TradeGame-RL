#!/usr/bin/env python
"""
检查录制 demo 是否可用于 PPO 训练。

用法：
  python check_demo.py [demo.npz]
  不传参数时扫描 runs/demos/*.npz 并检查最新一个。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# 确保能导入 trade_game
sys.path.insert(0, str(Path(__file__).resolve().parent))

from trade_game.sb3_env import EnvConfig, TradeGameMaskedEnv


def check_demo(path: str | Path) -> dict:
    """检查单个 demo 文件，返回诊断结果。"""
    path = Path(path)
    if not path.exists():
        return {"ok": False, "error": f"文件不存在: {path}"}

    try:
        data = np.load(path, allow_pickle=True)
    except Exception as e:
        return {"ok": False, "error": f"加载 npz 失败: {e}"}

    obs = data.get("obs")
    action = data.get("action")
    meta = data.get("meta", None)

    if obs is None or action is None:
        return {"ok": False, "error": "缺少 obs 或 action 字段"}

    obs = np.asarray(obs, dtype=np.float32)
    action = np.asarray(action, dtype=np.int64)

    n = len(action)
    if len(obs) != n:
        return {"ok": False, "error": f"obs 与 action 长度不一致: {len(obs)} vs {n}"}

    if n == 0:
        return {
            "ok": False,
            "error": "录制为空（0 条记录）",
            "n_steps": 0,
            "dropped": int(meta.item().get("dropped", 0)) if meta is not None else 0,
        }

    # 与训练环境对齐
    env_cfg = EnvConfig(game_mode="free", max_days=90, max_steps=90)
    env = TradeGameMaskedEnv(env_cfg)

    obs_shape = env.observation_space.shape
    n_actions = env.action_space.n

    issues: list[str] = []
    warnings: list[str] = []

    # 1. 观测维度
    if obs.ndim == 1:
        obs_flat = obs
    else:
        obs_flat = obs.reshape(obs.shape[0], -1)
    if obs_flat.shape[1] != obs_shape[0]:
        issues.append(f"观测维度不匹配: 录制 {obs_flat.shape[1]} vs 环境 {obs_shape[0]}")
    else:
        pass  # 维度匹配

    # 2. 动作范围
    act_min, act_max = int(action.min()), int(action.max())
    if act_min < 0 or act_max >= n_actions:
        issues.append(f"动作越界: 范围 [{act_min}, {act_max}]，环境动作空间 [0, {n_actions})")
    else:
        pass

    # 3. meta 信息
    dropped = 0
    dropped_details = []
    dropped_reason_counts = {}
    if meta is not None:
        m = meta.item() if hasattr(meta, "item") else meta
        if isinstance(m, dict):
            dropped = int(m.get("dropped", 0))
            dropped_details = list(m.get("dropped_details") or [])
            dropped_reason_counts = dict(m.get("dropped_reason_counts") or {})
            if dropped > 0:
                warnings.append(f"录制时丢弃了 {dropped} 个无法编码的动作")

    # 4. 训练可用性
    trainable = len(issues) == 0 and n > 0
    if trainable:
        msg = f"可用于训练：{n} 条 (obs, action)，观测/动作空间与 EnvConfig 一致"
    else:
        msg = "不可用于训练：" + "; ".join(issues) if issues else "未知"

    return {
        "ok": trainable,
        "path": str(path),
        "n_steps": n,
        "dropped": dropped,
        "dropped_details": dropped_details,
        "dropped_reason_counts": dropped_reason_counts,
        "obs_shape": obs_flat.shape,
        "action_range": (act_min, act_max),
        "n_actions_env": n_actions,
        "issues": issues,
        "warnings": warnings,
        "message": msg,
    }


def main() -> None:
    base = Path("runs") / "demos"
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
    elif base.exists():
        demos = sorted(base.glob("*.npz"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not demos:
            print(f"[check_demo] 未找到 demo 文件。请先在 GUI 中按 F8 录制一局，或选择「玩家演示模式」。")
            sys.exit(1)
        path = demos[0]
        print(f"[check_demo] 使用最新 demo: {path}")
    else:
        print("[check_demo] 用法: python check_demo.py [runs/demos/xxx.npz]")
        print("  或先运行 start_game.py 选择「玩家演示模式」录制一局。")
        sys.exit(1)

    r = check_demo(path)
    print()
    print("=== Demo 训练兼容性检查 ===")
    print(f"文件: {r.get('path', path)}")
    print(f"步数: {r.get('n_steps', 0)}")
    print(f"丢弃: {r.get('dropped', 0)}")
    print(f"观测形状: {r.get('obs_shape', '?')}")
    print(f"动作范围: [{r.get('action_range', (0,0))[0]}, {r.get('action_range', (0,0))[1]}] (环境动作数: {r.get('n_actions_env', '?')})")
    if r.get("dropped_reason_counts"):
        print("丢弃原因统计（Top）：")
        items = sorted(r["dropped_reason_counts"].items(), key=lambda kv: kv[1], reverse=True)[:8]
        for reason, cnt in items:
            print(f"  - {cnt}x {reason}")
    if r.get("dropped_details"):
        print("丢弃动作示例（最多 5 条）：")
        for d in list(r["dropped_details"])[:5]:
            day = d.get("day", "?")
            loc = d.get("location", "?")
            act = d.get("action", {})
            reason = d.get("reason", "")
            print(f"  - day={day} loc={loc} action={act} reason={reason}")
    if r.get("issues"):
        print("问题:")
        for i in r["issues"]:
            print(f"  - {i}")
    if r.get("warnings"):
        print("提示:")
        for w in r["warnings"]:
            print(f"  - {w}")
    print()
    print(r.get("message", ""))
    sys.exit(0 if r.get("ok") else 1)


if __name__ == "__main__":
    main()
