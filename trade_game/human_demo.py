from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from . import api
from .sb3_env import EnvConfig, TradeGameMaskedEnv


@dataclass
class DemoRecord:
    obs: np.ndarray
    action: int


class HumanDemoRecorder:
    """
    录制“人类玩家”产生的一条 trajectory，用于 PPO 预热（行为克隆）。

    存储内容：
    - obs: TradeGameMaskedEnv 的观测向量（float32）
    - action: TradeGameMaskedEnv 的离散动作 index（int）
    """

    def __init__(self, env_cfg: Optional[EnvConfig] = None):
        self.env_cfg = env_cfg or EnvConfig(game_mode="free", max_days=90, max_steps=90)
        self._encoder_env = TradeGameMaskedEnv(self.env_cfg)
        self._records: List[DemoRecord] = []
        self._dropped: int = 0
        self._dropped_details: List[Dict[str, Any]] = []

    @property
    def size(self) -> int:
        return len(self._records)

    @property
    def dropped(self) -> int:
        return self._dropped

    def clear(self) -> None:
        self._records.clear()
        self._dropped = 0

    def record(self, state: api.GameState, rng, action: api.Action) -> bool:
        """
        将“结构化 action”编码为离散 index，并保存 (obs, action_idx)。
        返回是否成功录制；若该动作无法编码则返回 False。
        """
        self._encoder_env.sync_state_for_encoding(state, rng=rng)
        idx, reason = self._encoder_env.encode_api_action_with_reason(action)
        if idx is None:
            self._dropped += 1
            if len(self._dropped_details) < 200:
                try:
                    payload = asdict(action) if is_dataclass(action) else {"repr": repr(action)}
                except Exception:
                    payload = {"repr": repr(action)}
                self._dropped_details.append(
                    {
                        "day": int(getattr(state.player, "day", 0)),
                        "location": str(getattr(state.player, "location", "")),
                        "action": payload,
                        "reason": str(reason or "无法编码"),
                    }
                )
            return False
        obs = self._encoder_env._obs()  # 使用同一套观测（与训练一致）
        self._records.append(DemoRecord(obs=np.asarray(obs, dtype=np.float32), action=int(idx)))
        return True

    def to_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        obs = np.stack([r.obs for r in self._records], axis=0) if self._records else np.zeros((0, 1), dtype=np.float32)
        act = np.asarray([r.action for r in self._records], dtype=np.int64) if self._records else np.zeros((0,), dtype=np.int64)
        return obs, act

    def save_npz(self, out_path: Path) -> Path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        obs, act = self.to_arrays()
        # 统计丢弃原因（便于快速定位）
        reason_counts: Dict[str, int] = {}
        for d in self._dropped_details:
            r = str(d.get("reason", ""))
            reason_counts[r] = int(reason_counts.get(r, 0) + 1)
        np.savez_compressed(
            out_path,
            obs=obs,
            action=act,
            meta=np.asarray(
                {
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                    "n_steps": int(len(act)),
                    "dropped": int(self._dropped),
                    "dropped_details": list(self._dropped_details),
                    "dropped_reason_counts": reason_counts,
                    "env_cfg": {
                        "game_mode": self.env_cfg.game_mode,
                        "max_days": int(self.env_cfg.max_days),
                        "max_steps": int(self.env_cfg.max_steps),
                    },
                },
                dtype=object,
            ),
        )
        return out_path


def default_demo_path(prefix: str = "human_demo") -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("runs") / "demos" / f"{prefix}_{ts}.npz"


if __name__ == "__main__":
    # 该模块设计为被 GUI 调用录制 demo；直接运行脚本时给出指引。
    raise SystemExit(
        "human_demo.py 不是独立可执行脚本。\n"
        "- 推荐：运行 `python start_game.py`，选择「玩家演示模式」自动录制并保存到 runs/demos/*.npz\n"
        "- 或者：GUI 内按 F8 手动开始/停止录制\n"
        "- 如果你确实要以模块方式运行：`python -m trade_game.human_demo`"
    )

