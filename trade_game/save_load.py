from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from dataclasses import fields
from typing import Any, Dict, List

from .loans import Loan
from .state import GameState, Player
from .inventory import CargoLot


def _base_dir() -> Path:
    """
    获取项目根目录。
    开发模式：trade_game/.. -> 项目根目录
    打包模式：可执行文件所在目录
    """
    if getattr(sys, 'frozen', False):
        # PyInstaller 打包后的情况：使用可执行文件所在目录
        return Path(sys.executable).parent
    else:
        # 开发模式：trade_game/.. -> 项目根目录
        return Path(__file__).resolve().parent.parent


SAVE_DIR = _base_dir() / "saves"


def _ensure_dir() -> None:
    SAVE_DIR.mkdir(parents=True, exist_ok=True)


def list_saves() -> List[str]:
    _ensure_dir()
    return sorted(p.stem for p in SAVE_DIR.glob("*.json"))


def save_path(name: str) -> Path:
    _ensure_dir()
    safe = name.strip()
    if not safe:
        safe = "save"
    return SAVE_DIR / f"{safe}.json"


def state_to_dict(state: GameState) -> Dict[str, Any]:
    return {
        "player": asdict(state.player),
        "game_mode": getattr(state, "game_mode", "free"),
        "daily_lambdas": dict(state.daily_lambdas),
        "previous_lambdas": dict(getattr(state, "previous_lambdas", {})),
        "price_history_buy_7d": {k: list(v) for k, v in getattr(state, "price_history_buy_7d", {}).items()},
        "price_history_sell_7d": {k: list(v) for k, v in getattr(state, "price_history_sell_7d", {}).items()},
        "loans": [asdict(l) for l in state.loans],
        "loss_by_product": dict(getattr(state, "loss_by_product", {})),
    }


def state_from_dict(d: Dict[str, Any]) -> GameState:
    p0 = d.get("player", {})
    # cargo_lots: list[dict] -> list[CargoLot]
    lots = [CargoLot(**x) for x in p0.get("cargo_lots", [])]
    p0["cargo_lots"] = lots
    # 兼容旧字段名：truck_capacity -> truck_total_capacity
    if "truck_capacity" in p0 and "truck_total_capacity" not in p0:
        p0["truck_total_capacity"] = p0.pop("truck_capacity")
    # 过滤掉旧存档/多余字段，避免 Player 结构演进导致无法读档
    allowed = {f.name for f in fields(Player)}
    p_filtered = {k: v for k, v in p0.items() if k in allowed}
    player = Player(**p_filtered)

    loans = [Loan(**x) for x in d.get("loans", [])]
    game_mode = d.get("game_mode", "free")
    state = GameState(
        player=player,
        game_mode=game_mode,
        daily_lambdas=dict(d.get("daily_lambdas", {})),
        previous_lambdas=dict(d.get("previous_lambdas", {})),
        price_history_buy_7d={k: list(v) for k, v in d.get("price_history_buy_7d", {}).items()},
        price_history_sell_7d={k: list(v) for k, v in d.get("price_history_sell_7d", {}).items()},
        loans=loans,
        loss_by_product=dict(d.get("loss_by_product", {})),
    )
    return state


def save_game(state: GameState, name: str) -> Path:
    path = save_path(name)
    payload = state_to_dict(state)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_game(name: str) -> GameState:
    path = save_path(name)
    if not path.exists():
        raise FileNotFoundError(str(path))
    payload = json.loads(path.read_text(encoding="utf-8"))
    return state_from_dict(payload)


def delete_game(name: str) -> None:
    """
    删除指定名称的存档文件（若不存在则静默忽略）。
    """
    path = save_path(name)
    try:
        if path.exists():
            path.unlink()
    except OSError:
        # 删除失败时保持静默，避免在 UI 中打断流程
        pass

