from __future__ import annotations

"""
游戏启动入口（统一前端）
--------------------------------
只保留一个真正的前端实现：`trade_game.arcade_app.TradeGameWindow`。
本文件只是一个轻量启动脚本，方便你在项目根目录直接运行：

    python start_game.py
"""

import arcade

from trade_game.arcade_app import TradeGameWindow


def run() -> None:
    """统一入口：启动 GUI。"""
    print("\n=== 风物千程 启动菜单 ===")
    print("1) 正常模式（自由/挑战/读档）")
    print("2) 玩家演示模式（自动录制 trajectory，用于 PPO warmstart）")
    s = input("请选择 [1-2]，默认 1: ").strip()
    if s == "":
        s = "1"
    demo_mode = s == "2"

    window = TradeGameWindow(demo_autorecord=demo_mode)
    if demo_mode:
        print("[demo] 已进入玩家演示模式：GUI 内会自动录制，退出窗口时自动保存到 runs/demos/*.npz")
    arcade.run()


if __name__ == "__main__":
    run()
