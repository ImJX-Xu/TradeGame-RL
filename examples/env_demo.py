"""
环境接口使用示例：不依赖 CLI/图形，用结构化动作驱动一局游戏。

运行：在项目根目录执行  python -m examples.env_demo
"""
from __future__ import annotations

import sys
from pathlib import Path

# 保证可导入 trade_game
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trade_game.api import (
    reset,
    step,
    get_observation,
    get_valid_cities,
    get_valid_product_ids,
    ActionNextDay,
    ActionBuy,
    ActionSell,
    ActionTravel,
    ActionRepairTruck,
    ActionBorrow,
    ActionRepay,
)


def main() -> None:
    state, rng, info = reset(seed=42, game_mode="free")
    print("--- reset ---")
    print("day:", info["day"], "wealth:", info["wealth"])
    obs = get_observation(state)
    print("location:", obs["location"], "cash:", obs["cash"], "capacity:", obs["capacity"])
    print("cities (sample):", get_valid_cities()[:5], "...")
    print("products (sample):", get_valid_product_ids()[:3], "...")

    # 买货
    state, reward, done, info = step(state, rng, ActionBuy(product_id="zz_flour", quantity=20))
    print("\n--- buy zz_flour 20 ---")
    print("reward:", reward, "done:", done, "messages:", info.get("messages"))
    print("wealth:", info["wealth"], "cargo:", get_observation(state)["cargo"])

    # 过一天
    state, reward, done, info = step(state, rng, ActionNextDay())
    print("\n--- next_day ---")
    print("reward:", reward, "day:", state.player.day)

    # 卖货
    state, reward, done, info = step(state, rng, ActionSell(product_id="zz_flour", quantity=10))
    print("\n--- sell zz_flour 10 ---")
    print("reward:", reward, "cash:", state.player.cash)

    # 无效动作（不在当前城市买不到的商品）
    state, reward, done, info = step(state, rng, ActionBuy(product_id="sjz_antibiotics", quantity=1))
    print("\n--- buy sjz_antibiotics (invalid at 郑州) ---")
    print("error:", info.get("error"), "reward:", reward)

    print("\n--- env interface demo done ---")


if __name__ == "__main__":
    main()
