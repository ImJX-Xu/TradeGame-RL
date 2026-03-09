from __future__ import annotations

from pathlib import Path

import numpy as np

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.env_util import make_vec_env

from trade_game.sb3_env import TradeGameMaskedEnv, EnvConfig
from trade_game.ppo_warmstart import (
    WarmstartConfig,
    pretrain_policy_from_baseline,
    pretrain_policy_from_demo_file,
)


def _decode_action_row(action: np.ndarray, info: dict) -> str:
    """将紧凑动作和 candidates 解码为可读的一行描述。"""
    a = np.asarray(action, dtype=int)
    a_type = int(a[0])
    buy_slot = int(a[1]) if len(a) > 1 else 0
    sell_slot = int(a[2]) if len(a) > 2 else 0
    travel_slot = int(a[3]) if len(a) > 3 else 0
    # mode_i = a[4]  # 当前未强制使用
    qty_idx = int(a[5]) if len(a) > 5 else 0
    amt_idx = int(a[6]) if len(a) > 6 else 0

    type_map = {
        0: "noop/next",
        1: "next",
        2: "buy",
        3: "sell",
        4: "travel",
        5: "repair",
        6: "borrow",
        7: "repay",
    }
    tname = type_map.get(a_type, f"unknown({a_type})")

    cands = info.get("candidates", {})
    buy_c = cands.get("buy") or []
    sell_c = cands.get("sell") or []
    trav_c = cands.get("travel") or []

    def pick(sl: int, arr: list[str]) -> str:
        if not arr:
            return "-"
        if sl < 0:
            sl = 0
        if sl >= len(arr):
            sl = len(arr) - 1
        return arr[sl]

    if tname == "buy":
        pid = pick(buy_slot, buy_c)
        return f"buy {pid} x slot{qty_idx}"
    if tname == "sell":
        pid = pick(sell_slot, sell_c)
        if pid == "-":
            return "idle (no cargo to sell)"
        return f"sell {pid} x slot{qty_idx}"
    if tname == "travel":
        city = pick(travel_slot, trav_c)
        mode = cands.get("mode", "-")
        return f"travel {city} via {mode}"
    if tname == "borrow":
        return f"borrow amt_slot{amt_idx}"
    if tname == "repay":
        return f"repay {'all' if amt_idx == 0 else f'slot{amt_idx}'}"
    if tname in ("noop/next", "next"):
        return tname
    if tname == "repair":
        return "repair truck"
    return tname


def _train_model(
    env_cfg: EnvConfig,
    save_to: Path,
    total_steps: int,
    seed: int = 123,
    warmstart: bool = True,
    warmstart_steps: int = 4000,
    warmstart_demo_path: str | None = None,
) -> MaskablePPO:
    """在紧凑环境上训练一个 PPO 模型并保存。"""
    save_to.parent.mkdir(parents=True, exist_ok=True)
    print(f"[menu] 训练 PPO 模型，步数={total_steps}，保存到 {save_to} ...")
    def env_fn():
        base_env = TradeGameMaskedEnv(env_cfg)
        return ActionMasker(base_env, lambda env: env.action_mask())

    vec_env = make_vec_env(env_fn, n_envs=2, seed=seed)
    model = MaskablePPO("MlpPolicy", vec_env, verbose=1, n_steps=256, batch_size=256)
    if warmstart and warmstart_steps > 0:
        print(f"[menu] 预热：demo_steps={warmstart_steps} ...")
        ws_cfg = WarmstartConfig(demo_steps=int(warmstart_steps), epochs=3, batch_size=256, learning_rate=3e-4)
        if warmstart_demo_path:
            print(f"[menu] 使用人类轨迹 demo 文件预热：{warmstart_demo_path}")
            pretrain_policy_from_demo_file(model, demo_path=warmstart_demo_path, cfg=ws_cfg)
        else:
            print("[menu] 使用脚本基线预热（无 demo 文件）。")
            pretrain_policy_from_baseline(model, single_env_fn=env_fn, cfg=ws_cfg, seed=seed)
    model.learn(total_timesteps=int(total_steps), progress_bar=True)
    model.save(str(save_to))
    print(f"[menu] 模型已保存：{save_to}")
    return model


def _ask_yes_no(prompt: str, default: bool = True) -> bool:
    d = "y" if default else "n"
    while True:
        s = input(f"{prompt} (y/n) [默认 {d}]: ").strip().lower()
        if s == "":
            return default
        if s in ("y", "yes"):
            return True
        if s in ("n", "no"):
            return False
        print("请输入 y 或 n。")


def _ask_int(prompt: str, default: int) -> int:
    while True:
        s = input(f"{prompt} [默认 {default}]: ").strip()
        if not s:
            return default
        try:
            v = int(s)
        except ValueError:
            print("请输入整数。")
            continue
        if v < 0:
            print("必须 >= 0。")
            continue
        return v


def _pick_demo_file() -> str | None:
    base = Path("runs") / "demos"
    if not base.exists():
        print("[menu] 未找到 runs/demos 目录（先在 GUI 按 F8 录制一局）。")
        return None
    demos = sorted(base.glob("*.npz"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not demos:
        print("[menu] runs/demos 下没有 .npz demo 文件（先在 GUI 按 F8 录制一局）。")
        return None
    print("\n可用人类轨迹 demo 列表：")
    for i, p in enumerate(demos[:20]):
        print(f"{i}) {p}")
    while True:
        s = input(f"选择 demo 编号 [0-{min(19, len(demos)-1)}]，或回车跳过: ").strip()
        if s == "":
            return None
        try:
            idx = int(s)
        except ValueError:
            print("请输入数字。")
            continue
        if 0 <= idx < min(20, len(demos)):
            return str(demos[idx])
        print("编号超出范围。")


def _ask_train_steps(default: int = 50_000) -> int:
    """交互式询问训练总步数上限。"""
    while True:
        s = input(f"训练总步数上限? [默认 {default}]: ").strip()
        if not s:
            return default
        try:
            v = int(s)
        except ValueError:
            print("请输入正整数。")
            continue
        if v <= 0:
            print("必须大于 0。")
            continue
        return v


def main() -> None:
    base_dir = Path("runs") / "ppo_trade_game"
    default_model = base_dir / "ppo_trade_game.zip"
    seed = 123
    env_cfg = EnvConfig(
        game_mode="free",
        max_days=90,
        max_steps=90,
    )

    # 扫描可用模型
    zips = []
    if base_dir.exists():
        zips = sorted(base_dir.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)

    print("\n=== 模型选择菜单 ===")
    print("1) 使用默认模型 (runs/ppo_trade_game/ppo_trade_game.zip)")
    print("2) 从列表中选择具体模型文件")
    print("3) 训练新模型后再游玩")
    print("4) 删除 runs/ppo_trade_game/ 下所有模型文件")
    print("5) 退出")

    choice = input("请选择 [1-5]，默认 1: ").strip()
    if choice == "":
        choice = "1"

    model_path: Path
    model: MaskablePPO | None = None

    if choice == "5":
        print("已退出。")
        return

    if choice == "4":
        if base_dir.exists():
            removed = 0
            for p in base_dir.glob("*.zip"):
                try:
                    p.unlink()
                    removed += 1
                except Exception as e:
                    print(f"[menu] 删除 {p} 失败：{e}")
            print(f"[menu] 已删除 {removed} 个模型文件。")
        else:
            print("[menu] 目录 runs/ppo_trade_game 不存在，无模型可删。")
        return

    if choice == "2":
        if not zips:
            print("[menu] 未找到任何模型，回退到选项 1。")
            model_path = default_model
        else:
            print("\n可用模型列表：")
            for idx, p in enumerate(zips):
                print(f"{idx}) {p}")
            while True:
                s = input(f"请输入要使用的模型编号 [0-{len(zips)-1}]，或回车取消: ").strip()
                if s == "":
                    print("[menu] 取消选择，回退到选项 1。")
                    model_path = default_model
                    break
                try:
                    i = int(s)
                except ValueError:
                    print("请输入有效数字。")
                    continue
                if 0 <= i < len(zips):
                    model_path = zips[i]
                    print(f"[menu] 已选择模型: {model_path}")
                    break
                print("编号超出范围。")
    elif choice == "3":
        model_path = default_model
        steps_to_train = _ask_train_steps()
        use_ws = _ask_yes_no("是否使用“玩家基线”预热 PPO（减少初期乱跑）?", default=True)
        ws_steps = _ask_int("基线预热步数（demo steps）?", default=4000) if use_ws else 0
        demo_path = _pick_demo_file() if use_ws else None
        model = _train_model(
            env_cfg,
            model_path,
            total_steps=steps_to_train,
            seed=seed,
            warmstart=use_ws,
            warmstart_steps=ws_steps,
            warmstart_demo_path=demo_path,
        )
    else:  # "1" 或其它
        model_path = default_model
        if not model_path.exists():
            print("[menu] 默认模型不存在，将先训练一个新模型。")
            steps_to_train = _ask_train_steps()
            use_ws = _ask_yes_no("是否使用“玩家基线”预热 PPO（减少初期乱跑）?", default=True)
            ws_steps = _ask_int("基线预热步数（demo steps）?", default=4000) if use_ws else 0
            demo_path = _pick_demo_file() if use_ws else None
            model = _train_model(
                env_cfg,
                model_path,
                total_steps=steps_to_train,
                seed=seed,
                warmstart=use_ws,
                warmstart_steps=ws_steps,
                warmstart_demo_path=demo_path,
            )

    base_env = TradeGameMaskedEnv(env_cfg)
    env = ActionMasker(base_env, lambda e: e.action_mask())

    # 若上面没有显式训练，则从文件加载
    if model is None:
        if not model_path.exists():
            steps_to_train = _ask_train_steps()
            model = _train_model(env_cfg, model_path, total_steps=steps_to_train, seed=seed)
        else:
            print(f"[menu] 从 {model_path} 加载模型...")
            model = MaskablePPO.load(str(model_path))

    obs, info = env.reset(seed=seed)
    terminated = truncated = False
    ep_reward = 0.0
    steps = 0

    # 询问是否打印进程表
    while True:
        ans = input("是否打印进程表? (y/n) [n]: ").strip().lower()
        if ans in ("", "n", "no"):
            trace = False
            break
        if ans in ("y", "yes"):
            trace = True
            break
        print("请输入 y 或 n。")

    # 初始状态行（step=0）+ 合法动作诊断
    if trace:
        init_obs = (info.get("api_info") or {}).get("observation", {})
        init_city = init_obs.get("location", "?")
        init_cash = init_obs.get("cash", 0.0)
        init_debt = init_obs.get("total_debt", 0.0)
        print(f"{'step':>5} {'day':>4} {'city':>4} {'cash':>10} {'debt':>10}  action")
        print(f"{0:5d} {init_obs.get('day', 1):4d} {init_city:>4} {init_cash:10.2f} {init_debt:10.2f}  -")
        summary = base_env.action_mask_summary()
        print(f"[合法动作] 共 {summary['n_valid']} 个: next_day={summary['next_day']}, buy={summary['buy']}, sell={summary['sell']}, travel_land={summary['travel_land']}, travel_sea={summary['travel_sea']}, borrow={summary['borrow']}, repay_all={summary['repay_all']}, repay_amt={summary['repay_amt']}")

    while not (terminated or truncated):
        # 在推理阶段同样使用 Action Mask，保证不可执行动作不会被策略选择
        current_mask = base_env.action_mask()
        action, _ = model.predict(obs, deterministic=True, action_masks=current_mask)
        obs, reward, terminated, truncated, info = env.step(action)
        ep_reward += float(reward)
        steps += 1
        if trace:
            obs_api = info.get("api", {}).get("observation", {})
            city = obs_api.get("location", "?")
            cash = obs_api.get("cash", info.get("cash", 0.0))
            debt = obs_api.get("total_debt", 0.0)
            day = obs_api.get("day", info.get("day", 0))
            act_desc = info.get("action_desc") or _decode_action_row(action, info)
            print(f"{steps:5d} {day:4d} {city:>4} {cash:10.2f} {debt:10.2f}  {act_desc}")
        elif steps % 50 == 0:
            print(f"step={steps} day={info.get('day')} cash={info.get('cash')}")

    print("done", {"terminated": terminated, "truncated": truncated, "steps": steps, "sum_reward": ep_reward})


if __name__ == "__main__":
    main()

