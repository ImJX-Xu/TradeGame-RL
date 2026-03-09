from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from . import api
from .sb3_env import TradeGameMaskedEnv


@dataclass(frozen=True)
class BaselinePolicyConfig:
    """
    基线玩家策略（脚本）配置。

    目标：提供一个稳定、可解释的“玩家玩法”基线，用于：
    - 作为 GUI/CLI 的自动游玩逻辑参考（同一套规则）
    - 作为 PPO 训练前的行为克隆数据来源（让探索从基线附近开始）
    """

    # 是否允许借贷（默认关掉：避免出现“借/还循环”占满训练）
    allow_loans: bool = False

    # 是否允许“同城卖出”（默认允许，但你已经在 reward 里对同城卖出做惩罚）
    allow_same_city_sell: bool = True


def _slice_any(mask: np.ndarray, start: int, end: int) -> np.ndarray:
    if start >= end:
        return np.zeros((0,), dtype=bool)
    return mask[start:end]


def choose_action(env: TradeGameMaskedEnv, cfg: Optional[BaselinePolicyConfig] = None) -> int:
    """
    给定 `TradeGameMaskedEnv`，按基线规则选择一个合法离散动作 index。

    说明：
    - 本函数会调用 `env.action_mask()`，并利用 env 内部缓存的 candidates（_last_*）来做决策。
    - 该策略刻意“保守且稳定”：优先跨城卖出，其次出行，其次买入，最后 next_day。
    """
    cfg = cfg or BaselinePolicyConfig()

    mask = env.action_mask()
    obs = api.get_observation(env._state)  # type: ignore[attr-defined]
    city = str(obs.get("location"))
    cargo: Dict[str, int] = dict(obs.get("cargo") or {})
    buy_prices: Dict[str, float] = dict(obs.get("buy_prices") or {})
    sell_prices: Dict[str, float] = dict(obs.get("sell_prices") or {})
    has_bank = bool(obs.get("has_bank"))

    # --- 1) 卖出：优先跨城卖出（origin != current city） ---
    # 卖出动作布局：slot -> base_idx (商品候选) + v_idx (数量档位)
    sell_mask = _slice_any(mask, env._idx_sell_start, env._idx_sell_end)  # type: ignore[attr-defined]
    if sell_mask.any():
        best_idx: Optional[int] = None
        best_score = -1e18
        for local_slot, ok in enumerate(sell_mask):
            if not ok:
                continue
            base_idx = local_slot // env._sell_qty_variants  # type: ignore[attr-defined]
            v_idx = local_slot % env._sell_qty_variants  # type: ignore[attr-defined]
            if base_idx >= len(env._last_sell_cands):  # type: ignore[attr-defined]
                continue
            pid = env._last_sell_cands[base_idx]  # type: ignore[attr-defined]
            have = int(cargo.get(pid, 0))
            if have <= 0:
                continue
            from .sb3_env import _compute_qty_variant
            qty = _compute_qty_variant(v_idx, have)
            if qty <= 0:
                continue

            # 跨城优先：若库存里该商品存在 origin_city != 当前城市的 lot，则加分
            cross_bonus = 0.0
            if env._state is not None:  # type: ignore[attr-defined]
                for lot in env._state.player.cargo_lots:  # type: ignore[attr-defined]
                    if lot.product_id == pid and lot.quantity > 0 and lot.origin_city != city:
                        cross_bonus = 1.0
                        break
            if (not cfg.allow_same_city_sell) and cross_bonus <= 0.0:
                continue

            price = float(sell_prices.get(pid, 0.0))
            score = price * qty + 1000.0 * cross_bonus  # 先保证跨城，其次按卖出总额
            if score > best_score:
                best_score = score
                best_idx = int(env._idx_sell_start + local_slot)  # type: ignore[attr-defined]
        if best_idx is not None:
            return best_idx

    # --- 2) 出行：有货时，去“该货物”卖价最高的城市（在可达候选里选） ---
    # 注意：基线不直接看运输成本/时间，先用“卖价差”驱动出行，足够当作基线。
    travel_land_mask = _slice_any(mask, env._idx_travel_land_start, env._idx_travel_land_end)  # type: ignore[attr-defined]
    if travel_land_mask.any():
        # 有货就尝试出行（跨城卖出更可能）
        if any(qty > 0 for qty in cargo.values()):
            best_t: Optional[int] = None
            best_t_score = -1e18
            # 选一个持仓最多的商品作为“主目标”
            main_pid = max(cargo.items(), key=lambda kv: kv[1])[0]
            for i, ok in enumerate(travel_land_mask):
                if not ok:
                    continue
                if i >= len(env._last_travel_land):  # type: ignore[attr-defined]
                    continue
                target = env._last_travel_land[i]  # type: ignore[attr-defined]
                # 用目的地的 sell_price 作为近似（直接从 api 取一次“假设到达城市的售价”需要完整状态，这里简化）
                # 简化策略：优先“离开产地/当前城市”，避免原地循环
                score = 1.0
                if target != city:
                    score += 1.0
                # 额外：若当前城市同城卖出被惩罚，出行有利，因此加权
                score += float(sell_prices.get(main_pid, 0.0)) * 0.0
                if score > best_t_score:
                    best_t_score = score
                    best_t = int(env._idx_travel_land_start + i)  # type: ignore[attr-defined]
            if best_t is not None:
                return best_t

    # --- 3) 买入：选择“单位价格最低”的可买动作（倾向买小量，减少同城卖出诱因） ---
    buy_mask = _slice_any(mask, env._idx_buy_start, env._idx_buy_end)  # type: ignore[attr-defined]
    if buy_mask.any():
        best_b: Optional[int] = None
        best_cost = 1e18
        for local_slot, ok in enumerate(buy_mask):
            if not ok:
                continue
            base_idx = local_slot // env._buy_qty_variants  # type: ignore[attr-defined]
            v_idx = local_slot % env._buy_qty_variants  # type: ignore[attr-defined]
            if base_idx >= len(env._last_buy_cands):  # type: ignore[attr-defined]
                continue
            pid = env._last_buy_cands[base_idx]  # type: ignore[attr-defined]
            price = buy_prices.get(pid)
            if price is None:
                continue
            rem_capacity = int(obs["capacity"] - obs["cargo_used"])
            from .sb3_env import _compute_qty_variant
            # 基线策略：买入变体上限取“容量上限”（现金不足会被 mask 过滤）
            qty = _compute_qty_variant(v_idx, rem_capacity)
            if qty <= 0:
                continue
            cost = float(price) * float(qty)
            # 偏好小成本买入，避免一次性全仓买导致策略陷入同城卖出循环
            if cost < best_cost:
                best_cost = cost
                best_b = int(env._idx_buy_start + local_slot)  # type: ignore[attr-defined]
        if best_b is not None:
            return best_b

    # --- 4) 贷款：默认禁用；若允许则选最小可借额 ---
    if cfg.allow_loans and has_bank:
        borrow_mask = _slice_any(mask, env._idx_borrow_start, env._idx_borrow_end)  # type: ignore[attr-defined]
        if borrow_mask.any():
            for i, ok in enumerate(borrow_mask):
                if ok:
                    return int(env._idx_borrow_start + i)  # type: ignore[attr-defined]

    # --- 5) fallback：next_day ---
    return int(env._idx_next)  # type: ignore[attr-defined]

