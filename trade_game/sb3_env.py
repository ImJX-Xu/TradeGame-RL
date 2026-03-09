from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import gymnasium as gym
from gymnasium import spaces

from . import api
from .transport import RouteNotFound, TransportMode, route_km, validate_mode_allowed
from .game_config import TRUCK_MIN_DURABILITY_FOR_TRAVEL
from .train_config import AMOUNT_FRACTIONS


def _compute_qty_variant(variant_idx: int, max_qty: int) -> int:
    """
    根据“当前上限”和数量档位计算本次数量（供各环境复用）。

    0: 1/5 剩余容量
    1: 2/5 剩余容量
    2: 3/5 剩余容量
    3: 4/5 剩余容量
    4: 全部剩余容量
    5: 固定 1 单位
    6: 固定 2 单位
    7: 固定 3 单位
    """
    m = int(max_qty)
    if m <= 0:
        return 0
    if variant_idx == 0:
        qty = max(1, m // 5)
    elif variant_idx == 1:
        qty = max(1, (m * 2) // 5)
    elif variant_idx == 2:
        qty = max(1, (m * 3) // 5)
    elif variant_idx == 3:
        qty = max(1, (m * 4) // 5)
    elif variant_idx == 4:
        qty = m
    elif variant_idx == 5:
        qty = 1
    elif variant_idx == 6:
        qty = 2
    elif variant_idx == 7:
        qty = 3
    else:
        qty = 0
    return max(0, min(qty, m))


@dataclass(frozen=True)
class EnvConfig:
    """
    SB3/Gymnasium 环境配置。

    - max_days: free 模式下的截断上限（否则游戏理论上可无限进行）。
    - max_steps: 步数上限，默认与 max_days 一致，保证一局最多 max_days 步（一步可选 next_day 则对应一天）。
    """

    game_mode: str = "free"
    max_days: int = 90
    max_steps: int = 90  # 与 max_days 一致，避免“不翻日”时跑满 2000 步

    # 紧凑动作空间：每一步仅在“有效候选集合”里选 Top-K，减少无效动作
    max_travel_choices: int = 6
    max_buy_choices: int = 6
    max_sell_choices: int = 6

    # 训练友好：把每个 env.step 视为“一天一次决策”
    # 若所选动作未推进时间（buy/sell/borrow/repay 等），则自动追加一次 next_day。
    auto_advance_day: bool = True

    # 数量变体固定 8 档（上限的 1/5~5/5 + 固定 1/2/3）；金额比例 1/3~3/3
    amount_fractions: Tuple[float, ...] = AMOUNT_FRACTIONS

    @property
    def amount_options(self) -> Tuple[float, ...]:
        """兼容：槽位数与 amount_fractions 一致。"""
        return self.amount_fractions


class TradeGameSB3Env(gym.Env[np.ndarray, np.ndarray]):
    """
    将 `trade_game.api` 包装为 Gymnasium 环境，供 stable-baselines3 的 PPO 直接训练。

    - 动作空间：MultiDiscrete（参数化离散动作）
    - 观测空间：Box(float32)（扁平向量）
    - reward：现金增量（cash_after - cash_before）
    - terminated：破产
    - truncated：达到 max_days 或 max_steps
    """

    metadata = {"render_modes": []}

    def __init__(self, config: EnvConfig | None = None):
        super().__init__()
        self.config = config or EnvConfig()

        self._cities = api.get_valid_cities()
        self._products = api.get_valid_product_ids()
        self._city_index = {c: i for i, c in enumerate(self._cities)}
        self._product_index = {pid: i for i, pid in enumerate(self._products)}

        n_types = 8  # noop, next_day, buy, sell, travel, repair, borrow, repay
        self.action_space = spaces.MultiDiscrete(
            np.array(
                [
                    n_types,
                    len(self._products),
                    8,  # qty_variant: 0..7（上限的 1/5~5/5 + 固定 1/2/3）
                    len(self._cities),
                    2,  # mode: 0 land, 1 sea
                    1 + len(self.config.amount_options),  # amount_idx: 0 means repay_all; 1..N amounts
                ],
                dtype=np.int64,
            )
        )

        # 观测向量：
        # [ day_norm, cash, durability_norm, capacity, debt,
        #   location_onehot(14),
        #   cargo_qty(18),
        #   buy_prices(18) (不可买=0),
        #   sell_prices(18)
        # ]
        self._obs_dim = (
            5
            + len(self._cities)
            + len(self._products)
            + len(self._products)
            + len(self._products)
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._obs_dim,),
            dtype=np.float32,
        )

        self._state = None
        self._rng = None
        self._steps = 0

    def _obs(self) -> np.ndarray:
        assert self._state is not None
        obs_d = api.get_observation(self._state)

        day = float(obs_d["day"])
        day_norm = day / max(1.0, float(self.config.max_days))
        cash = float(obs_d["cash"])
        durability_norm = float(obs_d["truck_durability"]) / 100.0
        capacity = float(obs_d["capacity"])
        debt = float(obs_d["total_debt"])

        loc_oh = np.zeros((len(self._cities),), dtype=np.float32)
        li = self._city_index.get(str(obs_d["location"]), 0)
        loc_oh[li] = 1.0

        cargo = np.zeros((len(self._products),), dtype=np.float32)
        for pid, qty in obs_d["cargo"].items():
            idx = self._product_index.get(pid)
            if idx is not None:
                cargo[idx] = float(qty)

        buy_prices = np.zeros((len(self._products),), dtype=np.float32)
        for pid, price in obs_d["buy_prices"].items():
            idx = self._product_index.get(pid)
            if idx is not None:
                buy_prices[idx] = float(price)

        sell_prices = np.zeros((len(self._products),), dtype=np.float32)
        for pid, price in obs_d["sell_prices"].items():
            idx = self._product_index.get(pid)
            if idx is not None:
                sell_prices[idx] = float(price)

        v = np.concatenate(
            [
                np.array([day_norm, cash, durability_norm, capacity, debt], dtype=np.float32),
                loc_oh,
                cargo,
                buy_prices,
                sell_prices,
            ],
            dtype=np.float32,
        )
        return v

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._steps = 0
        self._state, self._rng, info0 = api.reset(seed=seed, game_mode=self.config.game_mode)
        obs = self._obs()
        info = {"api_info": info0}
        return obs, info

    def step(self, action: np.ndarray):
        assert self._state is not None and self._rng is not None
        self._steps += 1

        cash_before = float(self._state.player.cash)
        debt_before = float(sum(l.debt_total() for l in self._state.loans))
        day_before = int(self._state.player.day)

        a = np.asarray(action, dtype=np.int64).tolist()
        a_type, prod_i, qty_i, city_i, mode_i, amt_i = a

        # 解码参数
        pid = self._products[int(prod_i)]
        qty_variant = int(qty_i)
        city = self._cities[int(city_i)]
        mode = "sea" if int(mode_i) == 1 else "land"

        # amount_idx: 0 means "all" for repay; 1..N for fraction slots (1/3, 2/3, 3/3)
        if int(amt_i) == 0:
            amount = None
        else:
            frac = float(self.config.amount_fractions[int(amt_i) - 1])
            amount = frac  # 仅表示比例；实际金额在 borrow/repay 分支里按状态计算

        # 构造 Action
        idle_step = False  # “啥也没干”的一天，用于惩罚
        # 约定：noop 也会消耗一天（等价 next_day），避免策略卡在 day=1
        if a_type == 0:
            act = api.ActionNextDay()
        elif a_type == 1:
            act = api.ActionNextDay()
        elif a_type == 2:
            obs = api.get_observation(self._state)
            rem_capacity = int(obs["capacity"] - obs["cargo_used"])
            price = float((obs.get("buy_prices") or {}).get(pid) or 0.0)
            cash = float(obs.get("cash") or 0.0)
            max_by_cash = int(cash // price) if price > 0 else rem_capacity
            max_qty = max(0, min(rem_capacity, max_by_cash))
            qty = _compute_qty_variant(qty_variant, max_qty)
            act = api.ActionBuy(product_id=pid, quantity=max(1, int(qty)))
        elif a_type == 3:
            obs = api.get_observation(self._state)
            have = int((obs.get("cargo") or {}).get(pid, 0))
            qty = _compute_qty_variant(qty_variant, have)
            act = api.ActionSell(product_id=pid, quantity=max(1, int(qty)))
        elif a_type == 4:
            act = api.ActionTravel(city=city, mode=mode)
        elif a_type == 5:
            act = api.ActionRepairTruck()
        elif a_type == 6:
            from .loans import total_outstanding_principal
            total_debt_amount = sum(l.debt_total() for l in self._state.loans)
            net_assets = self._state.player.cash - total_debt_amount
            principal_total = total_outstanding_principal(self._state.loans)
            max_loan = max(0.0, net_assets - principal_total)
            frac = float(self.config.amount_fractions[int(amt_i) - 1]) if int(amt_i) > 0 else self.config.amount_fractions[0]
            borrow_amt = max(100.0, round(frac * max_loan, 2))
            act = api.ActionBorrow(amount=borrow_amt)
        elif a_type == 7:
            if amount is None or int(amt_i) == 0:
                act = api.ActionRepay(amount="all")
            else:
                total_debt = sum(l.debt_total() for l in self._state.loans)
                max_repay = min(self._state.player.cash, total_debt)
                frac = float(self.config.amount_fractions[int(amt_i) - 1])
                if frac >= 0.99:
                    act = api.ActionRepay(amount="all")
                else:
                    act = api.ActionRepay(amount=max(100.0, round(frac * max_repay, 2)))
        else:
            act = api.ActionNoop()

        # 执行动作（若动作无效则 fallback 到 next_day，避免训练/推理卡死）
        _, _, done_api, info_api = api.step(self._state, self._rng, act)
        if info_api.get("error"):
            idle_step = True
            info_api["fallback"] = "next_day"
            _, _, done_api, info_api2 = api.step(self._state, self._rng, api.ActionNextDay())
            # 合并信息（保留原错误原因）
            info_api2["error"] = info_api.get("error")
            info_api2["fallback"] = "next_day"
            info_api = info_api2

        # 若本动作未推进时间，则自动推进 1 天（把每步视作“每日决策”）
        if self.config.auto_advance_day and int(self._state.player.day) == day_before:
            _, _, done_api2, info_api2 = api.step(self._state, self._rng, api.ActionNextDay())
            info_api2["auto_advance_day"] = True
            # 保留原动作的错误信息（若有）
            if info_api.get("error"):
                info_api2["error"] = info_api.get("error")
                info_api2["fallback"] = info_api.get("fallback")
            done_api = done_api2
            info_api = info_api2

        # reward = 净资产增量：现金 - 债务（防止策略刷“借贷套利”）
        cash_after = float(self._state.player.cash)
        debt_after = float(sum(l.debt_total() for l in self._state.loans))
        net_before = cash_before - debt_before
        net_after = cash_after - debt_after
        reward = net_after - net_before

        # terminated: 破产
        terminated = bool(info_api.get("done_reason") == "bankruptcy") or bool(info_api.get("bankruptcy"))

        # truncated: 时间或步数上限
        truncated = False
        if self._state.player.day >= self.config.max_days:
            truncated = True
        if self._steps >= self.config.max_steps:
            truncated = True
        if idle_step and not terminated:
            # 一天什么都没做：额外惩罚
            reward -= 10.0

        obs = self._obs()
        info: Dict[str, Any] = {
            "api": info_api,
            "cash": float(self._state.player.cash),
            "day": int(self._state.player.day),
        }

        # 如果 api 给了 done，但不是破产（例如挑战模式到期），这里也结束
        if done_api and not terminated:
            truncated = True

        return obs, reward, terminated, truncated, info


class TradeGameSB3EnvCompact(gym.Env[np.ndarray, np.ndarray]):
    """
    紧凑动作空间版本（推荐用于训练）：
    - travel 只从“当前城市、当前运输方式”可达目的地 Top-K 中选
    - buy 只从“当前城市可买商品”Top-K 中选
    - sell 只从“当前持仓商品”Top-K 中选

    这样可以显著减少无效动作比例，让 PPO 更容易学到买→运→卖链条。
    """

    metadata = {"render_modes": []}

    def __init__(self, config: EnvConfig | None = None):
        super().__init__()
        self.config = config or EnvConfig()

        self._cities = api.get_valid_cities()
        self._products = api.get_valid_product_ids()
        self._city_index = {c: i for i, c in enumerate(self._cities)}
        self._product_index = {pid: i for i, pid in enumerate(self._products)}

        n_types = 8  # noop, next_day, buy, sell, travel, repair, borrow, repay
        self.action_space = spaces.MultiDiscrete(
            np.array(
                [
                    n_types,
                    max(1, int(self.config.max_buy_choices)),
                    max(1, int(self.config.max_sell_choices)),
                    max(1, int(self.config.max_travel_choices)),
                    2,  # mode: 0 land, 1 sea
                    8,  # qty_variant
                    1 + len(self.config.amount_options),  # amount_idx: 0 repay_all; 1..N amounts
                ],
                dtype=np.int64,
            )
        )

        # 观测空间与 TradeGameSB3Env 一致
        self._obs_dim = (
            5
            + len(self._cities)
            + len(self._products)
            + len(self._products)
            + len(self._products)
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._obs_dim,),
            dtype=np.float32,
        )

        self._state = None
        self._rng = None
        self._steps = 0

    def _obs(self) -> np.ndarray:
        assert self._state is not None
        obs_d = api.get_observation(self._state)

        day = float(obs_d["day"])
        day_norm = day / max(1.0, float(self.config.max_days))
        cash = float(obs_d["cash"])
        durability_norm = float(obs_d["truck_durability"]) / 100.0
        capacity = float(obs_d["capacity"])
        debt = float(obs_d["total_debt"])

        loc_oh = np.zeros((len(self._cities),), dtype=np.float32)
        li = self._city_index.get(str(obs_d["location"]), 0)
        loc_oh[li] = 1.0

        cargo = np.zeros((len(self._products),), dtype=np.float32)
        for pid, qty in obs_d["cargo"].items():
            idx = self._product_index.get(pid)
            if idx is not None:
                cargo[idx] = float(qty)

        buy_prices = np.zeros((len(self._products),), dtype=np.float32)
        for pid, price in obs_d["buy_prices"].items():
            idx = self._product_index.get(pid)
            if idx is not None:
                buy_prices[idx] = float(price)

        sell_prices = np.zeros((len(self._products),), dtype=np.float32)
        for pid, price in obs_d["sell_prices"].items():
            idx = self._product_index.get(pid)
            if idx is not None:
                sell_prices[idx] = float(price)

        v = np.concatenate(
            [
                np.array([day_norm, cash, durability_norm, capacity, debt], dtype=np.float32),
                loc_oh,
                cargo,
                buy_prices,
                sell_prices,
            ],
            dtype=np.float32,
        )
        return v

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._steps = 0
        self._state, self._rng, info0 = api.reset(seed=seed, game_mode=self.config.game_mode)
        obs = self._obs()
        info = {"api_info": info0}
        return obs, info

    def _buy_candidates(self) -> List[str]:
        assert self._state is not None
        obs = api.get_observation(self._state)
        pids = sorted(obs["buy_prices"].keys())
        return pids[: max(1, int(self.config.max_buy_choices))]

    def _sell_candidates(self) -> List[str]:
        assert self._state is not None
        obs = api.get_observation(self._state)
        cargo: Dict[str, int] = obs["cargo"]
        # 持仓优先：按数量降序
        pids = sorted([pid for pid, qty in cargo.items() if qty > 0], key=lambda x: cargo[x], reverse=True)
        return pids[: max(1, int(self.config.max_sell_choices))]

    @staticmethod
    def _sea_rule_allowed(start: str, target: str, sea_departure_port: str) -> bool:
        port_cities = ("上海", "福州", "广州", "深圳", "海南", "台北", "高雄")
        if start not in port_cities:
            return False
        # 出海规则：大陆海港仅能去海岛；海岛仅能回 sea_departure_port 或去其他海岛
        if start in ("上海", "福州", "广州", "深圳") and target not in ("海南", "台北", "高雄"):
            return False
        if start in ("海南", "台北", "高雄"):
            if target in ("海南", "台北", "高雄"):
                return True
            return bool(sea_departure_port) and target == sea_departure_port
        return True

    def _travel_candidates(self, mode: TransportMode) -> List[str]:
        assert self._state is not None
        p = self._state.player
        start = p.location
        if mode == TransportMode.LAND and p.truck_durability <= TRUCK_MIN_DURABILITY_FOR_TRAVEL:
            return []

        pairs: List[tuple[int, str]] = []
        for city in self._cities:
            if city == start:
                continue
            if mode == TransportMode.SEA:
                if not self._sea_rule_allowed(start, city, p.sea_departure_port):
                    continue
            try:
                validate_mode_allowed(mode, start, city)
                km = route_km(mode, start, city)
            except RouteNotFound:
                continue
            pairs.append((int(km), city))

        pairs.sort(key=lambda t: t[0])
        k = max(1, int(self.config.max_travel_choices))
        return [c for _, c in pairs[:k]]

    def step(self, action: np.ndarray):
        assert self._state is not None and self._rng is not None
        self._steps += 1

        cash_before = float(self._state.player.cash)
        debt_before = float(sum(l.debt_total() for l in self._state.loans))
        day_before = int(self._state.player.day)

        a = np.asarray(action, dtype=np.int64).tolist()
        a_type, buy_slot, sell_slot, travel_slot, mode_i, qty_i, amt_i = a

        qty_variant = int(qty_i)

        # amount_idx: 0 means repay_all; 1..N amounts
        if int(amt_i) == 0:
            amount = None
        else:
            amount = float(self.config.amount_options[int(amt_i) - 1])

        mode = TransportMode.SEA if int(mode_i) == 1 else TransportMode.LAND

        # 候选集合（用于映射紧凑动作）
        buy_cands = self._buy_candidates()
        sell_cands = self._sell_candidates()
        travel_cands = self._travel_candidates(mode)

        def pick(cands: List[str], slot: int) -> Optional[str]:
            if not cands:
                return None
            i = int(slot)
            if i < 0:
                i = 0
            if i >= len(cands):
                i = len(cands) - 1
            return cands[i]

        # 构造 Action：无效/空候选会 fallback 到 next_day
        idle_step = False
        if a_type in (0, 1):  # noop / next_day
            act = api.ActionNextDay()
            idle_step = True
        elif a_type == 2:  # buy
            pid = pick(buy_cands, buy_slot)
            if pid:
                obs = api.get_observation(self._state)
                rem_capacity = int(obs["capacity"] - obs["cargo_used"])
                price = float((obs.get("buy_prices") or {}).get(pid) or 0.0)
                cash = float(obs.get("cash") or 0.0)
                max_by_cash = int(cash // price) if price > 0 else rem_capacity
                max_qty = max(0, min(rem_capacity, max_by_cash))
                qty = _compute_qty_variant(qty_variant, max_qty)
                act = api.ActionBuy(product_id=pid, quantity=max(1, int(qty)))
            else:
                act = api.ActionNextDay()
                idle_step = True
        elif a_type == 3:  # sell
            pid = pick(sell_cands, sell_slot)
            if pid:
                obs = api.get_observation(self._state)
                have = int((obs.get("cargo") or {}).get(pid, 0))
                qty = _compute_qty_variant(qty_variant, have)
                act = api.ActionSell(product_id=pid, quantity=max(1, int(qty)))
            else:
                act = api.ActionNextDay()
                idle_step = True
        elif a_type == 4:  # travel
            city = pick(travel_cands, travel_slot)
            if city:
                act = api.ActionTravel(city=city, mode=("sea" if mode == TransportMode.SEA else "land"))
            else:
                act = api.ActionNextDay()
                idle_step = True
        elif a_type == 5:  # repair
            act = api.ActionRepairTruck()
        elif a_type == 6:  # borrow
            borrow_amt = float(self.config.amount_options[0]) if amount is None else amount
            act = api.ActionBorrow(amount=borrow_amt)
        elif a_type == 7:  # repay
            act = api.ActionRepay(amount="all") if amount is None else api.ActionRepay(amount=amount)
        else:
            act = api.ActionNextDay()

        _, _, done_api, info_api = api.step(self._state, self._rng, act)
        if info_api.get("error"):
            idle_step = True
            info_api["fallback"] = "next_day"
            _, _, done_api, info_api2 = api.step(self._state, self._rng, api.ActionNextDay())
            info_api2["error"] = info_api.get("error")
            info_api2["fallback"] = "next_day"
            info_api = info_api2

        if self.config.auto_advance_day and int(self._state.player.day) == day_before:
            _, _, done_api2, info_api2 = api.step(self._state, self._rng, api.ActionNextDay())
            info_api2["auto_advance_day"] = True
            if info_api.get("error"):
                info_api2["error"] = info_api.get("error")
                info_api2["fallback"] = info_api.get("fallback")
            done_api = done_api2
            info_api = info_api2

        cash_after = float(self._state.player.cash)
        debt_after = float(sum(l.debt_total() for l in self._state.loans))
        net_before = cash_before - debt_before
        net_after = cash_after - debt_after
        reward = net_after - net_before

        terminated = bool(info_api.get("done_reason") == "bankruptcy") or bool(info_api.get("bankruptcy"))

        truncated = False
        if self._state.player.day >= self.config.max_days:
            truncated = True
        if self._steps >= self.config.max_steps:
            truncated = True
        if done_api and not terminated:
            truncated = True
        if idle_step and not terminated:
            reward -= 10.0

        obs = self._obs()
        info: Dict[str, Any] = {
            "api": info_api,
            "cash": float(self._state.player.cash),
            "day": int(self._state.player.day),
            "candidates": {
                "buy": buy_cands,
                "sell": sell_cands,
                "travel": travel_cands,
                "mode": mode.value,
            },
        }

        return obs, reward, terminated, truncated, info


class TradeGameMaskedEnv(gym.Env[np.ndarray, int]):
    """
    使用离散动作 + Action Mask 的环境版本，配合 sb3-contrib.MaskablePPO。

    动作空间：Discrete(N)，索引含义固定，由 action_mask() 告知当前哪些 index 有效。
    """

    metadata = {"render_modes": []}

    def __init__(self, config: EnvConfig | None = None):
        super().__init__()
        self.config = config or EnvConfig()

        self._cities = api.get_valid_cities()
        self._products = api.get_valid_product_ids()

        # --- BUY 数量变体：每个 buy 槽位拆成 8 档数量 ---
        # 0: 1/5 剩余容量
        # 1: 2/5 剩余容量
        # 2: 3/5 剩余容量
        # 3: 4/5 剩余容量
        # 4: 5/5 剩余容量（全部）
        # 5: 固定 1 单位
        # 6: 固定 2 单位
        # 7: 固定 3 单位
        self._buy_qty_variants = 8

        # 动作布局：
        # 0                   : next_day
        # [1, 1+K_buy)        : buy 槽位（包含商品×数量变体）
        # [B0, B0+K_sell)     : sell 槽位（商品×数量变体，仅当持仓>=数量时合法）
        # [S0, S0+K_travel)   : travel_land 槽位
        # [TL0, TL0+K_travel) : travel_sea 槽位
        # [TS0, TS0+M)        : borrow amount 槽位
        # R_ALL               : repay_all
        # [R0, R0+M)          : repay amount 槽位

        # 每个 buy 槽位 = (商品槽位 × 数量变体)
        base_buy_slots = max(1, int(self.config.max_buy_choices))
        Kb = base_buy_slots * self._buy_qty_variants
        Ks = max(1, int(self.config.max_sell_choices))
        self._sell_qty_variants = 8  # 与买入一致：上限的 1/5~5/5 + 固定 1/2/3
        Ks_slots = Ks * self._sell_qty_variants
        Kt = max(1, int(self.config.max_travel_choices))
        M = len(self.config.amount_fractions)

        self._idx_next = 0
        self._idx_buy_start = 1
        self._idx_buy_end = self._idx_buy_start + Kb
        self._idx_sell_start = self._idx_buy_end
        self._idx_sell_end = self._idx_sell_start + Ks_slots
        self._idx_travel_land_start = self._idx_sell_end
        self._idx_travel_land_end = self._idx_travel_land_start + Kt
        self._idx_travel_sea_start = self._idx_travel_land_end
        self._idx_travel_sea_end = self._idx_travel_sea_start + Kt
        self._idx_borrow_start = self._idx_travel_sea_end
        self._idx_borrow_end = self._idx_borrow_start + M
        self._idx_repay_all = self._idx_borrow_end
        self._idx_repay_start = self._idx_repay_all + 1
        self._idx_repay_end = self._idx_repay_start + M

        self._n_actions = self._idx_repay_end
        self.action_space = spaces.Discrete(self._n_actions)

        # 观测与上面环境一致
        self._obs_dim = (
            5
            + len(self._cities)
            + len(self._products)
            + len(self._products)
            + len(self._products)
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._obs_dim,),
            dtype=np.float32,
        )

        self._state = None
        self._rng = None
        self._steps = 0

        # 最近一步用于 debug
        self._last_buy_cands: List[str] = []
        self._last_sell_cands: List[str] = []
        self._last_travel_land: List[str] = []
        self._last_travel_sea: List[str] = []

    def _obs(self) -> np.ndarray:
        assert self._state is not None
        obs_d = api.get_observation(self._state)

        day = float(obs_d["day"])
        day_norm = day / max(1.0, float(self.config.max_days))
        cash = float(obs_d["cash"])
        durability_norm = float(obs_d["truck_durability"]) / 100.0
        capacity = float(obs_d["capacity"])
        debt = float(obs_d["total_debt"])

        loc_oh = np.zeros((len(self._cities),), dtype=np.float32)
        city_idx = {c: i for i, c in enumerate(self._cities)}
        li = city_idx.get(str(obs_d["location"]), 0)
        loc_oh[li] = 1.0

        cargo = np.zeros((len(self._products),), dtype=np.float32)
        pid_idx = {p: i for i, p in enumerate(self._products)}
        for pid, qty in obs_d["cargo"].items():
            idx = pid_idx.get(pid)
            if idx is not None:
                cargo[idx] = float(qty)

        buy_prices = np.zeros((len(self._products),), dtype=np.float32)
        for pid, price in obs_d["buy_prices"].items():
            idx = pid_idx.get(pid)
            if idx is not None:
                buy_prices[idx] = float(price)

        sell_prices = np.zeros((len(self._products),), dtype=np.float32)
        for pid, price in obs_d["sell_prices"].items():
            idx = pid_idx.get(pid)
            if idx is not None:
                sell_prices[idx] = float(price)

        v = np.concatenate(
            [
                np.array([day_norm, cash, durability_norm, capacity, debt], dtype=np.float32),
                loc_oh,
                cargo,
                buy_prices,
                sell_prices,
            ],
            dtype=np.float32,
        )
        return v

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._steps = 0
        self._state, self._rng, info0 = api.reset(seed=seed, game_mode=self.config.game_mode)
        obs = self._obs()
        info = {"api_info": info0}
        return obs, info

    # ---- 录制/编码辅助：将“结构化 Action”映射为离散 action index ----
    def sync_state_for_encoding(self, state: api.GameState, rng=None) -> None:  # type: ignore[name-defined]
        """
        让 MaskedEnv 使用外部 GameState（例如 GUI/CLI 人类游玩产生的状态）进行动作编码。

        注意：这不是一个可训练的“reset”；仅用于把外部状态挂到 env 上，然后调用：
        - action_mask()
        - encode_api_action(...)
        - _obs()
        """
        self._state = state  # type: ignore[assignment]
        if rng is not None:
            self._rng = rng  # type: ignore[assignment]

    def encode_api_action_with_reason(self, a: api.Action) -> tuple[Optional[int], str]:
        """
        将结构化 Action 编码为离散 index，并给出失败原因（用于 demo 录制诊断）。

        返回：
        - (idx, "")：成功编码
        - (None, reason)：无法编码
        """
        assert self._state is not None
        # 先刷新 candidates 缓存
        _ = self.action_mask()

        if isinstance(a, api.ActionNextDay) or isinstance(a, api.ActionNoop):
            return int(self._idx_next), ""

        # 录制动作空间不包含 repair 等“耗时多日的过程动作”
        if isinstance(a, api.ActionRepairTruck):
            return None, "repair 动作不在 MaskedEnv 动作空间中"

        if isinstance(a, api.ActionTravel):
            if a.mode == "land":
                if a.city in self._last_travel_land:
                    return int(self._idx_travel_land_start + self._last_travel_land.index(a.city)), ""
                return None, f"travel_land 目的地不在候选集合里: {a.city}"
            if a.mode == "sea":
                if a.city in self._last_travel_sea:
                    return int(self._idx_travel_sea_start + self._last_travel_sea.index(a.city)), ""
                return None, f"travel_sea 目的地不在候选集合里: {a.city}"
            return None, f"未知 travel mode: {a.mode}"

        if isinstance(a, api.ActionBorrow):
            from .loans import total_outstanding_principal

            total_debt_amount = sum(l.debt_total() for l in self._state.loans)
            net_assets = self._state.player.cash - total_debt_amount
            principal_total = total_outstanding_principal(self._state.loans)
            max_loan = max(0.0, net_assets - principal_total)
            if max_loan < 100.0:
                return None, f"可借额度不足: max_loan={max_loan:.2f}"
            ratio = float(a.amount) / max_loan if max_loan > 0 else 0.0
            best_i = 0
            best_err = abs(ratio - self.config.amount_fractions[0])
            for i, frac in enumerate(self.config.amount_fractions):
                err = abs(ratio - frac)
                if err < best_err:
                    best_err = err
                    best_i = i
            return int(self._idx_borrow_start + best_i), ""

        if isinstance(a, api.ActionRepay):
            if a.amount == "all":
                return int(self._idx_repay_all), ""
            total_debt = sum(l.debt_total() for l in self._state.loans)
            max_repay = min(self._state.player.cash, total_debt)
            if max_repay <= 0:
                return None, f"可还额度不足: max_repay={max_repay:.2f}"
            ratio = float(a.amount) / max_repay
            if ratio >= 0.99:
                return int(self._idx_repay_start + len(self.config.amount_fractions) - 1), ""
            best_i = 0
            best_err = abs(ratio - self.config.amount_fractions[0])
            for i, frac in enumerate(self.config.amount_fractions):
                err = abs(ratio - frac)
                if err < best_err:
                    best_err = err
                    best_i = i
            return int(self._idx_repay_start + best_i), ""

        if isinstance(a, api.ActionSell):
            pid = a.product_id
            if pid not in self._last_sell_cands:
                return None, f"sell 商品不在候选集合里: {pid}"
            qty = int(a.quantity)
            if qty <= 0:
                return None, "sell 数量<=0"
            obs = api.get_observation(self._state)
            have = int((obs.get("cargo") or {}).get(pid, 0))
            if have <= 0:
                return None, f"sell 无持仓: have={have}"
            match_variants: list[int] = []
            for vi in range(self._sell_qty_variants):
                if _compute_qty_variant(vi, have) == qty:
                    match_variants.append(vi)
            if not match_variants:
                return None, f"sell 数量不匹配 8 档规则: qty={qty} have={have}"
            for pref in (5, 6, 7, 4, 3, 2, 1, 0):
                if pref in match_variants:
                    v_idx = pref
                    break
            else:
                v_idx = match_variants[0]
            base_idx = self._last_sell_cands.index(pid)
            return int(self._idx_sell_start + base_idx * self._sell_qty_variants + int(v_idx)), ""

        if isinstance(a, api.ActionBuy):
            pid = a.product_id
            if pid not in self._last_buy_cands:
                return None, f"buy 商品不在候选集合里: {pid}"
            qty_req = int(a.quantity)
            if qty_req <= 0:
                return None, "buy 数量<=0"
            obs = api.get_observation(self._state)
            rem_capacity = int(obs["capacity"] - obs["cargo_used"])
            price = float((obs.get("buy_prices") or {}).get(pid) or 0.0)
            cash = float(obs.get("cash") or 0.0)
            max_by_cash = int(cash // price) if price > 0 else rem_capacity
            max_qty = max(0, min(rem_capacity, max_by_cash))
            match_variants: list[int] = []
            for vi in range(self._buy_qty_variants):
                if _compute_qty_variant(vi, max_qty) == qty_req:
                    match_variants.append(vi)
            if not match_variants:
                return None, f"buy 数量不匹配 8 档规则: qty={qty_req} max_qty={max_qty}"
            for pref in (5, 6, 7, 4, 3, 2, 1, 0):
                if pref in match_variants:
                    v_idx = pref
                    break
            else:
                v_idx = match_variants[0]
            base_idx = self._last_buy_cands.index(pid)
            return int(self._idx_buy_start + base_idx * self._buy_qty_variants + int(v_idx)), ""

        return None, f"不支持的动作类型: {type(a).__name__}"

    def encode_api_action(self, a: api.Action) -> Optional[int]:
        """
        将 `trade_game.api` 的结构化 Action 编码为当前 `Discrete(N)` 动作空间中的 index。
        若无法编码（不在候选/数量档不匹配等），返回 None。
        """
        idx, _reason = self.encode_api_action_with_reason(a)
        return idx

    def _buy_candidates(self) -> List[str]:
        assert self._state is not None
        obs = api.get_observation(self._state)
        pids = sorted(obs["buy_prices"].keys())
        return pids[: max(1, int(self.config.max_buy_choices))]

    def _sell_candidates(self) -> List[str]:
        assert self._state is not None
        obs = api.get_observation(self._state)
        cargo: Dict[str, int] = obs["cargo"]
        pids = sorted([pid for pid, qty in cargo.items() if qty > 0], key=lambda x: cargo[x], reverse=True)
        return pids[: max(1, int(self.config.max_sell_choices))]

    @staticmethod
    def _sea_rule_allowed(start: str, target: str, sea_departure_port: str) -> bool:
        port_cities = ("上海", "福州", "广州", "深圳", "海南", "台北", "高雄")
        if start not in port_cities:
            return False
        if start in ("上海", "福州", "广州", "深圳") and target not in ("海南", "台北", "高雄"):
            return False
        if start in ("海南", "台北", "高雄"):
            if target in ("海南", "台北", "高雄"):
                return True
            return bool(sea_departure_port) and target == sea_departure_port
        return True

    def _travel_candidates(self, mode: TransportMode) -> List[str]:
        assert self._state is not None
        p = self._state.player
        start = p.location
        if mode == TransportMode.LAND and p.truck_durability <= TRUCK_MIN_DURABILITY_FOR_TRAVEL:
            return []

        pairs: List[tuple[int, str]] = []
        for city in self._cities:
            if city == start:
                continue
            if mode == TransportMode.SEA and not self._sea_rule_allowed(start, city, p.sea_departure_port):
                continue
            try:
                validate_mode_allowed(mode, start, city)
                km = route_km(mode, start, city)
            except RouteNotFound:
                continue
            pairs.append((int(km), city))

        pairs.sort(key=lambda t: t[0])
        k = max(1, int(self.config.max_travel_choices))
        return [c for _, c in pairs[:k]]

    def action_mask(self) -> np.ndarray:
        """
        返回当前状态下每个离散动作是否可选的布尔数组，供 ActionMasker 使用。
        """
        assert self._state is not None
        mask = np.zeros(self._n_actions, dtype=bool)

        # 永远允许 next_day
        mask[self._idx_next] = True

        # 买/卖/旅行候选
        buy_cands = self._buy_candidates()
        sell_cands = self._sell_candidates()
        land_cands = self._travel_candidates(TransportMode.LAND)
        sea_cands = self._travel_candidates(TransportMode.SEA)

        self._last_buy_cands = buy_cands
        self._last_sell_cands = sell_cands
        self._last_travel_land = land_cands
        self._last_travel_sea = sea_cands

        # 需要容量和现金信息来判断哪些“商品×数量”组合是可行的
        obs = api.get_observation(self._state)
        rem_capacity = int(obs["capacity"] - obs["cargo_used"])
        buy_prices: Dict[str, float] = obs["buy_prices"]
        cash = float(obs["cash"])

        for bi, pid in enumerate(buy_cands):
            price = buy_prices.get(pid)
            if price is None:
                continue
            for vi in range(self._buy_qty_variants):
                slot = bi * self._buy_qty_variants + vi
                idx = self._idx_buy_start + slot
                if idx >= self._idx_buy_end:
                    continue
                max_by_cash = int(cash // float(price)) if float(price) > 0 else rem_capacity
                max_qty = max(0, min(rem_capacity, max_by_cash))
                qty = _compute_qty_variant(vi, max_qty)
                if qty <= 0:
                    continue
                cost = qty * price
                if cash >= cost:
                    mask[idx] = True
        cargo_dict = obs.get("cargo") or {}
        for si, pid in enumerate(sell_cands):
            have = int(cargo_dict.get(pid, 0))
            if have <= 0:
                continue
            for vi in range(self._sell_qty_variants):
                qty = _compute_qty_variant(vi, have)
                if qty <= 0:
                    continue
                slot = si * self._sell_qty_variants + vi
                idx = self._idx_sell_start + slot
                if idx < self._idx_sell_end:
                    mask[idx] = True
        for i in range(len(land_cands)):
            idx = self._idx_travel_land_start + i
            if idx < self._idx_travel_land_end:
                mask[idx] = True
        for i in range(len(sea_cands)):
            idx = self._idx_travel_sea_start + i
            if idx < self._idx_travel_sea_end:
                mask[idx] = True

        # 借贷 / 还款（按可借/可还额度的 1/3、2/3、3/3）
        from .loans import total_outstanding_principal

        p = self._state.player
        loc = p.location
        obs = api.get_observation(self._state)
        has_bank = obs["has_bank"]

        if has_bank:
            total_debt_amount = sum(l.debt_total() for l in self._state.loans)
            net_assets = p.cash - total_debt_amount
            principal_total = total_outstanding_principal(self._state.loans)
            max_loan = max(0.0, net_assets - principal_total)
            max_repay = min(p.cash, total_debt_amount) if self._state.loans else 0.0

            # borrow: 仅允许 1/3、2/3、3/3 可借额度且满足最小 100
            for i, frac in enumerate(self.config.amount_fractions):
                amt = max(100.0, round(frac * max_loan, 2))
                if principal_total + amt <= net_assets and amt >= 100.0:
                    idx = self._idx_borrow_start + i
                    if idx < self._idx_borrow_end:
                        mask[idx] = True

            # repay_all
            if self._state.loans:
                mask[self._idx_repay_all] = True
                # repay by fraction: 1/3、2/3、3/3 可还额度
                for i, frac in enumerate(self.config.amount_fractions):
                    amt = round(frac * max_repay, 2)
                    if amt >= 100.0 or (frac >= 0.99 and max_repay > 0):
                        idx = self._idx_repay_start + i
                        if idx < self._idx_repay_end:
                            mask[idx] = True

        return mask

    def action_mask_summary(self) -> Dict[str, Any]:
        """当前状态下各类合法动作数量，用于诊断（如开局是否有买入选项）。"""
        m = self.action_mask()
        return {
            "n_valid": int(m.sum()),
            "next_day": int(m[self._idx_next]),
            "buy": int(m[self._idx_buy_start : self._idx_buy_end].sum()),
            "sell": int(m[self._idx_sell_start : self._idx_sell_end].sum()),
            "travel_land": int(m[self._idx_travel_land_start : self._idx_travel_land_end].sum()),
            "travel_sea": int(m[self._idx_travel_sea_start : self._idx_travel_sea_end].sum()),
            "borrow": int(m[self._idx_borrow_start : self._idx_borrow_end].sum()),
            "repay_all": int(m[self._idx_repay_all]),
            "repay_amt": int(m[self._idx_repay_start : self._idx_repay_end].sum()),
        }

    def step(self, action: int):
        assert self._state is not None and self._rng is not None
        self._steps += 1

        cash_before = float(self._state.player.cash)
        debt_before = float(sum(l.debt_total() for l in self._state.loans))

        act_idx = int(action)
        idle_step = False

        # 解码离散动作
        if act_idx == self._idx_next:
            api_action = api.ActionNextDay()
            idle_step = True
            desc = "next_day"
        elif self._idx_buy_start <= act_idx < self._idx_buy_end:
            slot = act_idx - self._idx_buy_start
            base_idx = slot // self._buy_qty_variants
            v_idx = slot % self._buy_qty_variants
            if base_idx < len(self._last_buy_cands):
                pid = self._last_buy_cands[base_idx]
                # 重新根据当前容量计算数量（防止越界）
                obs = api.get_observation(self._state)
                rem_capacity = int(obs["capacity"] - obs["cargo_used"])
                price = float((obs.get("buy_prices") or {}).get(pid) or 0.0)
                cash = float(obs.get("cash") or 0.0)
                max_by_cash = int(cash // price) if price > 0 else rem_capacity
                max_qty = max(0, min(rem_capacity, max_by_cash))
                qty = _compute_qty_variant(v_idx, max_qty)
                price = obs["buy_prices"].get(pid)
                cash = float(obs["cash"])
                if qty > 0 and price is not None and cash >= qty * price:
                    api_action = api.ActionBuy(product_id=pid, quantity=qty)
                    desc = f"buy {pid} x{qty}"
                else:
                    api_action = api.ActionNextDay()
                    idle_step = True
                    desc = "idle_buy"
            else:
                api_action = api.ActionNextDay()
                idle_step = True
                desc = "idle_buy"
        elif self._idx_sell_start <= act_idx < self._idx_sell_end:
            slot = act_idx - self._idx_sell_start
            base_idx = slot // self._sell_qty_variants
            v_idx = slot % self._sell_qty_variants
            if base_idx < len(self._last_sell_cands):
                pid = self._last_sell_cands[base_idx]
                obs = api.get_observation(self._state)
                have = int((obs.get("cargo") or {}).get(pid, 0))
                qty = _compute_qty_variant(v_idx, have)
                if qty > 0:
                    api_action = api.ActionSell(product_id=pid, quantity=qty)
                    desc = f"sell {pid} x{qty}"
                else:
                    api_action = api.ActionNextDay()
                    idle_step = True
                    desc = "idle_sell"
            else:
                api_action = api.ActionNextDay()
                idle_step = True
                desc = "idle_sell"
        elif self._idx_travel_land_start <= act_idx < self._idx_travel_land_end:
            slot = act_idx - self._idx_travel_land_start
            if slot < len(self._last_travel_land):
                city = self._last_travel_land[slot]
                api_action = api.ActionTravel(city=city, mode="land")
                desc = f"travel {city} land"
            else:
                api_action = api.ActionNextDay()
                idle_step = True
                desc = "idle_travel_land"
        elif self._idx_travel_sea_start <= act_idx < self._idx_travel_sea_end:
            slot = act_idx - self._idx_travel_sea_start
            if slot < len(self._last_travel_sea):
                city = self._last_travel_sea[slot]
                api_action = api.ActionTravel(city=city, mode="sea")
                desc = f"travel {city} sea"
            else:
                api_action = api.ActionNextDay()
                idle_step = True
                desc = "idle_travel_sea"
        elif self._idx_borrow_start <= act_idx < self._idx_borrow_end:
            from .loans import total_outstanding_principal
            slot = act_idx - self._idx_borrow_start
            total_debt_amount = sum(l.debt_total() for l in self._state.loans)
            net_assets = self._state.player.cash - total_debt_amount
            principal_total = total_outstanding_principal(self._state.loans)
            max_loan = max(0.0, net_assets - principal_total)
            frac = self.config.amount_fractions[slot]
            amt = max(100.0, round(frac * max_loan, 2))
            api_action = api.ActionBorrow(amount=float(amt))
            desc = f"borrow {frac:.0%} max"
        elif act_idx == self._idx_repay_all:
            api_action = api.ActionRepay(amount="all")
            desc = "repay all"
        elif self._idx_repay_start <= act_idx < self._idx_repay_end:
            slot = act_idx - self._idx_repay_start
            total_debt = sum(l.debt_total() for l in self._state.loans)
            max_repay = min(self._state.player.cash, total_debt)
            frac = self.config.amount_fractions[slot]
            if frac >= 0.99:
                api_action = api.ActionRepay(amount="all")
                desc = "repay 3/3 (all)"
            else:
                amt = max(100.0, round(frac * max_repay, 2))
                api_action = api.ActionRepay(amount=float(amt))
                desc = f"repay {frac:.0%} max"
        else:
            api_action = api.ActionNextDay()
            idle_step = True
            desc = "next_day"

        _, _, done_api, info_api = api.step(self._state, self._rng, api_action)
        if info_api.get("error"):
            idle_step = True

        cash_after = float(self._state.player.cash)
        debt_after = float(sum(l.debt_total() for l in self._state.loans))
        net_before = cash_before - debt_before
        net_after = cash_after - debt_after
        reward = net_after - net_before

        terminated = bool(info_api.get("done_reason") == "bankruptcy") or bool(info_api.get("bankruptcy"))

        truncated = False
        if self._state.player.day >= self.config.max_days:
            truncated = True
        if self._steps >= self.config.max_steps:
            truncated = True
        if done_api and not terminated:
            truncated = True
        if idle_step and not terminated:
            reward -= 10.0

        obs = self._obs()
        info: Dict[str, Any] = {
            "api": info_api,
            "cash": float(self._state.player.cash),
            "day": int(self._state.player.day),
            "action_desc": desc,
        }

        return obs, reward, terminated, truncated, info

