"""
对外暴露的「算法 / 环境」接口模块。

提供标准的 RL 风格环境接口，便于策略/算法验证与批量仿真，不依赖 CLI 或图形界面。

核心接口：
- reset(seed, game_mode) -> (state, rng, info)
- step(state, rng, action) -> (state, reward, done, info)
- get_observation(state) -> dict
- new_game / advance_days：保留原有轻量接口

动作类型见 Action 及下方 ACTION_SPACE 说明。
"""

from __future__ import annotations

__all__ = [
    "reset",
    "step",
    "get_observation",
    "get_valid_cities",
    "get_valid_product_ids",
    "new_game",
    "advance_days",
    "ACTION_SPACE",
    "Action",
    "ActionNoop",
    "ActionNextDay",
    "ActionBuy",
    "ActionSell",
    "ActionTravel",
    "ActionRepairTruck",
    "ActionBorrow",
    "ActionRepay",
]

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Tuple, Union

from .data import CITIES, PRODUCTS
from .economy import purchase_price, refresh_daily_lambdas, sell_unit_price
from .game_config import (
    LAND_COST_PER_KM,
    SEA_COST_PER_KM,
    TAIWAN_CUSTOMS,
    TRUCK_DURABILITY_LOSS_PER_KM,
    TRUCK_MIN_DURABILITY_FOR_TRAVEL,
    TRUCK_REPAIR_COST_BASE,
    TRUCK_REPAIR_DAYS,
)
from .train_config import CROSS_CITY_SELL_BONUS, SAME_CITY_SELL_PENALTY
from .inventory import CargoLot, add_lot, apply_transport_loss, cargo_used, remove_quantity_fifo
from .loans import Bankruptcy, borrow, estimated_assets, repay, total_outstanding_principal
from .state import GameState
from .timeflow import advance_one_day
from .transport import RouteNotFound, TransportMode, route_km, sample_travel_days, validate_mode_allowed
from .capacity_utils import current_cargo_units, total_storage_capacity


# ---------------------------------------------------------------------------
# 动作类型（结构化，便于算法生成与序列化）
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ActionNoop:
    """无操作（占位或跳过）。"""
    type: Literal["noop"] = "noop"


@dataclass(frozen=True)
class ActionNextDay:
    """推进 1 天：刷新 λ、计息、人力成本、破产判定。"""
    type: Literal["next_day"] = "next_day"


@dataclass(frozen=True)
class ActionBuy:
    """在当前城市采购商品。"""
    type: Literal["buy"] = "buy"
    product_id: str = ""
    quantity: int = 0


@dataclass(frozen=True)
class ActionSell:
    """在当前城市售卖商品。"""
    type: Literal["sell"] = "sell"
    product_id: str = ""
    quantity: int = 0


@dataclass(frozen=True)
class ActionTravel:
    """前往目标城市（陆运或海运）。city 为中文城市名。"""
    type: Literal["travel"] = "travel"
    city: str = ""
    mode: Literal["land", "sea"] = "land"


@dataclass(frozen=True)
class ActionRepairTruck:
    """维修货车（花费金钱与天数）。"""
    type: Literal["repair_truck"] = "repair_truck"


@dataclass(frozen=True)
class ActionBorrow:
    """在银行借贷。"""
    type: Literal["borrow"] = "borrow"
    amount: float = 0.0


@dataclass(frozen=True)
class ActionRepay:
    """还款。amount 为正数金额，或使用 "all" 表示全部还清。"""
    type: Literal["repay"] = "repay"
    amount: Union[float, Literal["all"]] = 0.0


Action = Union[
    ActionNoop,
    ActionNextDay,
    ActionBuy,
    ActionSell,
    ActionTravel,
    ActionRepairTruck,
    ActionBorrow,
    ActionRepay,
]


# ---------------------------------------------------------------------------
# 动作空间说明（供算法/文档使用）
# ---------------------------------------------------------------------------

ACTION_SPACE = """
离散 + 参数化动作（扁平化后可用于策略网络或规则）：

- noop: 无操作
- next_day: 推进 1 天
- buy: product_id (str), quantity (int) — 商品 id 见 data.PRODUCTS
- sell: product_id (str), quantity (int)
- travel: city (str, 中文城市名), mode ("land" | "sea")
- repair_truck: 无参数
- borrow: amount (float)
- repay: amount (float | "all")

城市名必须与 data.CITIES 的 key 一致（如 "郑州", "北京", "海南"）。
"""


def get_valid_cities() -> List[str]:
    """所有可用的城市名（与 CITIES 的 key 一致），供算法枚举 travel 目标。"""
    return sorted(CITIES.keys())


def get_valid_product_ids() -> List[str]:
    """所有商品 id，供算法枚举 buy/sell。"""
    return sorted(PRODUCTS.keys())


def _wealth(state: GameState) -> float:
    """当前财富：现金 + 货物预估价值（与贷款模块的 estimated_assets 一致）。"""
    return estimated_assets(state.player.cash, state.player.cargo_lots)


def _is_done(state: GameState) -> Tuple[bool, str]:
    """
    判定是否结束。返回 (done, reason)。
    reason 用于 info["done_reason"]。
    """
    from .train_config import get_max_days
    max_days = get_max_days(state.game_mode)
    if max_days is not None and state.player.day >= max_days:
        return True, "time_up"
    # 破产在 step 内通过捕获 Bankruptcy 设置，这里不重复判定
    return False, ""


# ---------------------------------------------------------------------------
# 原有轻量接口（保留）
# ---------------------------------------------------------------------------

def new_game(seed: int | None = None, *, game_mode: str = "free") -> Tuple[GameState, random.Random]:
    """
    创建一个新的游戏状态和 RNG，用于算法/批量仿真。
    """
    rng = random.Random(seed)
    state = GameState()
    state.game_mode = game_mode
    state.daily_lambdas = refresh_daily_lambdas(rng, None)
    return state, rng


def advance_days(state: GameState, rng: random.Random, days: int = 1) -> Tuple[GameState, List[str]]:
    """在不依赖任何前端的情况下推进若干天。"""
    if days <= 0:
        return state, []
    all_msgs: List[str] = []
    for _ in range(days):
        state, msgs = advance_one_day(state, rng)
        all_msgs.extend(msgs)
    return state, all_msgs


# ---------------------------------------------------------------------------
# 标准环境接口
# ---------------------------------------------------------------------------

def reset(
    seed: int | None = None,
    *,
    game_mode: str = "free",
) -> Tuple[GameState, random.Random, Dict[str, Any]]:
    """
    重置环境，开始新一局。

    Returns:
        state: 初始游戏状态
        rng: 随机数生成器（可控种子）
        info: 包含 observation 等，供算法使用
    """
    state, rng = new_game(seed, game_mode=game_mode)
    info: Dict[str, Any] = {
        "observation": get_observation(state),
        "day": state.player.day,
        "wealth": _wealth(state),
    }
    return state, rng, info


def get_observation(state: GameState) -> Dict[str, Any]:
    """
    从当前状态提取可观测信息（便于算法输入）。

    包含：日期、位置、现金、载重、耐久、借贷、当日价格摘要、货物摘要等。
    不包含完整价格表/路线图，算法如需可自行读 data.PRODUCTS / CITIES。
    """
    p = state.player
    obs: Dict[str, Any] = {
        "day": p.day,
        "location": p.location,
        "cash": round(p.cash, 2),
        "cargo_used": cargo_used(p.cargo_lots),
        "capacity": total_storage_capacity(p, p.location),
        "truck_durability": round(p.truck_durability, 2),
        "truck_count": getattr(p, "truck_count", 1),
        "sea_departure_port": p.sea_departure_port or None,
        "game_mode": state.game_mode,
        "total_debt": round(sum(l.debt_total() for l in state.loans), 2),
        "has_bank": CITIES[p.location].has_bank,
        "has_port": CITIES[p.location].has_port,
    }
    # 当前城市可买/可卖商品 id 及当日买价/卖价（简化）
    buy_prices: Dict[str, float] = {}
    sell_prices: Dict[str, float] = {}
    for pid, product in PRODUCTS.items():
        buy_p = purchase_price(product, p.location, state.daily_lambdas)
        if buy_p is not None:
            buy_prices[pid] = round(buy_p, 2)
        sell_prices[pid] = round(sell_unit_price(product, p.location, state.daily_lambdas, quantity_sold=1), 2)
    obs["buy_prices"] = buy_prices
    obs["sell_prices"] = sell_prices
    # 当前货物：product_id -> quantity
    cargo_summary: Dict[str, int] = {}
    for lot in p.cargo_lots:
        cargo_summary[lot.product_id] = cargo_summary.get(lot.product_id, 0) + lot.quantity
    obs["cargo"] = cargo_summary
    return obs


def step(
    state: GameState,
    rng: random.Random,
    action: Action,
) -> Tuple[GameState, float, bool, Dict[str, Any]]:
    """
    执行一步动作，返回 (state, reward, done, info)。

    - state: 原地修改后返回同一对象（便于连续 step）。
    - reward: 本步带来的财富变化（wealth_after - wealth_before）；无效动作为 0。
    - done: 是否结束（破产或挑战模式到期）。
    - info: messages, observation, done_reason, wealth 等；无效动作时含 "error"。

    推进天数的操作（列出来）：
    - 推进 1 天：next_day、noop、buy、sell、borrow、repay/repay_all（以上执行后均 +1 天）
    - 推进多天：travel（路线所需天数）、repair（维修 TRUCK_REPAIR_DAYS 天）
    """
    info: Dict[str, Any] = {"messages": []}
    wealth_before = _wealth(state)
    p = state.player

    def fail(msg: str) -> Tuple[GameState, float, bool, Dict[str, Any]]:
        info["error"] = msg
        info["observation"] = get_observation(state)
        info["wealth"] = _wealth(state)
        done, reason = _is_done(state)
        info["done_reason"] = reason
        return state, 0.0, done, info

    def ok(msgs: List[str], reward_offset: float = 0.0) -> Tuple[GameState, float, bool, Dict[str, Any]]:
        info["messages"] = msgs
        wealth_after = _wealth(state)
        reward = round(wealth_after - wealth_before + reward_offset, 2)
        done, reason = _is_done(state)
        info["observation"] = get_observation(state)
        info["wealth"] = wealth_after
        info["wealth_delta"] = reward
        info["done_reason"] = reason
        return state, reward, done, info

    def ok_from_wealth(
        msgs: List[str],
        wealth_ref: float,
        reward_offset: float = 0.0,
    ) -> Tuple[GameState, float, bool, Dict[str, Any]]:
        """
        以给定的 wealth_ref 作为本步“基准财富”来计算 reward。
        用于希望剥离某些资金流（如借贷/还款本金）的奖励影响场景。
        """
        info["messages"] = msgs
        wealth_after = _wealth(state)
        reward = round(wealth_after - wealth_ref + reward_offset, 2)
        done, reason = _is_done(state)
        info["observation"] = get_observation(state)
        info["wealth"] = wealth_after
        info["wealth_delta"] = reward
        info["done_reason"] = reason
        return state, reward, done, info

    def advance_then_ok(msgs: List[str], reward_offset: float = 0.0) -> Tuple[GameState, float, bool, Dict[str, Any]]:
        """执行 advance_one_day 后返回 ok；若推进当天触发破产则返回 done=True。"""
        try:
            _, day_msgs = advance_one_day(state, rng)
            return ok(msgs + day_msgs, reward_offset=reward_offset)
        except Bankruptcy as e:
            info["bankruptcy"] = str(e)
            info["done_reason"] = "bankruptcy"
            info["observation"] = get_observation(state)
            info["wealth"] = _wealth(state)
            return state, round(_wealth(state) - wealth_before, 2), True, info

    # ----- 分发动作 -----
    # 以下操作原本不耗时，现统一改为执行后推进 1 天：noop, buy, sell, borrow, repay/repay_all
    if isinstance(action, ActionNoop):
        return advance_then_ok(["noop"])

    if isinstance(action, ActionNextDay):
        try:
            state, msgs = advance_one_day(state, rng)
            return ok(msgs + [f"进入第 {state.player.day} 天"])
        except Bankruptcy as e:
            info["bankruptcy"] = str(e)
            info["done_reason"] = "bankruptcy"
            info["observation"] = get_observation(state)
            info["wealth"] = _wealth(state)
            return state, round(_wealth(state) - wealth_before, 2), True, info

    if isinstance(action, ActionBuy):
        pid, qty = action.product_id, action.quantity
        if pid not in PRODUCTS:
            return fail("未知商品 id")
        if qty <= 0:
            return fail("数量必须大于 0")
        city = p.location
        product = PRODUCTS[pid]
        unit = purchase_price(product, city, state.daily_lambdas)
        if unit is None:
            return fail("该城市无法采购此商品")
        used = cargo_used(p.cargo_lots)
        cap = total_storage_capacity(p, p.location)
        if used + qty > cap:
            return fail(f"容量不足：当前 {used}/{cap}，需要 {qty}")
        cost = round(unit * qty, 2)
        if p.cash < cost:
            return fail(f"现金不足：需要 {cost:.2f}")
        p.cash = round(p.cash - cost, 2)
        add_lot(
            p.cargo_lots,
            CargoLot(product_id=pid, quantity=qty, origin_city=city, shelf_life_remaining_days=None),
        )
        return advance_then_ok([f"采购 {product.name} x{qty}，花费 {cost:.2f}"])

    if isinstance(action, ActionSell):
        pid, qty = action.product_id, action.quantity
        if pid not in PRODUCTS:
            return fail("未知商品 id")
        if qty <= 0:
            return fail("数量必须大于 0")
        have = sum(l.quantity for l in p.cargo_lots if l.product_id == pid)
        if have <= 0:
            return fail("没有该商品")
        sell_qty = min(qty, have)
        actual_qty, removed_lots = remove_quantity_fifo(p.cargo_lots, pid, sell_qty)
        if actual_qty <= 0:
            return fail("售卖失败")
        product = PRODUCTS[pid]
        unit = sell_unit_price(product, p.location, state.daily_lambdas, quantity_sold=actual_qty, shelf_life_remaining_days=None)
        revenue = round(unit * actual_qty, 2)
        p.cash = round(p.cash + revenue, 2)
        # 奖励塑形：跨城卖出给奖励，同城卖出给惩罚（按本次卖出批次的产地区分）
        sell_city = p.location
        same_city_qty = sum(lot.quantity for lot in removed_lots if lot.origin_city == sell_city)
        cross_city_qty = actual_qty - same_city_qty
        reward_offset = CROSS_CITY_SELL_BONUS * cross_city_qty - SAME_CITY_SELL_PENALTY * same_city_qty
        return advance_then_ok([f"售卖 {product.name} x{actual_qty}，收入 {revenue:.2f}"], reward_offset=reward_offset)

    if isinstance(action, ActionTravel):
        target = action.city.strip()
        if target not in CITIES:
            return fail("未知城市")
        mode = TransportMode.LAND if action.mode == "land" else TransportMode.SEA
        start = p.location
        if start == target:
            return fail("已在目标城市")
        if mode == TransportMode.LAND and p.truck_durability <= TRUCK_MIN_DURABILITY_FOR_TRAVEL:
            return fail(f"货车耐久度过低（≤{TRUCK_MIN_DURABILITY_FOR_TRAVEL:.0f}%），请先维修")
        port_cities = ("上海", "福州", "广州", "深圳", "海南", "台北", "高雄")
        if mode == TransportMode.SEA:
            if start not in port_cities:
                return fail("当前城市无法出海")
            if start in ("上海", "福州", "广州", "深圳") and target not in ("海南", "台北", "高雄"):
                return fail("大陆海港仅能前往海岛")
            if start in ("海南", "台北", "高雄"):
                if target not in ("海南", "台北", "高雄") and (not p.sea_departure_port or target != p.sea_departure_port):
                    return fail(f"海岛仅能前往其他海岛或返程港 {p.sea_departure_port or '未记录'}")
        try:
            validate_mode_allowed(mode, start, target)
            km = route_km(mode, start, target)
        except RouteNotFound as e:
            return fail(str(e))
        days = sample_travel_days(mode, km, rng)
        if mode == TransportMode.LAND:
            truck_count = max(1, getattr(p, "truck_count", 1))
            cost = round(km * LAND_COST_PER_KM * truck_count, 2)
        else:
            is_taiwan = (start in ("台北", "高雄")) ^ (target in ("台北", "高雄"))
            customs = TAIWAN_CUSTOMS if is_taiwan else 0.0
            units = current_cargo_units(p)
            total_cap = max(1, total_storage_capacity(p, p.location))
            load_mult = 1.0 + (units / total_cap)
            cost = round((km * SEA_COST_PER_KM + customs) * load_mult, 2)
        if p.cash < cost:
            return fail(f"现金不足：运输成本 {cost:.2f}")
        p.cash = round(p.cash - cost, 2)
        if mode == TransportMode.LAND:
            p.truck_durability = max(0.0, round(p.truck_durability - km * TRUCK_DURABILITY_LOSS_PER_KM, 2))
        try:
            for _ in range(days):
                state, msgs = advance_one_day(state, rng)
                info["messages"].extend(msgs)
        except Bankruptcy as e:
            info["bankruptcy"] = str(e)
            info["done_reason"] = "bankruptcy"
            info["observation"] = get_observation(state)
            info["wealth"] = _wealth(state)
            p.location = target
            if mode == TransportMode.SEA:
                if start in ("上海", "福州", "广州", "深圳") and target in ("海南", "台北", "高雄"):
                    p.sea_departure_port = start
                elif start in ("海南", "台北", "高雄") and target in ("上海", "福州", "广州", "深圳"):
                    p.sea_departure_port = ""
            return state, round(_wealth(state) - wealth_before, 2), True, info
        lost = apply_transport_loss(
            p.cargo_lots,
            origin_city=start,
            target_city=target,
            km=float(km),
            days=days,
            rng=rng,
            loss_stats=None,
        )
        if lost > 0:
            info["messages"].append(f"运输损耗 {lost} 单位")
        p.location = target
        if mode == TransportMode.SEA:
            if start in ("上海", "福州", "广州", "深圳") and target in ("海南", "台北", "高雄"):
                p.sea_departure_port = start
            elif start in ("海南", "台北", "高雄") and target in ("上海", "福州", "广州", "深圳"):
                p.sea_departure_port = ""
        info["messages"].append(f"到达 {target}，{mode.value}，{km}km，{days} 天，成本 {cost:.2f}")
        return ok(info["messages"])

    if isinstance(action, ActionRepairTruck):
        truck_count = max(1, getattr(p, "truck_count", 1))
        repair_pct = max(0, int(round(100.0 - p.truck_durability)))
        if repair_pct <= 0:
            return fail("无需维修")
        cost = round(TRUCK_REPAIR_COST_BASE * repair_pct * truck_count, 2)
        if p.cash < cost:
            return fail("现金不足")
        p.cash = round(p.cash - cost, 2)
        p.truck_durability = 100.0
        for _ in range(TRUCK_REPAIR_DAYS):
            state, msgs = advance_one_day(state, rng)
            info["messages"].extend(msgs)
        info["messages"].append(f"货车已维修，花费 {cost:.0f}，耗时 {TRUCK_REPAIR_DAYS} 天")
        return ok(info["messages"])

    if isinstance(action, ActionBorrow):
        if not CITIES[p.location].has_bank:
            return fail("当前城市没有银行")
        amount = action.amount
        if amount <= 0:
            return fail("借贷金额必须大于 0")
        # 借贷上限：净资产 = 总现金 - 总债务，可借金额 ≤ 净资产（可为负）
        total_debt_amount = sum(l.debt_total() for l in state.loans)
        net_assets = p.cash - total_debt_amount
        principal_total = total_outstanding_principal(state.loans)
        if principal_total + amount > net_assets:
            return fail(f"超出借贷额度：净资产 {net_assets:.2f}，已借本金 {principal_total:.2f}")
        borrow(state.loans, amount=amount, day=p.day, interest_mode="simple")
        p.cash = round(p.cash + amount, 2)
        # 不将“借到的本金”计入 reward：以借贷完成后的 wealth 作为基准，仅计算推进一天带来的影响（利息/成本等）。
        wealth_mid = _wealth(state)
        try:
            _, day_msgs = advance_one_day(state, rng)
            return ok_from_wealth([f"借贷 {amount:.2f} 元"] + day_msgs, wealth_ref=wealth_mid)
        except Bankruptcy as e:
            info["bankruptcy"] = str(e)
            info["done_reason"] = "bankruptcy"
            info["observation"] = get_observation(state)
            info["wealth"] = _wealth(state)
            return state, round(_wealth(state) - wealth_mid, 2), True, info

    if isinstance(action, ActionRepay):
        if not state.loans:
            return fail("当前无借贷")
        if not CITIES[p.location].has_bank:
            return fail("当前城市没有银行")
        amt: float | None = None if action.amount == "all" else float(action.amount)
        before = p.cash
        p.cash = repay(state.loans, cash=p.cash, amount=amt)
        spent = round(before - p.cash, 2)
        # 不将“还掉的本金”计入 reward：以还款完成后的 wealth 作为基准，仅计算推进一天带来的影响（利息/成本等）。
        wealth_mid = _wealth(state)
        try:
            _, day_msgs = advance_one_day(state, rng)
            return ok_from_wealth([f"还款 {spent:.2f} 元"] + day_msgs, wealth_ref=wealth_mid)
        except Bankruptcy as e:
            info["bankruptcy"] = str(e)
            info["done_reason"] = "bankruptcy"
            info["observation"] = get_observation(state)
            info["wealth"] = _wealth(state)
            return state, round(_wealth(state) - wealth_mid, 2), True, info

    return fail("未知动作类型")
