from __future__ import annotations

import random
from typing import List, Tuple

from .economy import purchase_price, refresh_daily_lambdas, sell_unit_price
from .loans import Bankruptcy, force_collect_if_needed, process_one_day, total_outstanding_principal
from .state import GameState
from .game_config import DAILY_LABOR_PER_TRUCK


def advance_one_day(state: GameState, rng: random.Random) -> Tuple[GameState, List[str]]:
    """
    推进 1 天（用于运输途中/手动 next）：
    - 贷款计息/逾期/强制扣款
    - 车辆每日人力成本（每辆车每天固定扣除）
    - 天数 +1，刷新当日 λ
    """
    p = state.player
    msgs: List[str] = []

    # 借贷计息（先按“旧的一天”计息）
    p.cash, loan_msgs = process_one_day(state.loans, cash=p.cash)
    msgs.extend(loan_msgs)

    # 车辆每日人力成本：仅“额外车辆”计费（首辆车不计）
    truck_count = max(1, int(getattr(p, "truck_count", 1)))
    extra_trucks = max(0, truck_count - 1)
    labor_cost = int(DAILY_LABOR_PER_TRUCK * extra_trucks)
    if labor_cost > 0:
        p.cash -= labor_cost
        msgs.append(
            f"车辆人力成本：-{labor_cost:,.0f} 元（额外 {extra_trucks} 辆 × {DAILY_LABOR_PER_TRUCK:.0f} 元/天；首辆车免）。"
        )

    # 强制扣款（若触发）
    try:
        p.cash, p.cargo_lots, forced_msgs = force_collect_if_needed(
            state.loans, cash=p.cash, cargo_lots=p.cargo_lots
        )
        msgs.extend(forced_msgs)
    except Bankruptcy as e:
        raise Bankruptcy(str(e)) from e

    # 破产判定：无任何货物 + 无贷款额度 + 金额<=0 同时成立
    cash_nonpos = p.cash <= 0
    no_cargo = len(p.cargo_lots) == 0
    # 净资产 = 总现金 - 总债务，可借金额 ≤ 净资产（可为负）
    total_debt_amount = sum(l.debt_total() for l in state.loans)
    net_assets = p.cash - total_debt_amount
    principal_total = total_outstanding_principal(state.loans)
    max_loan = max(0.0, net_assets - principal_total)
    no_loan_capacity = max_loan <= 0
    if cash_nonpos and no_cargo and no_loan_capacity:
        raise Bankruptcy("资金耗尽（无货物、无贷款额度）")

    # 下一天
    p.day += 1
    # 保存昨日λ，然后刷新今日λ（带惯性波动），并更新 7 日价格历史
    state.previous_lambdas = state.daily_lambdas.copy()
    state.daily_lambdas = refresh_daily_lambdas(rng, state.previous_lambdas)

    # 更新 7 日价格历史（对所有城市记录采购价/售卖价）
    from .data import PRODUCTS, CITIES
    for city in CITIES.keys():
        for pid, product in PRODUCTS.items():
            key = f"{city}|{pid}"

            # 采购价历史（仅产地城市有采购价）
            try:
                buy_price = purchase_price(product, city, state.daily_lambdas)
            except Exception:
                buy_price = None
            if buy_price is not None and buy_price > 0:
                hist_buy = state.price_history_buy_7d.get(key, [])
                hist_buy = list(hist_buy)[-6:]  # 保留最近 6 天
                hist_buy.append(round(buy_price, 2))
                state.price_history_buy_7d[key] = hist_buy

            # 售卖价历史（理论售价始终可计算）
            try:
                sell_price = sell_unit_price(product, city, state.daily_lambdas, quantity_sold=1)
            except Exception:
                continue
            if sell_price <= 0:
                continue
            hist_sell = state.price_history_sell_7d.get(key, [])
            hist_sell = list(hist_sell)[-6:]
            hist_sell.append(round(sell_price, 2))
            state.price_history_sell_7d[key] = hist_sell

    return state, msgs

