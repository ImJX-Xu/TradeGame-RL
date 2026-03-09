from __future__ import annotations

import random
import sys
from dataclasses import asdict

from .state import GameState
from .data import CITIES, PRODUCTS
from .economy import purchase_price, refresh_daily_lambdas, sell_unit_price
from .inventory import CargoLot, add_lot, apply_transport_loss, cargo_used, remove_quantity_fifo, wipe_to_capacity_lifo
from .transport import RouteNotFound, TransportMode, route_km, sample_travel_days, validate_mode_allowed
from .timeflow import advance_one_day
from .loans import Bankruptcy, borrow, repay, total_outstanding_principal
from .save_load import list_saves, load_game, save_game
from .capacity_utils import (
    current_cargo_units,
    effective_truck_capacity,
    is_island_city,
    total_storage_capacity,
)


CITY_CODES = {
    "zz": "郑州",
    "sjz": "石家庄",
    "ty": "太原",
    "sy": "沈阳",
    "cc": "长春",
    "heb": "哈尔滨",
    "bj": "北京",
    "gz": "广州",
    "sz": "深圳",
    "fz": "福州",
    "sh": "上海",
    "hn": "海南",
    "tb": "台北",
    "gx": "高雄",
}


def _cities_sorted() -> list[str]:
    return sorted(CITIES.keys())


def resolve_city(token: str) -> str | None:
    t = token.strip()
    if not t:
        return None
    if t in CITIES:
        return t
    low = t.lower()
    if low in CITY_CODES:
        return CITY_CODES[low]
    # index: 1..N
    if low.isdigit():
        idx = int(low)
        cities = _cities_sorted()
        if 1 <= idx <= len(cities):
            return cities[idx - 1]
    return None


HELP = """\
可用命令：
  status        查看当前状态
  cities        列出所有场景（城市）
  products      列出所有商品（当前仅基础数据）
  prices        查看当前城市当日买/卖价（λ 已刷新）
  cargo         查看当前货物
  buy <id> <n>  在当前城市采购（n 为整数）
  sell <id> <n> 在当前城市售卖（n 为整数）
  ship          查看租船状态
  rent <ships> <days>   在港口租船（预付租期费用；可租多艘）
  addship <n>          在租船港加租船只（按剩余天数补缴）
  extend <days>        在租船港续租（所有船一起续）
  returnship           在租船港归还全部船只
  travel <city|code|index> <land|sea|land_fast|sea_fast>
                前往目标城市；*_fast 为快速出行（耗时约 1/FAST_TRAVEL_TIME_DIVISOR，费用×FAST_TRAVEL_COST_MULTIPLIER）
                例：travel bj land / travel 1 land_fast
  repair truck  维修货车
  loans         查看借贷
  borrow <amount>      银行借贷（上限≈当前净资产）
  repay <amount|all>   还款
  save <name>   手动存档
  load <name>   读取存档
  saves         列出存档
  next          跳到下一天（刷新 λ）
  help          查看帮助
  quit/exit     退出
"""


def main() -> int:
    # Windows 下尽量使用 UTF-8，避免中文输出乱码（若环境不支持则忽略）
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        sys.stdin.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass

    state = GameState()
    rng = random.Random()
    # 开局Day1：所有商品λ=0（previous_lambdas=None）
    state.daily_lambdas = refresh_daily_lambdas(rng, None)
    print("《风物千程》贸易模拟（CLI 原型）")
    print("输入 `help` 查看命令。")
    print()

    while True:
        p = state.player
        try:
            raw = input(f"[Day {p.day}] {p.location} | 现金 {p.cash:.0f}元 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if not raw:
            continue

        cmd = raw.lower()
        if cmd in ("quit", "exit"):
            return 0
        if cmd == "help":
            print(HELP)
            continue
        if cmd == "status":
            # 临时：先把状态打出来，后续会变成更友好的面板
            d = asdict(state.player)
            d["prices_ready"] = bool(state.daily_lambdas)
            d["cargo_used"] = cargo_used(state.player.cargo_lots)
            d["total_debt"] = sum(l.debt_total() for l in state.loans)
            print(d)
            continue
        if cmd == "cities":
            cities = _cities_sorted()
            code_rev = {v: k for k, v in CITY_CODES.items()}
            for i, name in enumerate(cities, 1):
                c = CITIES[name]
                modes = ",".join(m.value for m in sorted(c.modes, key=lambda x: x.value))
                flags = []
                if c.has_bank:
                    flags.append("银行")
                if c.has_port:
                    flags.append("港口")
                code = code_rev.get(name)
                extra = f" code={code}" if code else ""
                print(f"- {i:>2}. {name} ({modes}){extra}" + (f" [{'、'.join(flags)}]" if flags else ""))
            continue
        if cmd == "products":
            for pid in sorted(PRODUCTS.keys()):
                pdt = PRODUCTS[pid]
                origin = "全场景" if len(pdt.origins) == len(CITIES) else "、".join(sorted(pdt.origins))
                print(
                    f"- {pdt.name} ({pid}) | 类别={pdt.category.value} | 基础买={pdt.base_purchase_price:g} "
                    f"| 产地={origin}"
                )
            continue
        if cmd == "prices":
            city = state.player.location
            print(f"当日价格（地点：{city}）")
            for pid in sorted(PRODUCTS.keys()):
                pdt = PRODUCTS[pid]
                buy = purchase_price(pdt, city, state.daily_lambdas)
                sell = sell_unit_price(pdt, city, state.daily_lambdas, quantity_sold=1)
                buy_s = f"{buy:.2f}" if buy is not None else "不可买"
                print(f"- {pdt.name} | 买 {buy_s} | 卖 {sell:.2f}")
            continue
        if cmd == "cargo":
            lots = state.player.cargo_lots
            if not lots:
                print("当前无货物。")
                continue
            used = cargo_used(lots)
            cap = total_storage_capacity(state.player, state.player.location)
            print(f"货物占用：{used}/{cap}")
            for lot in lots:
                pdt = PRODUCTS[lot.product_id]
                print(f"- {pdt.name} ({lot.product_id}) x{lot.quantity} | 产地={lot.origin_city}")
            continue
        if cmd == "ship":
            p = state.player
            loc = p.location
            cap = total_storage_capacity(p, loc)
            if p.sea_departure_port:
                print(f"出海机制：当前在海岛，返程港={p.sea_departure_port}；总载重={cap}。")
            elif cap > 0:
                print(f"出海机制：当前在大陆海港，可出海至海岛；总载重={cap}。")
            else:
                print("当前城市无海港，无法出海。")
            continue

        if cmd.startswith("travel"):
            parts = raw.split()
            if len(parts) != 3:
                print("格式错误。示例：travel 北京 land / travel 海南 sea / travel 北京 land_fast")
                continue
            target = resolve_city(parts[1])
            mode_s = parts[2].lower()
            if not target or target not in CITIES:
                print("未知城市。用 `cities` 查看列表（可用 index 或 code）。")
                continue

            # 支持 land_fast / sea_fast（快速出行）
            fast = False
            if mode_s.endswith("_fast"):
                fast = True
                mode_core = mode_s[:-5]
            else:
                mode_core = mode_s

            if mode_core not in ("land", "sea"):
                print("运输方式必须是 land / sea / land_fast / sea_fast。")
                continue
            mode = TransportMode.LAND if mode_core == "land" else TransportMode.SEA

            p = state.player
            start = p.location
            if start == target:
                print("你已经在该城市。")
                continue

            # 维修门槛（耐久低于配置值不能出发）
            from trade_game.game_config import TRUCK_MIN_DURABILITY_FOR_TRAVEL
            if mode == TransportMode.LAND and p.truck_durability <= TRUCK_MIN_DURABILITY_FOR_TRAVEL:
                print(f"货车耐久度过低（≤{TRUCK_MIN_DURABILITY_FOR_TRAVEL:.0f}%），必须先维修才能出发。")
                continue
            if mode == TransportMode.SEA:
                if start not in ("上海", "福州", "广州", "深圳", "海南", "台北", "高雄"):
                    print("当前城市无法出海（非海港）。")
                    continue
                # 出海规则：大陆海港仅能去海岛；海岛仅能回 sea_departure_port
                if start in ("上海", "福州", "广州", "深圳") and target not in ("海南", "台北", "高雄"):
                    print("大陆海港出海仅能前往海岛（海南/台北/高雄）。")
                    continue
                if start in ("海南", "台北", "高雄"):
                    # 海岛可前往：其他海岛 或 返程大陆出海港
                    if target in ("海南", "台北", "高雄"):
                        pass  # 岛间航线允许
                    elif p.sea_departure_port and target == p.sea_departure_port:
                        pass  # 返程大陆允许
                    else:
                        print(f"海岛出海仅能前往其他海岛或返程港（{p.sea_departure_port or '未记录'}）。")
                        continue

            try:
                validate_mode_allowed(mode, start, target)
                km = route_km(mode, start, target)
            except RouteNotFound as e:
                print(f"无法到达：{e}")
                continue

            base_days = sample_travel_days(mode, km, rng)

            # --- 运输成本（按每趟） ---
            from trade_game.game_config import (
                LAND_COST_PER_KM,
                SEA_COST_PER_KM,
                TAIWAN_CUSTOMS,
                FAST_TRAVEL_COST_MULTIPLIER,
                FAST_TRAVEL_TIME_DIVISOR,
                FAST_TRAVEL_MIN_DAYS,
            )

            # 时间：快速出行减少耗时
            days = base_days
            if fast:
                days = max(FAST_TRAVEL_MIN_DAYS, int(days // FAST_TRAVEL_TIME_DIVISOR))

            if mode == TransportMode.LAND:
                truck_count = max(1, int(getattr(p, "truck_count", 1)))
                cost = km * LAND_COST_PER_KM * truck_count
            else:
                # 海运：费用 = (里程×单价 + 关税) × 载重乘数，载重乘数 = 1 + 当前货物/总载重
                is_taiwan_route = (start in ("台北", "高雄")) ^ (target in ("台北", "高雄"))
                customs = TAIWAN_CUSTOMS if is_taiwan_route else 0.0
                units = current_cargo_units(p)
                total_cap = total_storage_capacity(p, p.location)
                base_sea = km * SEA_COST_PER_KM + customs
                load_mult = 1.0 + (units / max(1, total_cap))
                cost = base_sea * load_mult

            # 费用：快速出行乘以价格倍数
            if fast:
                cost *= FAST_TRAVEL_COST_MULTIPLIER
            cost = round(cost, 2)

            if p.cash < cost:
                print(f"现金不足：本次运输成本 {cost:.2f}，当前 {p.cash:.2f}。")
                continue

            p.cash = round(p.cash - cost, 2)

            # --- 耐久损耗 ---
            if mode == TransportMode.LAND:
                from trade_game.game_config import TRUCK_DURABILITY_LOSS_PER_KM
                loss = km * TRUCK_DURABILITY_LOSS_PER_KM
                p.truck_durability = max(0.0, round(p.truck_durability - loss, 2))

            # --- 运输时间推进（逐日结算） ---
            try:
                for _ in range(days):
                    state, msgs = advance_one_day(state, rng)
                    for m in msgs:
                        print(m)
            except Bankruptcy as e:
                print(f"判定破产：{e}，游戏结束。")
                return 0

            # 运输损耗（每趟：电子×里程乘数、生鲜×时间乘数，实际再×0.9～1.1）
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
                print(f"运输损耗：损失 {lost} 单位货物。")

            p.location = target
            if mode == TransportMode.SEA:
                if start in ("上海", "福州", "广州", "深圳") and target in ("海南", "台北", "高雄"):
                    p.sea_departure_port = start
                elif start in ("海南", "台北", "高雄") and target in ("上海", "福州", "广州", "深圳"):
                    p.sea_departure_port = ""
            print(f"已到达：{target}（{mode.value}，里程 {km}km，用时 {days} 天，运输成本 {cost:.2f}）。")

            # 自动存档：每次运输结束
            try:
                save_game(state, "autosave")
            except Exception:
                pass
            continue

        if cmd.startswith("repair"):
            parts = raw.split()
            if len(parts) != 2:
                print("格式错误。示例：repair truck")
                continue
            what = parts[1].lower()
            if what == "truck":
                from trade_game.game_config import TRUCK_REPAIR_COST_BASE, TRUCK_REPAIR_DAYS
                truck_count = max(1, int(getattr(p, "truck_count", 1)))
                repair_percent = max(0, int(round(100.0 - p.truck_durability)))
                cost = round(TRUCK_REPAIR_COST_BASE * repair_percent * truck_count, 2)
                if repair_percent <= 0:
                    print("无需维修。")
                    continue
                if p.cash < cost:
                    print("现金不足。")
                    continue
                p.cash = round(p.cash - cost, 2)
                p.truck_durability = 100.0
                for _ in range(TRUCK_REPAIR_DAYS):
                    advance_one_day(state, rng)
                print(f"货车已维修至 100%，花费 {cost:,.0f} 元，时间流逝 {TRUCK_REPAIR_DAYS} 天。")
                continue
            print("未知维修对象。")
            continue

        if cmd == "loans":
            if not state.loans:
                print("当前无借贷。")
                continue
            for i, l in enumerate(state.loans, 1):
                print(
                    f"- #{i} 本金={l.principal:.2f} 利息={l.accrued_interest:.2f} 滞纳金={l.late_fees:.2f} "
                    f"模式={l.interest_mode} 逾期={l.overdue_days}天 总欠款={l.debt_total():.2f}"
                )
            continue

        if cmd.startswith("borrow"):
            parts = raw.split()
            if len(parts) != 2:
                print("格式错误。示例：borrow 10000")
                continue
            if not CITIES[p.location].has_bank:
                print("当前城市没有银行，无法借贷。")
                continue
            try:
                amount = float(parts[1])
            except ValueError:
                print("金额必须是数字。")
                continue

            # 借贷上限：净资产 = 总现金 - 总债务，可借金额 ≤ 净资产（可为负）
            total_debt_amount = sum(l.debt_total() for l in state.loans)
            net_assets = p.cash - total_debt_amount
            principal_total = total_outstanding_principal(state.loans)
            if principal_total + amount > net_assets:
                print(f"超出借贷额度：当前净资产 {net_assets:.2f}，已借本金 {principal_total:.2f}。")
                continue

            loan = borrow(state.loans, amount=amount, day=p.day, interest_mode="simple")
            p.cash = round(p.cash + amount, 2)
            from trade_game.game_config import LOAN_DAILY_INTEREST_RATE
            rate_pct = LOAN_DAILY_INTEREST_RATE * 100
            print(f"借贷成功：{amount:.2f} 元（日利率 {rate_pct:.1f}%，可随时还款）。")
            continue

        if cmd.startswith("repay"):
            parts = raw.split()
            if len(parts) != 2:
                print("格式错误。示例：repay all / repay 5000")
                continue
            if not state.loans:
                print("当前无借贷。")
                continue
            if not CITIES[p.location].has_bank:
                print("当前城市没有银行，无法还款。")
                continue
            arg = parts[1].lower()
            amt = None
            if arg != "all":
                try:
                    amt = float(arg)
                except ValueError:
                    print("金额必须是数字或 all。")
                    continue
            before = p.cash
            p.cash = repay(state.loans, cash=p.cash, amount=amt)
            spent = round(before - p.cash, 2)
            print(f"已还款：{spent:.2f} 元。")
            continue

        if cmd == "saves":
            names = list_saves()
            if not names:
                print("暂无存档。")
            else:
                for n in names:
                    print(f"- {n}")
            continue
        if cmd.startswith("save"):
            parts = raw.split(maxsplit=1)
            name = parts[1].strip() if len(parts) == 2 else "save"
            path = save_game(state, name)
            print(f"已存档：{path.name}")
            continue
        if cmd.startswith("load"):
            parts = raw.split(maxsplit=1)
            if len(parts) != 2:
                print("格式错误。示例：load autosave")
                continue
            name = parts[1].strip()
            try:
                state = load_game(name)
                print(f"已读取存档：{name}")
            except FileNotFoundError:
                print("未找到该存档。")
            continue

        parts = raw.split()
        if parts and parts[0].lower() in ("buy", "sell"):
            if len(parts) != 3:
                print("格式错误。示例：buy rice 10 / sell rice 10")
                continue
            action = parts[0].lower()
            pid = parts[1]
            try:
                qty = int(parts[2])
            except ValueError:
                print("数量必须是整数。")
                continue
            if qty <= 0:
                print("数量必须大于 0。")
                continue
            if pid not in PRODUCTS:
                print("未知商品 id。用 `products` 查看列表。")
                continue

            city = state.player.location
            pdt = PRODUCTS[pid]

            if action == "buy":
                unit = purchase_price(pdt, city, state.daily_lambdas)
                if unit is None:
                    print("该城市无法采购此商品（仅产地可买）。")
                    continue
                used = cargo_used(state.player.cargo_lots)
                cap = total_storage_capacity(state.player, state.player.location)
                if used + qty > cap:
                    print(f"容量不足：当前 {used}/{cap}，本次需要 {qty}。")
                    continue
                cost = round(unit * qty, 2)
                if state.player.cash < cost:
                    print(f"现金不足：需要 {cost:.2f}，当前 {state.player.cash:.2f}。")
                    continue

                state.player.cash = round(state.player.cash - cost, 2)
                add_lot(
                    state.player.cargo_lots,
                    CargoLot(product_id=pid, quantity=qty, origin_city=city, shelf_life_remaining_days=None),
                )
                print(f"已采购：{pdt.name} x{qty}，单价 {unit:.2f}，共 {cost:.2f}。")
                continue

            # sell
            have = sum(l.quantity for l in state.player.cargo_lots if l.product_id == pid)
            if have <= 0:
                print("你没有该商品。")
                continue
            sell_qty = qty if qty <= have else have
            actual_qty, removed_lots = remove_quantity_fifo(state.player.cargo_lots, pid, sell_qty)
            if actual_qty <= 0:
                print("售卖失败。")
                continue
            unit = sell_unit_price(
                pdt,
                city,
                state.daily_lambdas,
                quantity_sold=actual_qty,
                shelf_life_remaining_days=None,
            )
            revenue = round(unit * actual_qty, 2)
            state.player.cash = round(state.player.cash + revenue, 2)
            print(f"已售卖：{pdt.name} x{actual_qty}，单价 {unit:.2f}，共 {revenue:.2f}。")
            continue
        if cmd == "next":
            try:
                state, msgs = advance_one_day(state, rng)
                for m in msgs:
                    print(m)
                print(f"已进入第 {state.player.day} 天（价格波动已刷新）。")
            except Bankruptcy as e:
                print(f"判定破产：{e}，游戏结束。")
                return 0
            continue

        print("未知命令。输入 `help` 查看可用命令。")

