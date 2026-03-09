from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .data import PRODUCTS
from .game_config import LOAN_DAILY_INTEREST_RATE
from .inventory import CargoLot


class Bankruptcy(Exception):
    pass


@dataclass(slots=True)
class Loan:
    principal: float
    start_day: int
    due_day: int  # 兼容旧存档：当前不作为“固定期限”使用
    interest_mode: str  # 兼容旧存档字段（当前统一简单计息，不再区分模式）
    accrued_interest: float = 0.0
    late_fees: float = 0.0
    overdue_days: int = 0

    def debt_total(self) -> float:
        return round(self.principal + self.accrued_interest + self.late_fees, 2)


def estimated_assets(cash: float, cargo_lots: List[CargoLot]) -> float:
    """
    预估资产 = 现金 + 未售商品基础采购价（不叠加 λ，不考虑产地折扣）。
    """
    goods = 0.0
    for lot in cargo_lots:
        p = PRODUCTS[lot.product_id]
        goods += p.base_purchase_price * lot.quantity
    return round(cash + goods, 2)


def total_outstanding_principal(loans: List[Loan]) -> float:
    return round(sum(l.principal for l in loans), 2)


def borrow(loans: List[Loan], *, amount: float, day: int, interest_mode: str | None = None) -> Loan:
    if amount <= 0:
        raise ValueError("amount must be > 0")
    # 简化后：不再区分利息模式，统一按日利率 0.1% 计息，任意时间可部分/全部还清。
    mode = interest_mode or "simple"
    # 这里用 due_day=day 占位（兼容旧存档字段），不作为期限判定。
    loan = Loan(principal=round(amount, 2), start_day=day, due_day=day, interest_mode=mode)
    loans.append(loan)
    return loan


def process_one_day(loans: List[Loan], *, cash: float) -> tuple[float, List[str]]:
    """
    日结借贷（简化版）：
    - 日利率 0.1%
    - 每条借款每天按“本金 × 0.1%”计入 accrued_interest
    - 不再区分利息模式，也不再自动扣利息 / 计算逾期与滞纳金
    - 还款时一次性按“利息 + 本金”统一清算
    """
    msgs: List[str] = []
    if not loans:
        return cash, msgs

    for loan in loans:
        interest_today = round(loan.principal * LOAN_DAILY_INTEREST_RATE, 2)
        if interest_today <= 0:
            continue

        loan.accrued_interest = round(loan.accrued_interest + interest_today, 2)

    if loans:
        rate_pct = LOAN_DAILY_INTEREST_RATE * 100
        msgs.append(f"今日贷款按日利率 {rate_pct:.1f}% 计息。")

    return cash, msgs


def repay(loans: List[Loan], *, cash: float, amount: float | None) -> float:
    """
    还款：默认按“最早到期”优先，顺序扣：滞纳金 -> 利息 -> 本金。
    返回剩余现金。
    """
    if not loans:
        return cash
    start_cash = cash
    budget = start_cash if amount is None else min(start_cash, max(0.0, amount))
    pay = budget

    # 最早到期优先
    loans.sort(key=lambda l: (l.due_day, l.start_day))
    for loan in list(loans):
        if pay <= 0:
            break

        # late fees
        d = min(pay, loan.late_fees)
        loan.late_fees = round(loan.late_fees - d, 2)
        pay = round(pay - d, 2)

        # interest
        d = min(pay, loan.accrued_interest)
        loan.accrued_interest = round(loan.accrued_interest - d, 2)
        pay = round(pay - d, 2)

        # principal
        d = min(pay, loan.principal)
        loan.principal = round(loan.principal - d, 2)
        pay = round(pay - d, 2)

        if loan.principal <= 0 and loan.accrued_interest <= 0 and loan.late_fees <= 0:
            loans.remove(loan)

    spent = round(budget - pay, 2)
    return round(start_cash - spent, 2)


def force_collect_if_needed(
    loans: List[Loan],
    *,
    cash: float,
    cargo_lots: List[CargoLot],
) -> tuple[float, List[CargoLot], List[str]]:
    """
    逾期超过 7 天：强制扣除玩家自有资产抵扣。
    简化实现：直接没收全部现金与货物（按基础采购价折算）进行清偿。
    若仍不足，视为负债 -> 破产。
    """
    msgs: List[str] = []
    if not loans:
        return cash, cargo_lots, msgs

    if not any(l.overdue_days > 7 for l in loans):
        return cash, cargo_lots, msgs

    # 计算可扣资产
    seized_cash = cash
    seized_goods_value = 0.0
    for lot in cargo_lots:
        p = PRODUCTS[lot.product_id]
        seized_goods_value += p.base_purchase_price * lot.quantity

    seized_total = round(seized_cash + seized_goods_value, 2)
    msgs.append(f"银行强制扣款：没收现金 {seized_cash:.2f} + 货物折价 {seized_goods_value:.2f} = {seized_total:.2f}。")

    cash = 0.0
    cargo_lots = []

    # 用扣押资金偿还所有贷款（按最早到期）
    loans.sort(key=lambda l: (l.due_day, l.start_day))
    remaining = seized_total
    for loan in list(loans):
        if remaining <= 0:
            break
        debt = loan.debt_total()
        pay = min(remaining, debt)
        remaining = round(remaining - pay, 2)

        # 按顺序扣：late -> interest -> principal
        d = min(pay, loan.late_fees)
        loan.late_fees = round(loan.late_fees - d, 2)
        pay = round(pay - d, 2)

        d = min(pay, loan.accrued_interest)
        loan.accrued_interest = round(loan.accrued_interest - d, 2)
        pay = round(pay - d, 2)

        d = min(pay, loan.principal)
        loan.principal = round(loan.principal - d, 2)
        pay = round(pay - d, 2)

        if loan.principal <= 0 and loan.accrued_interest <= 0 and loan.late_fees <= 0:
            loans.remove(loan)

    if loans:
        raise Bankruptcy("强制扣款后仍欠款（产生负债）")

    # 剩余（若有）不会返还，按“扣走即扣走”处理，贴合强制处置
    return cash, cargo_lots, msgs

