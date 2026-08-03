"""贷款额度、按日计息与 FIFO 还款规则。"""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal

from .catalog import Catalog
from .commands import Borrow, Repay
from .models import GameState, Loan
from .price_functions import money
from .results import CommandRejection, CommandResult, GameEvent, RejectionCode
from .rules import GameRules


def total_principal(loans: tuple[Loan, ...]) -> Decimal:
    """返回全部未还本金。"""

    return money(sum((loan.principal for loan in loans), start=Decimal("0")))


def total_debt(loans: tuple[Loan, ...]) -> Decimal:
    """返回本金和已计利息之和。"""

    return money(sum((loan.principal + loan.accrued_interest for loan in loans), start=Decimal("0")))


def assessed_assets(catalog: Catalog, state: GameState) -> Decimal:
    """按货物基础采购价评估可用于授信的流动资产。"""

    cargo_value = sum(
        (catalog.product(lot.product_id).base_purchase_price * lot.quantity for lot in state.player.cargo_lots),
        start=Decimal("0"),
    )
    return money(state.player.cash + cargo_value)


def available_credit(catalog: Catalog, rules: GameRules, state: GameState) -> Decimal:
    """以净资产决定最大债务，避免新借入现金重复扩大授信额度。"""

    net_assets = money(assessed_assets(catalog, state) - total_debt(state.loans))
    debt_limit = money(max(Decimal("0"), net_assets) * rules.finance.loan_asset_multiplier)
    return max(Decimal("0"), money(debt_limit - total_debt(state.loans)))


def borrow(catalog: Catalog, rules: GameRules, state: GameState, command: Borrow) -> CommandResult:
    """在银行城市新增一笔贷款，并立即增加现金。"""

    if not catalog.city(state.player.location).has_bank:
        return _reject(command, state, RejectionCode.NOT_ALLOWED, "当前城市没有银行")
    amount = money(command.amount)
    if amount <= 0:
        return _reject(command, state, RejectionCode.NOT_ALLOWED, "借贷金额至少为 0.01")
    credit = available_credit(catalog, rules, state)
    if amount > credit:
        return _reject(command, state, RejectionCode.NOT_ALLOWED, "借贷金额超过当前可用额度")

    loan = Loan(principal=amount, start_day=state.day)
    player = replace(state.player, cash=state.player.cash + amount)
    next_state = replace(state, player=player, loans=(*state.loans, loan))
    return CommandResult.succeed(
        command,
        next_state,
        GameEvent("loan_borrowed", {"amount": amount, "available_credit": credit - amount}),
    )


def repay(catalog: Catalog, rules: GameRules, state: GameState, command: Repay) -> CommandResult:
    """在银行城市按借入顺序偿还利息后再偿还本金。"""

    if not catalog.city(state.player.location).has_bank:
        return _reject(command, state, RejectionCode.NOT_ALLOWED, "当前城市没有银行")
    debt = total_debt(state.loans)
    if debt == 0:
        return _reject(command, state, RejectionCode.NOT_ALLOWED, "当前没有未偿还贷款")
    amount = money(command.amount)
    if amount <= 0:
        return _reject(command, state, RejectionCode.NOT_ALLOWED, "还款金额至少为 0.01")
    if amount > state.player.cash:
        return _reject(command, state, RejectionCode.INSUFFICIENT_CASH, "现金不足以完成还款")
    if amount > debt:
        return _reject(command, state, RejectionCode.NOT_ALLOWED, "还款金额超过未偿债务")

    remaining = amount
    updated_loans: list[Loan] = []
    for loan in sorted(state.loans, key=lambda item: item.start_day):
        interest_paid = min(remaining, loan.accrued_interest)
        remaining -= interest_paid
        principal_paid = min(remaining, loan.principal)
        remaining -= principal_paid
        next_interest = money(loan.accrued_interest - interest_paid)
        next_principal = money(loan.principal - principal_paid)
        if next_principal > 0 or next_interest > 0:
            updated_loans.append(
                Loan(principal=next_principal, start_day=loan.start_day, accrued_interest=next_interest)
            )

    player = replace(state.player, cash=state.player.cash - amount)
    next_state = replace(state, player=player, loans=tuple(updated_loans))
    return CommandResult.succeed(
        command,
        next_state,
        GameEvent("loan_repaid", {"amount": amount, "remaining_debt": total_debt(next_state.loans)}),
    )


def accrue_daily_interest(loans: tuple[Loan, ...], daily_rate: Decimal) -> tuple[tuple[Loan, ...], Decimal]:
    """为每笔未还本金计入一天的简单利息。"""

    updated: list[Loan] = []
    accrued = Decimal("0")
    for loan in loans:
        interest = money(loan.principal * daily_rate)
        accrued += interest
        updated.append(replace(loan, accrued_interest=loan.accrued_interest + interest))
    return tuple(updated), money(accrued)


def _reject(command: Borrow | Repay, state: GameState, code: RejectionCode, message: str) -> CommandResult:
    return CommandResult.reject(command, state, CommandRejection(code, message))
