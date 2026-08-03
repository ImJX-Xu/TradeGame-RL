"""基于核心命令协议的终端人类游玩界面。"""

from __future__ import annotations

import shlex
from collections.abc import Callable, Mapping, Sequence
from decimal import Decimal, InvalidOperation

from trade_game.core import (
    Borrow,
    Buy,
    BuyTruck,
    Command,
    CommandResult,
    GameMode,
    GameSession,
    NextDay,
    RepairTruck,
    Repay,
    Sell,
    Travel,
    TransportMode,
    available_credit,
    cargo_quantity,
    create_game_session,
    remote_sale_distance_multiplier,
    purchase_unit_price,
    sale_unit_price,
    total_debt,
)


Input = Callable[[str], str]
Output = Callable[[str], None]


class UserInputError(ValueError):
    """终端输入无法转换为有效命令时使用。"""


class TerminalGame:
    """将文本输入映射为核心命令的交互循环。"""

    def __init__(self, session: GameSession, *, input_fn: Input = input, output_fn: Output = print) -> None:
        self.session = session
        self._input = input_fn
        self._output = output_fn

    def run(self) -> GameSession:
        """运行交互循环，直至玩家退出或对局结束。"""

        self._output("TradeGame-RL")
        self._show_status()
        self._show_help()
        while self.session.state.outcome is None:
            try:
                raw = self._input(self._prompt())
            except (EOFError, KeyboardInterrupt):
                self._output("已退出游戏。")
                break
            if not self._handle_line(raw):
                break
        if self.session.state.outcome is not None:
            outcome = self.session.state.outcome
            self._output(f"游戏结束：{_outcome_label(outcome.reason.value)}，结算资产 {outcome.final_assets:,.2f}")
        return self.session

    def _handle_line(self, raw: str) -> bool:
        try:
            parts = shlex.split(raw)
        except ValueError as error:
            self._output(f"输入格式错误：{error}")
            return True
        if not parts:
            return True

        command_name = parts[0].casefold()
        if command_name in {"quit", "exit", "退出"}:
            self._output("已退出游戏。")
            return False
        if command_name in {"help", "帮助"}:
            self._show_help()
            return True
        if command_name in {"status", "状态"}:
            self._show_status()
            return True
        if command_name in {"market", "市场"}:
            self._show_market(parts[1:])
            return True
        if command_name in {"inventory", "库存"}:
            self._show_inventory()
            return True
        if command_name in {"cities", "城市"}:
            self._show_cities()
            return True
        if command_name in {"loans", "贷款"}:
            self._show_loans()
            return True

        try:
            command = _parse_game_command(parts)
        except UserInputError as error:
            self._output(f"输入错误：{error}")
            return True
        except ValueError as error:
            self._output(f"命令参数错误：{error}")
            return True
        self._show_result(self.session.dispatch(command))
        return True

    def _prompt(self) -> str:
        state = self.session.state
        return f"{state.player.location} 第 {state.day} 天> "

    def _show_result(self, result: CommandResult) -> None:
        if result.rejection is not None:
            self._output(f"操作未执行：{result.rejection.message}")
            return
        for event in result.events:
            self._output(_format_event(event.name, event.attributes))
        self._show_status(compact=True)

    def _show_status(self, *, compact: bool = False) -> None:
        state = self.session.state
        player = state.player
        debt = total_debt(state.loans)
        cargo = cargo_quantity(player.cargo_lots)
        if compact:
            self._output(
                f"第 {state.day} 天 | {player.location} | 现金 {player.cash:,.2f} | "
                f"货物 {cargo}/{player.truck_total_capacity} | 债务 {debt:,.2f}"
            )
            return
        self._output(f"日期：第 {state.day} 天    地点：{player.location}    模式：{state.mode.value}")
        self._output(
            f"现金：{player.cash:,.2f}    债务：{debt:,.2f}    可借："
            f"{available_credit(self.session.catalog, self.session.rules, state):,.2f}"
        )
        self._output(
            f"货车：{player.truck_count} 辆    容量：{cargo}/{player.truck_total_capacity}    "
            f"耐久：{player.truck_durability}%"
        )

    def _show_market(self, arguments: Sequence[str]) -> None:
        if len(arguments) > 1:
            self._output("输入错误：market 最多接受一个商品 ID")
            return
        product_ids = tuple(self.session.catalog.products)
        if arguments:
            product_id = arguments[0]
            if product_id not in self.session.catalog.products:
                self._output("商品不存在。")
                return
            product_ids = (product_id,)
        city_name = self.session.state.player.location
        self._output(f"{city_name} 市场")
        self._output("ID                 商品             采购价       售价")
        for product_id in product_ids:
            product = self.session.catalog.product(product_id)
            purchase = "-"
            if city_name in product.origins:
                purchase = f"{purchase_unit_price(self.session.catalog, self.session.rules, self.session.state, product_id, city_name):,.2f}"
            sale = sale_unit_price(
                self.session.catalog,
                self.session.rules,
                self.session.state,
                product_id,
                city_name,
                remote_distance_multiplier=remote_sale_distance_multiplier(
                    self.session.catalog, self.session.rules, product, city_name
                ),
            )
            self._output(f"{product_id:<18} {product.name:<12} {purchase:>10} {sale:>10,.2f}")

    def _show_inventory(self) -> None:
        lots = self.session.state.player.cargo_lots
        if not lots:
            self._output("库存为空。")
            return
        self._output("商品 ID              数量   产地       剩余保质期   已存天数")
        for lot in lots:
            shelf_life = "-" if lot.shelf_life_remaining_days is None else str(lot.shelf_life_remaining_days)
            self._output(
                f"{lot.product_id:<20} {lot.quantity:>4}   {lot.origin_city:<8} {shelf_life:>8}   {lot.age_days:>6}"
            )

    def _show_cities(self) -> None:
        self._output("城市       运输方式     银行  港口")
        for city in self.session.catalog.cities.values():
            modes = "+".join(sorted(mode.value for mode in city.modes))
            self._output(
                f"{city.name:<10} {modes:<12} {'有' if city.has_bank else '无':<4} "
                f"{'有' if city.has_port else '无'}"
            )

    def _show_loans(self) -> None:
        loans = self.session.state.loans
        if not loans:
            self._output("当前没有贷款。")
            return
        self._output("借入日    本金          已计利息")
        for loan in loans:
            self._output(f"{loan.start_day:>5}   {loan.principal:>12,.2f}  {loan.accrued_interest:>12,.2f}")

    def _show_help(self) -> None:
        self._output("status | market [商品 ID] | inventory | cities | loans")
        self._output("buy <商品 ID> <数量> | sell <商品 ID> <数量>")
        self._output("travel <城市> <land|sea> [fast] | repair | buy-truck <数量>")
        self._output("borrow <金额> | repay <金额> | next | quit")


def _parse_game_command(parts: Sequence[str]) -> Command:
    command_name = parts[0].casefold()
    arguments = parts[1:]
    if command_name in {"next", "next-day", "下一天"}:
        _expect_count(arguments, 0, command_name)
        return NextDay()
    if command_name in {"buy", "买入"}:
        _expect_count(arguments, 2, command_name)
        return Buy(arguments[0], _quantity(arguments[1]))
    if command_name in {"sell", "卖出"}:
        _expect_count(arguments, 2, command_name)
        return Sell(arguments[0], _quantity(arguments[1]))
    if command_name in {"travel", "旅行"}:
        if len(arguments) not in {2, 3}:
            raise UserInputError("travel 格式为：travel <城市> <land|sea> [fast]")
        try:
            mode = TransportMode(arguments[1].casefold())
        except ValueError as error:
            raise UserInputError("运输方式只能是 land 或 sea") from error
        fast = len(arguments) == 3 and arguments[2].casefold() == "fast"
        if len(arguments) == 3 and not fast:
            raise UserInputError("第三个参数只能是 fast")
        return Travel(arguments[0], mode, fast=fast)
    if command_name in {"repair", "维修"}:
        _expect_count(arguments, 0, command_name)
        return RepairTruck()
    if command_name in {"buy-truck", "购车"}:
        _expect_count(arguments, 1, command_name)
        return BuyTruck(_quantity(arguments[0]))
    if command_name in {"borrow", "借贷"}:
        _expect_count(arguments, 1, command_name)
        return Borrow(_amount(arguments[0]))
    if command_name in {"repay", "还款"}:
        _expect_count(arguments, 1, command_name)
        return Repay(_amount(arguments[0]))
    raise UserInputError("未知命令，输入 help 查看可用命令")


def run_terminal_game(*, seed: int | None = None, mode: GameMode = GameMode.FREE) -> GameSession:
    """创建默认会话并启动终端游玩。"""

    game = TerminalGame(create_game_session(seed=seed, mode=mode))
    return game.run()


def _expect_count(arguments: Sequence[str], expected: int, command_name: str) -> None:
    if len(arguments) != expected:
        raise UserInputError(f"{command_name} 需要 {expected} 个参数")


def _quantity(value: str) -> int:
    try:
        return int(value)
    except ValueError as error:
        raise UserInputError("数量必须是整数") from error


def _amount(value: str) -> Decimal:
    try:
        return Decimal(value)
    except InvalidOperation as error:
        raise UserInputError("金额必须是十进制数") from error


def _format_event(name: str, attributes: Mapping[str, object]) -> str:
    labels = {
        "day_advanced": "进入下一天",
        "days_settled": "日结完成",
        "goods_bought": "买入完成",
        "goods_sold": "卖出完成",
        "travel_completed": "旅行完成",
        "truck_repaired": "维修完成",
        "trucks_bought": "购车完成",
        "loan_borrowed": "借贷完成",
        "loan_repaid": "还款完成",
        "cargo_lost_in_transit": "运输货损",
        "game_finished": "游戏结束",
    }
    details = "，".join(f"{key}={value}" for key, value in attributes.items())
    return f"{labels.get(name, name)}：{details}" if details else labels.get(name, name)


def _outcome_label(reason: str) -> str:
    return {"bankruptcy": "破产", "time_limit": "达到挑战时间上限"}[reason]
