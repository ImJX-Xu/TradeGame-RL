"""可完整游玩的图形人类界面，只通过核心命令改变游戏状态。"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, replace
from decimal import Decimal

import arcade

from trade_game.core import (
    Borrow,
    Buy,
    BuyTruck,
    Command,
    GameEvent,
    GameMode,
    GameSession,
    MarketEventKind,
    NextDay,
    RepairTruck,
    Repay,
    RouteNotFound,
    Sell,
    TransportMode,
    Travel,
    available_credit,
    cargo_quantity,
    create_game_session,
    free_capacity,
    market_bulletins,
    money,
    purchase_unit_price,
    quote_sale,
    reference_sale_origin,
    remote_sale_distance_premium,
    sale_unit_price,
    shortest_distance,
    trade_total,
    total_debt,
)

from .layout import MainLayout, main_layout
from .map_view import ReachableRoute, draw_map
from .theme import (
    ACCENT_TEAL,
    APP_BACKGROUND,
    CURRENT_CITY,
    DARK_SHADOW,
    MUTED,
    NEGATIVE,
    PANEL_FACE,
    PANEL_INSET,
    POSITIVE,
    Rect,
    TEXT_DARK,
    TEXT_LIGHT,
    TITLE_BLUE,
    draw_command_button,
    draw_navigation_item,
    draw_raised_panel,
    draw_status_value,
    draw_sunken_panel,
    draw_title_bar,
    draw_toggle,
)


WINDOW_MIN_WIDTH = 1100
WINDOW_MIN_HEIGHT = 720
WINDOW_TITLE = "风物千程"
TAB_NAMES = ("采购", "出售", "行情", "路线", "车辆", "融资", "库存", "路书")
FINANCE_STEP = Decimal("100")
MARKET_PAGE_SIZE = 8


@dataclass(frozen=True, slots=True)
class TabHitbox:
    """一个可以切换的右侧工作区页签。"""

    name: str
    rect: Rect


@dataclass(frozen=True, slots=True)
class ActionHitbox:
    """一个由表现层持有、执行时转换为核心命令的可点击控件。"""

    action: str
    rect: Rect
    value: str | int | TransportMode | tuple[str, str] | None = None
    enabled: bool = True


@dataclass(frozen=True, slots=True)
class TradeOrder:
    """一张尚未提交核心会话的图形交易订单。"""

    mode: str
    product_id: str
    quantity: int = 1


class TradeGameWindow(arcade.Window):
    """经营主屏窗口；局部 UI 状态与核心游戏状态保持严格分离。"""

    def __init__(self, session: GameSession, *, seed: int | None = None) -> None:
        super().__init__(
            WINDOW_MIN_WIDTH,
            WINDOW_MIN_HEIGHT,
            WINDOW_TITLE,
            resizable=True,
            antialiasing=True,
            center_window=True,
        )
        self.session = session
        self._seed = seed
        self._mode = session.state.mode
        self.active_tab = "采购"
        self.hovered_tab: str | None = None
        self.hovered_action: tuple[str, str | int | TransportMode | tuple[str, str] | None] | None = None
        self.selected_city: str | None = session.state.player.location
        self.market_city = session.state.player.location
        self.market_page = 0
        self.truck_quantity = 1
        self.finance_amount = FINANCE_STEP
        self.fast_travel = False
        self.trade_order: TradeOrder | None = None
        self._city_positions: Mapping[str, tuple[float, float]] = {}
        self._tab_hitboxes: list[TabHitbox] = []
        self._action_hitboxes: list[ActionHitbox] = []
        self._price_cache: dict[tuple[str, str], tuple[Decimal | None, Decimal]] = {}
        self._route_cache: dict[str, dict[str, tuple[ReachableRoute, ...]]] = {}
        self.event_log: deque[str] = deque(("郑州货站今日开张。",), maxlen=80)
        self.notice = "郑州货站正在等候调度。"
        self.background_color = APP_BACKGROUND
        self.set_minimum_size(WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT)

    def on_draw(self) -> None:
        """根据当前核心快照和 UI 焦点绘制整张经营主屏。"""

        self.clear()
        self._tab_hitboxes.clear()
        self._action_hitboxes.clear()
        layout = main_layout(self.width, self.height, route_view=self.active_tab == "路线")
        self._draw_title(layout)
        self._draw_status(layout)
        self._draw_market_notice(layout.market_notice)
        if self.active_tab == "路线":
            self._draw_map(layout)
        else:
            self._city_positions = {}
        self._draw_workspace(layout)
        self._draw_footer(layout)
        if self.trade_order is not None and self.session.state.outcome is None:
            self._draw_trade_order_modal()
        if self.session.state.outcome is not None:
            self._draw_outcome_modal()

    def on_mouse_motion(self, x: float, y: float, _dx: float, _dy: float) -> None:
        """更新页签、按钮和复选开关的悬停反馈。"""

        action = next((item for item in reversed(self._action_hitboxes) if item.rect.contains(x, y)), None)
        self.hovered_action = (action.action, action.value) if action is not None else None
        tab = next((item.name for item in self._tab_hitboxes if item.rect.contains(x, y)), None)
        self.hovered_tab = tab

    def on_mouse_press(self, x: float, y: float, _button: int, _modifiers: int) -> None:
        """消费图形控件事件，并将有效操作映射为核心命令。"""

        action = next((item for item in reversed(self._action_hitboxes) if item.rect.contains(x, y)), None)
        if action is not None:
            if action.enabled:
                self._handle_action(action)
            return
        if self.trade_order is not None:
            return
        if self.session.state.outcome is not None:
            return
        tab = next((item for item in self._tab_hitboxes if item.rect.contains(x, y)), None)
        if tab is not None:
            self.active_tab = tab.name
            return
        nearest = min(
            self._city_positions.items(),
            key=lambda item: (item[1][0] - x) ** 2 + (item[1][1] - y) ** 2,
            default=None,
        )
        if nearest is not None and (nearest[1][0] - x) ** 2 + (nearest[1][1] - y) ** 2 <= 24**2:
            reachable_routes = self._reachable_routes(self.session.state.player.location)
            if nearest[0] != self.session.state.player.location and nearest[0] not in reachable_routes:
                self.notice = f"当前无法从本站前往 {nearest[0]}。"
                self.event_log.append(self.notice)
                return
            self.selected_city = nearest[0]
            self.active_tab = "路线"
            self.fast_travel = False
            self.event_log.append(f"调度焦点切换至 {nearest[0]}。")
            self.notice = f"已选择 {nearest[0]} 作为调度目标。"

    def on_key_press(self, symbol: int, _modifiers: int) -> None:
        """提供页签切换和返回当前位置的基础键盘导航。"""

        if self.session.state.outcome is not None:
            return
        if self.trade_order is not None:
            if symbol == arcade.key.ESCAPE:
                self.trade_order = None
            return
        if arcade.key.KEY_1 <= symbol < arcade.key.KEY_1 + len(TAB_NAMES):
            self.active_tab = TAB_NAMES[symbol - arcade.key.KEY_1]
        elif symbol == arcade.key.HOME:
            self.selected_city = self.session.state.player.location
            self.fast_travel = False

    def _draw_title(self, layout: MainLayout) -> None:
        draw_raised_panel(layout.title, fill=PANEL_FACE)
        draw_title_bar(layout.title.inset(3), "风物千程")
        mode = "挑战模式" if self.session.state.mode is GameMode.CHALLENGE else "自由模式"
        arcade.draw_text(
            mode,
            layout.title.right - 14,
            layout.title.center_y,
            TEXT_LIGHT,
            13,
            font_name="Microsoft YaHei UI",
            anchor_x="right",
            anchor_y="center",
        )

    def _draw_status(self, layout: MainLayout) -> None:
        draw_raised_panel(layout.status)
        content = layout.status.inset(8)
        state = self.session.state
        player = state.player
        used_capacity = cargo_quantity(player.cargo_lots)
        metrics = (
            ("日期", f"第 {state.day} 天", TEXT_DARK),
            ("当前城市", player.location, CURRENT_CITY),
            ("现金", f"{player.cash:,.2f}", POSITIVE if player.cash >= 0 else NEGATIVE),
            ("债务", f"{total_debt(state.loans):,.2f}", NEGATIVE if state.loans else TEXT_DARK),
            ("货物容量", f"{used_capacity}/{player.truck_total_capacity}", TEXT_DARK),
            ("车队耐久", f"{player.truck_durability}%", CURRENT_CITY if player.truck_durability < 50 else ACCENT_TEAL),
        )
        cell_width = content.width / len(metrics)
        for index, (label, value, color) in enumerate(metrics):
            cell = Rect(
                content.left + index * cell_width,
                content.bottom,
                content.left + (index + 1) * cell_width,
                content.top,
            )
            draw_status_value(cell, label, value, color=color)
            if index:
                arcade.draw_line(cell.left, cell.bottom + 5, cell.left, cell.top - 5, DARK_SHADOW, 1)

    def _draw_market_notice(self, bounds: Rect) -> None:
        """在所有工作页签上方显示当前决策关联城市的简短重要行情。"""

        state = self.session.state
        bulletins = market_bulletins(self.session.catalog, self.session.rules, state)
        fill = (212, 226, 222) if bulletins else (218, 218, 214)
        draw_raised_panel(bounds, fill=fill)
        arcade.draw_lrbt_rectangle_filled(
            bounds.left + 5,
            bounds.left + 9,
            bounds.bottom + 6,
            bounds.top - 6,
            ACCENT_TEAL if bulletins else MUTED,
        )
        arcade.draw_text(
            "市场电报",
            bounds.left + 20,
            bounds.center_y,
            TEXT_DARK,
            13,
            font_name="Microsoft YaHei UI",
            bold=True,
            anchor_y="center",
        )
        arcade.draw_line(
            bounds.left + 90,
            bounds.bottom + 10,
            bounds.left + 90,
            bounds.top - 10,
            MUTED,
            1,
        )
        if bulletins:
            bulletin = bulletins[0]
            message_text = _format_market_message(
                "、".join(bulletin.cities),
                self.session.catalog.product(bulletin.product_id).name,
                bulletin.kind,
                bulletin.remaining_days,
            )
            text_color = TEXT_DARK
        else:
            message_text = "全国市场：近期供需平稳，留意行情变化。"
            text_color = MUTED
        arcade.draw_text(
            _short_text(message_text, max(24, int((bounds.width - 126) / 12))),
            bounds.left + 106,
            bounds.center_y,
            text_color,
            12,
            font_name="Microsoft YaHei UI",
            anchor_y="center",
        )

    def _draw_map(self, layout: MainLayout) -> None:
        draw_raised_panel(layout.map)
        origin = self.session.state.player.location
        self._city_positions = draw_map(
            self.session.catalog,
            self.session.state,
            layout.map.inset(5),
            selected_city=self.selected_city,
            reachable_routes=self._reachable_routes(origin),
        )

    def _draw_workspace(self, layout: MainLayout) -> None:
        """绘制导航与当前业务台面，交易和路线不再争抢同一块狭窄侧栏。"""

        draw_raised_panel(layout.navigation)
        navigation_content = layout.navigation.inset(6)
        row_height = 42
        row_gap = 7
        for index, name in enumerate(TAB_NAMES):
            rect = Rect(
                navigation_content.left,
                navigation_content.top - (index + 1) * row_height - index * row_gap,
                navigation_content.right,
                navigation_content.top - index * (row_height + row_gap),
            )
            self._tab_hitboxes.append(TabHitbox(name, rect))
            draw_navigation_item(
                rect,
                name,
                active=name == self.active_tab,
                hovered=name == self.hovered_tab,
                shortcut=index + 1,
            )

        arcade.draw_text(
            "调度台",
            navigation_content.left + 8,
            navigation_content.bottom + 12,
            MUTED,
            11,
            font_name="Microsoft YaHei UI",
            anchor_y="bottom",
        )
        draw_sunken_panel(layout.detail)
        content = layout.detail.inset(16)
        if self.active_tab == "采购":
            self._draw_market(content, mode="buy")
        elif self.active_tab == "出售":
            self._draw_market(content, mode="sell")
        elif self.active_tab == "行情":
            self._draw_market_board(content)
        elif self.active_tab == "路线":
            self._draw_routes(content)
        elif self.active_tab == "车辆":
            self._draw_vehicles(content)
        elif self.active_tab == "融资":
            self._draw_finance(content)
        elif self.active_tab == "库存":
            self._draw_inventory(content)
        else:
            self._draw_journal(content)

    def _draw_market(self, bounds: Rect, *, mode: str) -> None:
        """绘制独立的采购台或出货台，只展示当前可提交的订单。"""

        state = self.session.state
        city_name = state.player.location
        product_ids = self._available_trade_products(mode)
        is_buy = mode == "buy"
        title = f"{city_name} 采购台" if is_buy else f"{city_name} 出货台"
        availability = (
            f"现金 {state.player.cash:,.2f}    可用运力 {free_capacity(state.player.cargo_lots, state.player.truck_total_capacity)}"
            if is_buy
            else f"车载货物 {cargo_quantity(state.player.cargo_lots)}    总运力 {state.player.truck_total_capacity}"
        )
        arcade.draw_text(title, bounds.left, bounds.top, TEXT_DARK, 20, font_name="Microsoft YaHei UI", bold=True, anchor_y="top")
        arcade.draw_text(
            availability,
            bounds.right,
            bounds.top - 4,
            MUTED,
            12,
            font_name="Microsoft YaHei UI",
            anchor_x="right",
            anchor_y="top",
        )
        header = Rect(bounds.left, bounds.top - 72, bounds.right, bounds.top - 38)
        arcade.draw_lrbt_rectangle_filled(header.left, header.right, header.bottom, header.top, (211, 219, 216))
        quantity_label = "最多可购" if is_buy else "车载数量"
        self._draw_market_columns(header, quantity_label)
        row_top = header.bottom - 6
        for index, product_id in enumerate(product_ids):
            product = self.session.catalog.product(product_id)
            unit_price, quantity_limit = self._order_quote(mode, product_id)
            price_history = self._price_history(mode, product_id)
            row = Rect(bounds.left, row_top - (index + 1) * 58, bounds.right, row_top - index * 58)
            self._draw_trade_row(
                row,
                product_id,
                product.name,
                _category_label(product.category.value),
                unit_price,
                price_history,
                quantity_limit,
                mode,
            )

        if not product_ids:
            empty_title = "没有可采购商品" if is_buy else "没有可出售货物"
            empty_detail = "现金或车辆运力不足。" if is_buy else "当前车载数量为 0。"
            empty_y = bounds.top - 128
            arcade.draw_text(empty_title, bounds.center_x, empty_y, MUTED, 17, font_name="Microsoft YaHei UI", bold=True, anchor_x="center", anchor_y="top")
            arcade.draw_text(empty_detail, bounds.center_x, empty_y - 30, MUTED, 12, font_name="Microsoft YaHei UI", anchor_x="center", anchor_y="top")

    def _draw_market_columns(self, rect: Rect, quantity_label: str) -> None:
        """按工作区比例绘制表头，列位置与订单行共用同一套计算。"""

        columns = (
            ("商品", rect.left + 16, "left"),
            ("类别", rect.left + rect.width * 0.25, "left"),
            ("当前进价" if quantity_label == "最大可购" else "当前均价", rect.left + rect.width * 0.45, "right"),
            ("昨日变动", rect.left + rect.width * 0.58, "right"),
            ("7 日均价", rect.left + rect.width * 0.70, "right"),
            (quantity_label, rect.left + rect.width * 0.80, "right"),
            ("下单", rect.right - 16, "right"),
        )
        for label, x, anchor_x in columns:
            arcade.draw_text(
                label,
                x,
                rect.center_y,
                MUTED,
                12,
                font_name="Microsoft YaHei UI",
                bold=True,
                anchor_x=anchor_x,
                anchor_y="center",
            )

    def _draw_trade_row(
        self,
        rect: Rect,
        product_id: str,
        product_name: str,
        category: str,
        unit_price: Decimal,
        price_history: tuple[Decimal, ...],
        quantity_limit: int,
        mode: str,
    ) -> None:
        action = "open-order"
        label = "采购" if mode == "buy" else "出售"
        color = ACCENT_TEAL if mode == "buy" else CURRENT_CITY
        arcade.draw_lrbt_rectangle_filled(rect.left, rect.right, rect.bottom, rect.top, PANEL_INSET)
        arcade.draw_line(rect.left, rect.bottom, rect.right, rect.bottom, MUTED, 1)
        arcade.draw_text(product_name, rect.left + 16, rect.center_y, TEXT_DARK, 15, font_name="Microsoft YaHei UI", bold=True, anchor_y="center")
        change_text, change_color = _price_change(price_history, mode)
        average_price = money(sum(price_history) / len(price_history))
        arcade.draw_text(category, rect.left + rect.width * 0.25, rect.center_y, MUTED, 12, font_name="Microsoft YaHei UI", anchor_y="center")
        arcade.draw_text(
            f"{unit_price:,.2f}",
            rect.left + rect.width * 0.45,
            rect.center_y,
            color,
            14,
            font_name="Microsoft YaHei UI",
            anchor_x="right",
            anchor_y="center",
        )
        arcade.draw_text(
            change_text,
            rect.left + rect.width * 0.58,
            rect.center_y,
            change_color,
            12,
            font_name="Microsoft YaHei UI",
            anchor_x="right",
            anchor_y="center",
        )
        arcade.draw_text(
            f"{average_price:,.2f}",
            rect.left + rect.width * 0.70,
            rect.center_y,
            TEXT_DARK,
            12,
            font_name="Microsoft YaHei UI",
            anchor_x="right",
            anchor_y="center",
        )
        arcade.draw_text(
            str(quantity_limit),
            rect.left + rect.width * 0.80,
            rect.center_y,
            TEXT_DARK,
            14,
            font_name="Microsoft YaHei UI",
            anchor_x="right",
            anchor_y="center",
        )
        self._draw_action_button(
            Rect(rect.right - 84, rect.bottom + 10, rect.right - 12, rect.top - 10),
            label,
            action,
            (mode, product_id),
            emphasis=mode == "buy",
        )

    def _available_trade_products(self, mode: str) -> tuple[str, ...]:
        """返回当前模式下能实际提交订单的商品，而不是全量目录。"""

        city_name = self.session.state.player.location
        if mode == "buy":
            available: list[str] = []
            for product in self.session.catalog.products.values():
                if city_name not in product.origins:
                    continue
                _unit_price, limit = self._order_quote("buy", product.id)
                if limit > 0:
                    available.append(product.id)
            return tuple(available)
        return tuple(
            product_id
            for product_id in self.session.catalog.products
            if cargo_quantity(self.session.state.player.cargo_lots, product_id) > 0
        )

    def _order_quote(self, mode: str, product_id: str) -> tuple[Decimal, int]:
        """为订单窗口提供当前单价和一次可执行的最大数量。"""

        state = self.session.state
        product = self.session.catalog.product(product_id)
        if mode == "buy":
            purchase, _sale = self._market_prices(product_id, state.player.location)
            assert purchase is not None
            capacity = free_capacity(state.player.cargo_lots, state.player.truck_total_capacity)
            affordable = int(state.player.cash / purchase)
            return purchase, min(capacity, affordable)
        quantity = cargo_quantity(state.player.cargo_lots, product.id)
        if quantity == 0:
            return Decimal("0"), 0
        quote = quote_sale(
            self.session.catalog,
            self.session.rules,
            state,
            product.id,
            quantity,
        )
        return quote.average_unit_price, quantity

    def _price_history(
        self, mode: str, product_id: str, *, city_name: str | None = None
    ) -> tuple[Decimal, ...]:
        """将核心保存的价格扰动还原为玩家在当前城市可见的价格序列。"""

        state = self.session.state
        city_name = city_name or state.player.location
        product = self.session.catalog.product(product_id)
        factor = product.base_purchase_price
        if mode == "sell":
            origin_city = reference_sale_origin(self.session.catalog, product, city_name)
            distance_premium = remote_sale_distance_premium(
                self.session.catalog,
                self.session.rules,
                origin_city,
                city_name,
            )
        else:
            origin_city = city_name
            distance_premium = Decimal("0")
        prices: list[Decimal] = []
        for price_adjustment in state.market.price_adjustment_history[(city_name, product_id)]:
            price = factor * (Decimal("1") + price_adjustment)
            if mode == "sell" and origin_city != city_name:
                price *= Decimal("1") + product.profit_margin_rate
                price += distance_premium
                if self.session.catalog.city(city_name).is_high_consumption:
                    price *= self.session.rules.pricing.high_consumption_multiplier
            prices.append(money(price))
        return tuple(prices)

    def _draw_trade_order_modal(self) -> None:
        """以订单单据集中确认商品、数量、金额和最终动作。"""

        order = self.trade_order
        assert order is not None
        unit_price, maximum = self._order_quote(order.mode, order.product_id)
        quantity = min(order.quantity, maximum) if maximum else 1
        product = self.session.catalog.product(order.product_id)
        is_buy = order.mode == "buy"
        title = "采购订单" if is_buy else "出售订单"
        action_label = "采购" if is_buy else "出售"
        if is_buy:
            total = trade_total(unit_price, quantity)
        else:
            sale_quote = quote_sale(
                self.session.catalog,
                self.session.rules,
                self.session.state,
                order.product_id,
                quantity,
            )
            unit_price = sale_quote.average_unit_price
            total = sale_quote.total
        price_history = self._price_history(order.mode, order.product_id)
        change_text, change_color = _price_change(price_history, order.mode)
        average_price = money(sum(price_history) / len(price_history))
        player = self.session.state.player
        after_value = player.cash - total if is_buy else player.cash + total
        capacity_text = (
            f"交易后运力 {cargo_quantity(player.cargo_lots) + quantity}/{player.truck_total_capacity}"
            if is_buy
            else f"交易后持有 {maximum - quantity}"
        )

        arcade.draw_lrbt_rectangle_filled(0, self.width, 0, self.height, (0, 0, 0, 120))
        rect = Rect(self.width / 2 - 290, self.height / 2 - 200, self.width / 2 + 290, self.height / 2 + 200)
        draw_raised_panel(rect)
        draw_title_bar(Rect(rect.left + 3, rect.top - 31, rect.right - 3, rect.top - 3), title)
        arcade.draw_text(product.name, rect.left + 24, rect.top - 54, TEXT_DARK, 19, font_name="Microsoft YaHei UI", bold=True, anchor_y="top")
        rows = (
            ("当前进价" if is_buy else "本单均价", f"{unit_price:,.2f}", TEXT_DARK),
            ("昨日变动", change_text, change_color),
            ("7 日均价", f"{average_price:,.2f}", TEXT_DARK),
            ("最多可购" if is_buy else "可售数量", str(maximum), TEXT_DARK),
            ("订单合计", f"{total:,.2f}", ACCENT_TEAL if is_buy else CURRENT_CITY),
            ("交易后现金", f"{after_value:,.2f}", POSITIVE if after_value >= 0 else NEGATIVE),
        )
        y = rect.top - 92
        for label, value, value_color in rows:
            arcade.draw_text(label, rect.left + 20, y, MUTED, 12, font_name="Microsoft YaHei UI", anchor_y="top")
            arcade.draw_text(value, rect.right - 24, y, value_color, 14, font_name="Microsoft YaHei UI", bold=label == "订单合计", anchor_x="right", anchor_y="top")
            y -= 24
        arcade.draw_text(capacity_text, rect.left + 20, rect.bottom + 128, MUTED, 12, font_name="Microsoft YaHei UI", anchor_y="top")
        arcade.draw_line(rect.left + 20, rect.bottom + 106, rect.right - 20, rect.bottom + 106, MUTED, 1)
        button_y = rect.bottom + 64
        left = rect.left + 24
        self._draw_action_button(Rect(left, button_y, left + 54, button_y + 30), "-10", "order-step", -10, enabled=quantity > 1)
        self._draw_action_button(Rect(left + 60, button_y, left + 96, button_y + 30), "-", "order-step", -1, enabled=quantity > 1)
        quantity_rect = Rect(left + 102, button_y, left + 188, button_y + 30)
        draw_sunken_panel(quantity_rect)
        arcade.draw_text(str(quantity), quantity_rect.center_x, quantity_rect.center_y, TEXT_DARK, 15, font_name="Microsoft YaHei UI", bold=True, anchor_x="center", anchor_y="center")
        self._draw_action_button(Rect(left + 194, button_y, left + 230, button_y + 30), "+", "order-step", 1, enabled=maximum > quantity)
        self._draw_action_button(Rect(left + 236, button_y, left + 290, button_y + 30), "+10", "order-step", 10, enabled=maximum > quantity)
        self._draw_action_button(Rect(left + 296, button_y, left + 362, button_y + 30), "最大", "order-max", enabled=maximum > quantity)
        self._draw_action_button(Rect(rect.left + 24, rect.bottom + 20, rect.left + 166, rect.bottom + 48), "取消", "close-order")
        self._draw_action_button(
            Rect(rect.right - 166, rect.bottom + 20, rect.right - 24, rect.bottom + 48),
            action_label,
            "confirm-order",
            enabled=maximum > 0,
            emphasis=True,
        )

    def _draw_market_board(self, bounds: Rect) -> None:
        """显示规划行程时需要的城市商品行情，不暴露内部状态和调试数据。"""

        catalog = self.session.catalog
        city_names = tuple(catalog.cities)
        page_size = MARKET_PAGE_SIZE
        products = tuple(catalog.products.values())
        page_count = (len(products) + page_size - 1) // page_size
        self.market_page = min(self.market_page, max(0, page_count - 1))

        arcade.draw_text("行商行情", bounds.left, bounds.top, TEXT_DARK, 20, font_name="Microsoft YaHei UI", bold=True, anchor_y="top")
        arcade.draw_text(
            f"查看 {self.market_city} 的市价",
            bounds.right,
            bounds.top - 4,
            MUTED,
            12,
            font_name="Microsoft YaHei UI",
            anchor_x="right",
            anchor_y="top",
        )

        city_width = bounds.width / 7
        city_top = bounds.top - 42
        for index, city_name in enumerate(city_names):
            row = index // 7
            column = index % 7
            left = bounds.left + column * city_width + 2
            right = bounds.left + (column + 1) * city_width - 2
            top = city_top - row * 30
            self._draw_action_button(
                Rect(left, top - 26, right, top),
                city_name,
                "select-market-city",
                city_name,
                emphasis=city_name == self.market_city,
            )

        header = Rect(bounds.left, bounds.top - 148, bounds.right, bounds.top - 116)
        arcade.draw_lrbt_rectangle_filled(header.left, header.right, header.bottom, header.top, (211, 219, 216))
        columns = (
            ("商品", header.left + header.width * 0.02, "left"),
            ("货源", header.left + header.width * 0.22, "left"),
            ("进货价", header.left + header.width * 0.48, "right"),
            ("出货参考", header.left + header.width * 0.61, "right"),
            ("7 日低 / 高", header.left + header.width * 0.83, "right"),
            ("涨跌", header.right - 16, "right"),
        )
        for label, x, anchor_x in columns:
            arcade.draw_text(
                label,
                x,
                header.center_y,
                MUTED,
                11,
                font_name="Microsoft YaHei UI",
                bold=True,
                anchor_x=anchor_x,
                anchor_y="center",
            )

        page_start = self.market_page * page_size
        row_top = header.bottom - 4
        for product in products[page_start : page_start + page_size]:
            row = Rect(bounds.left, row_top - 28, bounds.right, row_top)
            purchase, sale = self._market_prices(product.id, self.market_city)
            history = self._price_history("sell", product.id, city_name=self.market_city)
            change_text, change_color = _price_change(history, "sell")
            price_range = f"{min(history):,.2f} / {max(history):,.2f}"
            arcade.draw_lrbt_rectangle_filled(row.left, row.right, row.bottom, row.top, PANEL_INSET)
            arcade.draw_line(row.left, row.bottom, row.right, row.bottom, MUTED, 1)
            arcade.draw_text(product.name, row.left + row.width * 0.02, row.center_y, TEXT_DARK, 12, font_name="Microsoft YaHei UI", bold=True, anchor_y="center")
            arcade.draw_text("、".join(product.origins), row.left + row.width * 0.22, row.center_y, MUTED, 11, font_name="Microsoft YaHei UI", anchor_y="center")
            arcade.draw_text(f"{purchase:,.2f}" if purchase is not None else "-", row.left + row.width * 0.48, row.center_y, ACCENT_TEAL if purchase is not None else MUTED, 12, font_name="Microsoft YaHei UI", anchor_x="right", anchor_y="center")
            arcade.draw_text(f"{sale:,.2f}", row.left + row.width * 0.61, row.center_y, TEXT_DARK, 12, font_name="Microsoft YaHei UI", anchor_x="right", anchor_y="center")
            arcade.draw_text(price_range, row.left + row.width * 0.83, row.center_y, TEXT_DARK, 11, font_name="Microsoft YaHei UI", anchor_x="right", anchor_y="center")
            arcade.draw_text(change_text, row.right - 16, row.center_y, change_color, 11, font_name="Microsoft YaHei UI", anchor_x="right", anchor_y="center")
            row_top = row.bottom

        caption = "进货价只在货源地可用；出货参考价按最近货源估算，实际出售按货物来源结算。"
        arcade.draw_text(caption, bounds.left, bounds.bottom + 26, MUTED, 11, font_name="Microsoft YaHei UI", anchor_y="center")
        if page_count > 1:
            button_y = bounds.bottom + 10
            self._draw_action_button(Rect(bounds.right - 116, button_y, bounds.right - 82, button_y + 26), "<", "market-page", -1, enabled=self.market_page > 0)
            arcade.draw_text(f"{self.market_page + 1} / {page_count}", bounds.right - 60, button_y + 13, MUTED, 11, font_name="Microsoft YaHei UI", anchor_x="center", anchor_y="center")
            self._draw_action_button(Rect(bounds.right - 38, button_y, bounds.right - 4, button_y + 26), ">", "market-page", 1, enabled=self.market_page < page_count - 1)

    def _draw_routes(self, bounds: Rect) -> None:
        state = self.session.state
        origin = state.player.location
        destination = self.selected_city or origin
        grouped = self._reachable_routes(origin)
        if destination == origin:
            arcade.draw_text(f"{origin} 发运路线", bounds.left, bounds.top, TEXT_DARK, 16, font_name="Microsoft YaHei UI", bold=True, anchor_y="top")
            arcade.draw_text(f"可达城市  {len(grouped)}", bounds.left, bounds.top - 31, MUTED, 11, font_name="Microsoft YaHei UI", anchor_y="top")
            y = bounds.top - 55
            row_height = min(28, max(20, int((bounds.height - 58) / max(1, len(grouped)))))
            for city_name, routes in grouped.items():
                rect = Rect(bounds.left, y - row_height + 2, bounds.right, y)
                self._draw_destination_row(rect, city_name, routes)
                y -= row_height
            return

        routes = grouped.get(destination, ())
        arcade.draw_text("运输计划", bounds.left, bounds.top, TEXT_DARK, 16, font_name="Microsoft YaHei UI", bold=True, anchor_y="top")
        arcade.draw_text(f"{origin}  ->  {destination}", bounds.left, bounds.top - 31, MUTED, 12, font_name="Microsoft YaHei UI", anchor_y="top")
        y = bounds.top - 59
        for route in routes:
            mode_name = "陆运" if route.mode is TransportMode.LAND else "海运"
            color = CURRENT_CITY if route.mode is TransportMode.LAND else ACCENT_TEAL
            arcade.draw_text(mode_name, bounds.left, y, color, 14, font_name="Microsoft YaHei UI", bold=True, anchor_y="top")
            arcade.draw_text(f"{route.distance_km} km", bounds.left + 68, y, TEXT_DARK, 13, font_name="Microsoft YaHei UI", anchor_y="top")
            self._draw_action_button(
                Rect(bounds.right - 74, y - 5, bounds.right, y + 22),
                "出发",
                "travel",
                route.mode,
                emphasis=True,
            )
            y -= 39
        toggle_rect = Rect(bounds.left, y - 31, bounds.left + 100, y - 3)
        self._action_hitboxes.append(ActionHitbox("toggle-fast", toggle_rect))
        draw_toggle(
            toggle_rect,
            "加急运输",
            checked=self.fast_travel,
            hovered=self._is_hovered("toggle-fast", None),
            enabled=True,
        )

    def _draw_destination_row(self, rect: Rect, city_name: str, routes: tuple[ReachableRoute, ...]) -> None:
        selected = city_name == self.selected_city
        arcade.draw_lrbt_rectangle_filled(rect.left, rect.right, rect.bottom, rect.top, (212, 226, 222) if selected else PANEL_INSET)
        if selected:
            arcade.draw_lrbt_rectangle_outline(rect.left, rect.right, rect.bottom, rect.top, ACCENT_TEAL, 2)
        self._action_hitboxes.append(ActionHitbox("select-city", rect, city_name))
        mode_names = "/".join("陆" if route.mode is TransportMode.LAND else "海" for route in routes)
        distances = "/".join(f"{route.distance_km:,}" for route in routes)
        region = self.session.catalog.city(city_name).region
        arcade.draw_text(city_name, rect.left + 7, rect.center_y, TEXT_DARK, 11, font_name="Microsoft YaHei UI", bold=selected, anchor_y="center")
        arcade.draw_text(region, rect.left + 61, rect.center_y, MUTED, 10, font_name="Microsoft YaHei UI", anchor_y="center")
        arcade.draw_text(mode_names, rect.left + 108, rect.center_y, MUTED, 10, font_name="Microsoft YaHei UI", anchor_y="center")
        arcade.draw_text(f"{distances} km", rect.right - 7, rect.center_y, TEXT_DARK, 10, font_name="Microsoft YaHei UI", anchor_x="right", anchor_y="center")

    def _draw_vehicles(self, bounds: Rect) -> None:
        player = self.session.state.player
        rules = self.session.rules.vehicles
        arcade.draw_text("车队管理", bounds.left, bounds.top, TEXT_DARK, 20, font_name="Microsoft YaHei UI", bold=True, anchor_y="top")
        arcade.draw_text("运输能力与维护", bounds.right, bounds.top - 4, MUTED, 12, font_name="Microsoft YaHei UI", anchor_x="right", anchor_y="top")
        metrics = (
            ("货车数量", f"{player.truck_count} 辆", TEXT_DARK),
            ("总运力", str(player.truck_total_capacity), TEXT_DARK),
            ("当前耐久", f"{player.truck_durability}%", NEGATIVE if player.truck_durability < 50 else ACCENT_TEAL),
            ("购车单价", f"{rules.purchase_price:,.2f}", TEXT_DARK),
        )
        metric_top = bounds.top - 42
        metric_bottom = metric_top - 62
        cell_width = bounds.width / len(metrics)
        for index, (label, value, color) in enumerate(metrics):
            cell = Rect(bounds.left + index * cell_width, metric_bottom, bounds.left + (index + 1) * cell_width, metric_top)
            draw_status_value(cell, label, value, color=color)
            if index:
                arcade.draw_line(cell.left, cell.bottom + 5, cell.left, cell.top - 5, MUTED, 1)

        maintenance_line = metric_bottom - 20
        arcade.draw_line(bounds.left, maintenance_line, bounds.right, maintenance_line, MUTED, 1)
        arcade.draw_text("车辆维护", bounds.left, maintenance_line - 20, TEXT_DARK, 15, font_name="Microsoft YaHei UI", bold=True, anchor_y="top")
        arcade.draw_text(
            f"维修将耗时 {rules.repair_days} 天；耐久低于 50% 时建议优先处理。",
            bounds.left,
            maintenance_line - 48,
            MUTED,
            12,
            font_name="Microsoft YaHei UI",
            anchor_y="top",
        )
        repair_y = maintenance_line - 104
        self._draw_action_button(
            Rect(bounds.left, repair_y, bounds.left + 112, repair_y + 32),
            "维修",
            "repair",
            enabled=player.truck_durability < Decimal("100"),
            emphasis=True,
        )

        purchase_line = repair_y - 42
        arcade.draw_line(bounds.left, purchase_line, bounds.right, purchase_line, MUTED, 1)
        arcade.draw_text("增购货车", bounds.left, purchase_line - 20, TEXT_DARK, 15, font_name="Microsoft YaHei UI", bold=True, anchor_y="top")
        arcade.draw_text(
            f"单价 {rules.purchase_price:,.2f}；每辆增加 {rules.capacity_per_vehicle} 运力。",
            bounds.left,
            purchase_line - 48,
            MUTED,
            12,
            font_name="Microsoft YaHei UI",
            anchor_y="top",
        )
        button_y = purchase_line - 104
        self._draw_action_button(Rect(bounds.left, button_y, bounds.left + 36, button_y + 30), "-", "truck-step", -1, enabled=self.truck_quantity > 1)
        quantity_rect = Rect(bounds.left + 42, button_y, bounds.left + 108, button_y + 30)
        draw_sunken_panel(quantity_rect)
        arcade.draw_text(str(self.truck_quantity), quantity_rect.center_x, quantity_rect.center_y, TEXT_DARK, 14, font_name="Microsoft YaHei UI", bold=True, anchor_x="center", anchor_y="center")
        self._draw_action_button(Rect(bounds.left + 114, button_y, bounds.left + 150, button_y + 30), "+", "truck-step", 1)
        arcade.draw_text(
            f"本次支出 {rules.purchase_price * self.truck_quantity:,.2f}",
            bounds.left + 174,
            button_y + 15,
            MUTED,
            12,
            font_name="Microsoft YaHei UI",
            anchor_y="center",
        )
        self._draw_action_button(
            Rect(bounds.right - 112, button_y, bounds.right, button_y + 30),
            "购车",
            "buy-truck",
            enabled=player.cash >= rules.purchase_price * self.truck_quantity,
            emphasis=True,
        )

    def _draw_finance(self, bounds: Rect) -> None:
        state = self.session.state
        debt = total_debt(state.loans)
        credit = available_credit(self.session.catalog, self.session.rules, state)
        has_bank = self.session.catalog.city(state.player.location).has_bank
        repay_limit = min(debt, state.player.cash)
        arcade.draw_text("融资管理", bounds.left, bounds.top, TEXT_DARK, 20, font_name="Microsoft YaHei UI", bold=True, anchor_y="top")
        arcade.draw_text(
            "本地银行可用" if has_bank else "本地没有银行",
            bounds.right,
            bounds.top - 4,
            POSITIVE if has_bank else NEGATIVE,
            12,
            font_name="Microsoft YaHei UI",
            anchor_x="right",
            anchor_y="top",
        )
        metrics = (
            ("未偿债务", f"{debt:,.2f}", NEGATIVE if debt else TEXT_DARK),
            ("可借额度", f"{credit:,.2f}", POSITIVE),
            ("可还上限", f"{repay_limit:,.2f}", TEXT_DARK),
            ("日利率", f"{self.session.rules.finance.daily_interest_rate * 100}%", TEXT_DARK),
        )
        metric_top = bounds.top - 42
        metric_bottom = metric_top - 62
        cell_width = bounds.width / len(metrics)
        for index, (label, value, color) in enumerate(metrics):
            cell = Rect(bounds.left + index * cell_width, metric_bottom, bounds.left + (index + 1) * cell_width, metric_top)
            draw_status_value(cell, label, value, color=color)
            if index:
                arcade.draw_line(cell.left, cell.bottom + 5, cell.left, cell.top - 5, MUTED, 1)

        control_line = metric_bottom - 20
        arcade.draw_line(bounds.left, control_line, bounds.right, control_line, MUTED, 1)
        arcade.draw_text("借款与还款", bounds.left, control_line - 20, TEXT_DARK, 15, font_name="Microsoft YaHei UI", bold=True, anchor_y="top")
        amount_y = control_line - 112
        arcade.draw_text("本次金额", bounds.left, amount_y + 16, MUTED, 12, font_name="Microsoft YaHei UI", anchor_y="center")
        amount_rect = Rect(bounds.left + 92, amount_y, bounds.left + 250, amount_y + 32)
        draw_sunken_panel(amount_rect)
        arcade.draw_text(f"{self.finance_amount:,.2f}", amount_rect.center_x, amount_rect.center_y, TEXT_DARK, 14, font_name="Microsoft YaHei UI", bold=True, anchor_x="center", anchor_y="center")
        self._draw_action_button(Rect(bounds.left + 258, amount_y, bounds.left + 306, amount_y + 32), "-100", "finance-step", -100, enabled=has_bank and self.finance_amount > Decimal("1"))
        self._draw_action_button(Rect(bounds.left + 312, amount_y, bounds.left + 360, amount_y + 32), "+100", "finance-step", 100, enabled=has_bank)
        self._draw_action_button(Rect(bounds.left + 366, amount_y, bounds.left + 436, amount_y + 32), "借满", "borrow-max", enabled=has_bank and credit > 0)
        self._draw_action_button(Rect(bounds.left + 442, amount_y, bounds.left + 512, amount_y + 32), "还清", "repay-max", enabled=has_bank and repay_limit > 0)
        action_line = amount_y - 44
        arcade.draw_line(bounds.left, action_line, bounds.right, action_line, MUTED, 1)
        arcade.draw_text(
            "借款按日计息，还款会优先结清已有债务。",
            bounds.left,
            action_line - 18,
            MUTED,
            12,
            font_name="Microsoft YaHei UI",
            anchor_y="top",
        )
        action_y = action_line - 90
        self._draw_action_button(Rect(bounds.left, action_y, bounds.left + 132, action_y + 32), "借入", "borrow", enabled=has_bank and credit > 0, emphasis=True)
        self._draw_action_button(Rect(bounds.left + 140, action_y, bounds.left + 272, action_y + 32), "还款", "repay", enabled=has_bank and repay_limit > 0)

    def _draw_inventory(self, bounds: Rect) -> None:
        lots = self.session.state.player.cargo_lots
        player = self.session.state.player
        used_capacity = cargo_quantity(lots)
        arcade.draw_text("随车库存", bounds.left, bounds.top, TEXT_DARK, 20, font_name="Microsoft YaHei UI", bold=True, anchor_y="top")
        arcade.draw_text(
            f"已占用 {used_capacity}/{player.truck_total_capacity}",
            bounds.right,
            bounds.top - 4,
            MUTED,
            12,
            font_name="Microsoft YaHei UI",
            anchor_x="right",
            anchor_y="top",
        )
        if not lots:
            arcade.draw_text("当前没有随车货物", bounds.center_x, bounds.top - 116, MUTED, 17, font_name="Microsoft YaHei UI", bold=True, anchor_x="center", anchor_y="top")
            self._draw_action_button(Rect(bounds.left, bounds.bottom + 22, bounds.left + 112, bounds.bottom + 52), "去采购", "goto-market", "buy", emphasis=True)
            return

        header = Rect(bounds.left, bounds.top - 72, bounds.right, bounds.top - 38)
        arcade.draw_lrbt_rectangle_filled(header.left, header.right, header.bottom, header.top, (211, 219, 216))
        columns = (
            ("商品", header.left + 16, "left"),
            ("数量", header.left + header.width * 0.48, "right"),
            ("产地", header.left + header.width * 0.64, "left"),
            ("保质期", header.left + header.width * 0.82, "right"),
            ("在车天数", header.right - 16, "right"),
        )
        for label, x, anchor_x in columns:
            arcade.draw_text(label, x, header.center_y, MUTED, 12, font_name="Microsoft YaHei UI", bold=True, anchor_x=anchor_x, anchor_y="center")
        row_top = header.bottom - 6
        reserved_bottom = bounds.bottom + 68
        visible_rows = max(1, int((row_top - reserved_bottom) // 42))
        for lot in lots[:visible_rows]:
            product = self.session.catalog.product(lot.product_id)
            shelf_life = "-" if lot.shelf_life_remaining_days is None else f"{lot.shelf_life_remaining_days} 天"
            color = NEGATIVE if lot.shelf_life_remaining_days is not None and lot.shelf_life_remaining_days <= 2 else TEXT_DARK
            row = Rect(bounds.left, row_top - 42, bounds.right, row_top)
            arcade.draw_lrbt_rectangle_filled(row.left, row.right, row.bottom, row.top, PANEL_INSET)
            arcade.draw_line(row.left, row.bottom, row.right, row.bottom, MUTED, 1)
            arcade.draw_text(product.name, row.left + 16, row.center_y, color, 14, font_name="Microsoft YaHei UI", bold=True, anchor_y="center")
            arcade.draw_text(str(lot.quantity), row.left + row.width * 0.48, row.center_y, TEXT_DARK, 13, font_name="Microsoft YaHei UI", anchor_x="right", anchor_y="center")
            arcade.draw_text(lot.origin_city, row.left + row.width * 0.64, row.center_y, TEXT_DARK, 13, font_name="Microsoft YaHei UI", anchor_y="center")
            arcade.draw_text(shelf_life, row.left + row.width * 0.82, row.center_y, color, 13, font_name="Microsoft YaHei UI", anchor_x="right", anchor_y="center")
            arcade.draw_text(f"{lot.age_days} 天", row.right - 16, row.center_y, TEXT_DARK, 13, font_name="Microsoft YaHei UI", anchor_x="right", anchor_y="center")
            row_top = row.bottom
        self._draw_action_button(Rect(bounds.left, bounds.bottom + 22, bounds.left + 112, bounds.bottom + 52), "去采购", "goto-market", "buy")
        self._draw_action_button(Rect(bounds.right - 112, bounds.bottom + 22, bounds.right, bounds.bottom + 52), "去出货", "goto-market", "sell", emphasis=True)

    def _draw_journal(self, bounds: Rect) -> None:
        arcade.draw_text("行商路书", bounds.left, bounds.top, TEXT_DARK, 20, font_name="Microsoft YaHei UI", bold=True, anchor_y="top")
        arcade.draw_text(f"最近记下 {len(self.event_log)} 笔", bounds.right, bounds.top - 4, MUTED, 12, font_name="Microsoft YaHei UI", anchor_x="right", anchor_y="top")
        row_top = bounds.top - 46
        maximum_length = max(18, int((bounds.width - 46) / 12))
        for index, item in enumerate(reversed(self.event_log)):
            row = Rect(bounds.left, row_top - 34, bounds.right, row_top)
            arcade.draw_lrbt_rectangle_filled(row.left, row.right, row.bottom, row.top, PANEL_INSET if index % 2 == 0 else (224, 224, 220))
            arcade.draw_text(str(index + 1), row.left + 12, row.center_y, MUTED, 11, font_name="Microsoft YaHei UI", anchor_y="center")
            arcade.draw_text(_short_text(item, maximum_length), row.left + 38, row.center_y, TEXT_DARK, 12, font_name="Microsoft YaHei UI", anchor_y="center")
            row_top = row.bottom - 2
            if row_top - 34 < bounds.bottom:
                break

    def _draw_footer(self, layout: MainLayout) -> None:
        draw_raised_panel(layout.footer)
        state = self.session.state
        selected = self.selected_city or state.player.location
        text_width = layout.footer.width - 216
        max_notice_length = max(16, int(text_width / 12) - 12)
        footer_text = f"{_short_text(self.notice, max_notice_length)}    目标 {selected}"
        arcade.draw_text(
            footer_text,
            layout.footer.left + 10,
            layout.footer.center_y,
            TEXT_DARK,
            12,
            font_name="Microsoft YaHei UI",
            anchor_y="center",
        )
        self._draw_action_button(
            Rect(layout.footer.right - 194, layout.footer.bottom + 6, layout.footer.right - 102, layout.footer.top - 6),
            "新局",
            "new-game",
        )
        self._draw_action_button(
            Rect(layout.footer.right - 96, layout.footer.bottom + 6, layout.footer.right - 6, layout.footer.top - 6),
            "日结",
            "next-day",
            enabled=state.outcome is None,
            emphasis=True,
        )

    def _draw_outcome_modal(self) -> None:
        """在核心会话确认结局后显示唯一的终局操作入口。"""

        arcade.draw_lrbt_rectangle_filled(0, self.width, 0, self.height, (0, 0, 0, 150))
        rect = Rect(self.width / 2 - 190, self.height / 2 - 105, self.width / 2 + 190, self.height / 2 + 105)
        draw_raised_panel(rect)
        draw_title_bar(Rect(rect.left + 3, rect.top - 31, rect.right - 3, rect.top - 3), "本局结束")
        outcome = self.session.state.outcome
        assert outcome is not None
        reason = "破产" if outcome.reason.value == "bankruptcy" else "达到挑战时间上限"
        arcade.draw_text(reason, rect.center_x, rect.center_y + 24, TEXT_DARK, 18, font_name="Microsoft YaHei UI", bold=True, anchor_x="center", anchor_y="center")
        arcade.draw_text(f"结算资产 {outcome.final_assets:,.2f}", rect.center_x, rect.center_y - 8, MUTED, 13, font_name="Microsoft YaHei UI", anchor_x="center", anchor_y="center")
        self._draw_action_button(Rect(rect.left + 64, rect.bottom + 20, rect.right - 64, rect.bottom + 52), "开始新局", "new-game", emphasis=True)

    def _draw_action_button(
        self,
        rect: Rect,
        label: str,
        action: str,
        value: str | int | TransportMode | tuple[str, str] | None = None,
        *,
        enabled: bool = True,
        emphasis: bool = False,
    ) -> None:
        """注册命中区域并以统一的 x86 控件风格绘制命令按钮。"""

        self._action_hitboxes.append(ActionHitbox(action, rect, value, enabled))
        draw_command_button(
            rect,
            label,
            enabled=enabled,
            hovered=self._is_hovered(action, value),
            emphasis=emphasis,
        )

    def _is_hovered(self, action: str, value: str | int | TransportMode | tuple[str, str] | None) -> bool:
        return self.hovered_action == (action, value)

    def _handle_action(self, hitbox: ActionHitbox) -> None:
        """处理纯界面状态，或构造一条命令交给核心会话执行。"""

        action = hitbox.action
        if action == "open-order":
            assert isinstance(hitbox.value, tuple)
            mode, product_id = hitbox.value
            self.trade_order = TradeOrder(mode, product_id)
            return
        if action == "order-step":
            assert self.trade_order is not None
            assert isinstance(hitbox.value, int)
            _unit_price, maximum = self._order_quote(self.trade_order.mode, self.trade_order.product_id)
            quantity = max(1, min(maximum, self.trade_order.quantity + hitbox.value))
            self.trade_order = replace(self.trade_order, quantity=quantity)
            return
        if action == "order-max":
            assert self.trade_order is not None
            _unit_price, maximum = self._order_quote(self.trade_order.mode, self.trade_order.product_id)
            self.trade_order = replace(self.trade_order, quantity=maximum)
            return
        if action == "close-order":
            self.trade_order = None
            return
        if action == "confirm-order":
            assert self.trade_order is not None
            order = self.trade_order
            self.trade_order = None
            command: Command = Buy(order.product_id, order.quantity) if order.mode == "buy" else Sell(order.product_id, order.quantity)
            self._dispatch(command)
            return
        if action == "select-city":
            assert isinstance(hitbox.value, str)
            self.selected_city = hitbox.value
            self.fast_travel = False
            self.notice = f"已选择 {hitbox.value} 作为调度目标。"
            return
        if action == "select-market-city":
            assert isinstance(hitbox.value, str)
            self.market_city = hitbox.value
            self.market_page = 0
            return
        if action == "market-page":
            assert isinstance(hitbox.value, int)
            page_size = MARKET_PAGE_SIZE
            page_count = (len(self.session.catalog.products) + page_size - 1) // page_size
            self.market_page = max(0, min(page_count - 1, self.market_page + hitbox.value))
            return
        if action == "toggle-fast":
            self.fast_travel = not self.fast_travel
            return
        if action == "travel":
            assert isinstance(hitbox.value, TransportMode)
            destination = self.selected_city or self.session.state.player.location
            self._dispatch(Travel(destination, hitbox.value, fast=self.fast_travel))
            return
        if action == "truck-step":
            assert isinstance(hitbox.value, int)
            self.truck_quantity = max(1, self.truck_quantity + hitbox.value)
            return
        if action == "repair":
            self._dispatch(RepairTruck())
            return
        if action == "buy-truck":
            self._dispatch(BuyTruck(self.truck_quantity))
            return
        if action == "finance-step":
            assert isinstance(hitbox.value, int)
            self.finance_amount = max(Decimal("1"), self.finance_amount + Decimal(hitbox.value))
            return
        if action == "borrow-max":
            self.finance_amount = available_credit(self.session.catalog, self.session.rules, self.session.state)
            return
        if action == "repay-max":
            self.finance_amount = min(total_debt(self.session.state.loans), self.session.state.player.cash)
            return
        if action == "borrow":
            amount = min(self.finance_amount, available_credit(self.session.catalog, self.session.rules, self.session.state))
            self.finance_amount = amount
            self._dispatch(Borrow(amount))
            return
        if action == "repay":
            amount = min(self.finance_amount, total_debt(self.session.state.loans), self.session.state.player.cash)
            self.finance_amount = amount
            self._dispatch(Repay(amount))
            return
        if action == "goto-market":
            self.active_tab = "出售" if hitbox.value == "sell" else "采购"
            return
        if action == "next-day":
            self._dispatch(NextDay())
            return
        if action == "new-game":
            self._restart_game()
            return
        raise RuntimeError(f"未知图形操作：{action}")

    def _dispatch(self, command: Command) -> None:
        """执行命令、记录领域事件，并同步所有依赖状态的界面数据。"""

        result = self.session.dispatch(command)
        self._price_cache.clear()
        if result.rejection is not None:
            self.notice = f"操作未执行：{result.rejection.message}"
            self.event_log.append(self.notice)
            return
        for event in result.events:
            self.event_log.append(_format_event(event, self.session))
        self.notice = _format_event(result.events[0], self.session) if result.events else "这一项安排已经完成。"
        self.selected_city = self.session.state.player.location
        self.fast_travel = False

    def _restart_game(self) -> None:
        self.session = create_game_session(seed=self._seed, mode=self._mode)
        self.active_tab = "采购"
        self.selected_city = self.session.state.player.location
        self.market_city = self.session.state.player.location
        self.market_page = 0
        self.truck_quantity = 1
        self.finance_amount = FINANCE_STEP
        self.fast_travel = False
        self.trade_order = None
        self._price_cache.clear()
        self._route_cache.clear()
        self.event_log.clear()
        self.event_log.append("新的行程从郑州货站开始。")
        self.notice = "郑州货站正在等候调度。"

    def _market_prices(self, product_id: str, city_name: str) -> tuple[Decimal | None, Decimal]:
        key = (city_name, product_id)
        cached = self._price_cache.get(key)
        if cached is not None:
            return cached
        product = self.session.catalog.product(product_id)
        purchase = (
            purchase_unit_price(self.session.catalog, self.session.rules, self.session.state, product_id, city_name)
            if city_name in product.origins
            else None
        )
        origin_city = reference_sale_origin(self.session.catalog, product, city_name)
        sale = sale_unit_price(
            self.session.catalog,
            self.session.rules,
            self.session.state,
            product_id,
            city_name,
            origin_city=origin_city,
            remote_distance_premium=remote_sale_distance_premium(
                self.session.catalog,
                self.session.rules,
                origin_city,
                city_name,
            ),
        )
        prices = (purchase, sale)
        self._price_cache[key] = prices
        return prices

    def _reachable_routes(self, city_name: str) -> dict[str, tuple[ReachableRoute, ...]]:
        """按区域和目录顺序列出从当前城市可完成的单一运输方式行程。"""

        catalog = self.session.catalog
        cached = self._route_cache.get(city_name)
        if cached is not None:
            return cached
        origin = catalog.city(city_name)
        grouped: dict[str, tuple[ReachableRoute, ...]] = {}
        regions = tuple(dict.fromkeys(city.region for city in catalog.cities.values()))
        for region in regions:
            for destination in catalog.cities.values():
                if destination.name == city_name or destination.region != region:
                    continue
                routes: list[ReachableRoute] = []
                for mode in (TransportMode.LAND, TransportMode.SEA):
                    if mode not in origin.modes or mode not in destination.modes:
                        continue
                    try:
                        distance_km = shortest_distance(catalog, city_name, destination.name, mode)
                    except RouteNotFound:
                        continue
                    routes.append(ReachableRoute(mode=mode, distance_km=distance_km))
                if routes:
                    grouped[destination.name] = tuple(routes)
        self._route_cache[city_name] = grouped
        return grouped


def _event_label(name: str) -> str:
    return {
        "day_advanced": "进入下一天",
        "days_settled": "日结完成",
        "goods_bought": "采购完成",
        "goods_sold": "出售完成",
        "travel_completed": "运输完成",
        "truck_repaired": "车辆维修完成",
        "trucks_bought": "购车完成",
        "loan_borrowed": "借贷完成",
        "loan_repaid": "还款完成",
        "cargo_lost_in_transit": "运输货损",
        "game_finished": "本局结束",
    }.get(name, "完成了一项安排")


def _format_event(event: GameEvent, session: GameSession) -> str:
    """将领域事件写成行程记事，而不是暴露字段名和内部事件代码。"""

    attributes = event.attributes
    if event.name == "day_advanced":
        return f"第 {attributes['day']} 天开始。"
    if event.name == "days_settled":
        parts = [f"完成 {attributes['days']} 天日结"]
        if attributes["interest_accrued"]:
            parts.append(f"利息 {attributes['interest_accrued']:,.2f}")
        if attributes["labor_cost"]:
            parts.append(f"车队开销 {attributes['labor_cost']:,.2f}")
        if attributes["expired_cargo"]:
            parts.append(f"损耗货物 {attributes['expired_cargo']} 单位")
        return "，".join(parts) + "。"
    if event.name in {"goods_bought", "goods_sold"}:
        product = session.catalog.product(str(attributes["product_id"])).name
        verb = "买入" if event.name == "goods_bought" else "卖出"
        cash_word = "花费" if event.name == "goods_bought" else "收入"
        return (
            f"在 {session.state.player.location}{verb} {attributes['quantity']} 份{product}，"
            f"{cash_word} {attributes['total']:,.2f}。"
        )
    if event.name == "travel_completed":
        mode = "陆运" if attributes["mode"] == TransportMode.LAND.value else "海运"
        return (
            f"从 {attributes['origin']} 经{mode}抵达 {attributes['destination']}，"
            f"行程 {attributes['days']} 天，运费 {attributes['cost']:,.2f}。"
        )
    if event.name == "cargo_lost_in_transit":
        return f"途中损失了 {attributes['quantity']} 单位货物。"
    if event.name == "truck_repaired":
        return f"车队完成维修，耗时 {attributes['days']} 天，花费 {attributes['cost']:,.2f}。"
    if event.name == "trucks_bought":
        return (
            f"新增 {attributes['quantity']} 辆货车，"
            f"运力增加 {attributes['capacity_added']}，花费 {attributes['cost']:,.2f}。"
        )
    if event.name == "loan_borrowed":
        return f"向本地银行借入 {attributes['amount']:,.2f}。"
    if event.name == "loan_repaid":
        return f"偿还 {attributes['amount']:,.2f}，剩余债务 {attributes['remaining_debt']:,.2f}。"
    if event.name == "game_finished":
        return f"本局告一段落，结算资产 {attributes['final_assets']:,.2f}。"
    return _event_label(event.name)


def _short_text(value: str, maximum_length: int) -> str:
    """在固定高度状态栏中保留完整前缀，避免提示文字覆盖底部命令。"""

    return value if len(value) <= maximum_length else f"{value[: maximum_length - 3]}..."


def _format_market_message(
    market_scope: str,
    product_name: str,
    kind: MarketEventKind,
    remaining_days: int,
) -> str:
    """将公开市场事件改写为玩家能直接采取行动的行情句子。"""

    if kind is MarketEventKind.SURPLUS:
        return f"{market_scope}：{product_name}库存积压，当前进货价格偏低，预计持续 {remaining_days} 天。"
    return f"{market_scope}：{product_name}货源紧张，当地出货价格走高，预计持续 {remaining_days} 天。"


def _category_label(category: str) -> str:
    return {
        "base": "基础物资",
        "light_industry": "轻工业",
        "electronics": "电子产品",
        "perishable": "生鲜",
    }[category]


def _price_change(history: tuple[Decimal, ...], mode: str) -> tuple[str, tuple[int, int, int]]:
    """返回相对昨日的变动；采购时降价更有利，出售时涨价更有利。"""

    if len(history) < 2:
        return "--", MUTED
    previous, current = history[-2:]
    change = (current - previous) / previous * Decimal("100")
    if change == 0:
        return "0.0%", MUTED
    favorable = change < 0 if mode == "buy" else change > 0
    return f"{change:+.1f}%", POSITIVE if favorable else NEGATIVE


def run_arcade_game(*, seed: int | None = None, mode: GameMode = GameMode.FREE) -> TradeGameWindow:
    """创建图形窗口并将控制权交给 Arcade 主循环。"""

    window = TradeGameWindow(create_game_session(seed=seed, mode=mode), seed=seed)
    arcade.run()
    return window
