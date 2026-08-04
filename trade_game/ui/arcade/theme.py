"""图形界面的 x86 风格颜色、尺寸与基础绘制控件。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import arcade


Color: TypeAlias = tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class Rect:
    """使用左、下、右、上的稳定矩形坐标。"""

    left: float
    bottom: float
    right: float
    top: float

    @property
    def width(self) -> float:
        return self.right - self.left

    @property
    def height(self) -> float:
        return self.top - self.bottom

    @property
    def center_x(self) -> float:
        return (self.left + self.right) / 2

    @property
    def center_y(self) -> float:
        return (self.bottom + self.top) / 2

    def inset(self, amount: float) -> "Rect":
        return Rect(self.left + amount, self.bottom + amount, self.right - amount, self.top - amount)

    def contains(self, x: float, y: float) -> bool:
        return self.left <= x <= self.right and self.bottom <= y <= self.top


APP_BACKGROUND: Color = (32, 34, 35)
DESKTOP_SURFACE: Color = (104, 111, 106)
PANEL_FACE: Color = (192, 192, 192)
PANEL_INSET: Color = (232, 232, 228)
PANEL_DISABLED: Color = (158, 158, 154)
HIGHLIGHT: Color = (255, 255, 255)
SHADOW: Color = (72, 72, 72)
DARK_SHADOW: Color = (25, 25, 25)
TITLE_BLUE: Color = (0, 72, 156)
TITLE_BLUE_DARK: Color = (0, 45, 104)
MAP_SURFACE: Color = (27, 45, 47)
MAP_TICK: Color = (68, 96, 94)
MAP_TRACK_SHADOW: Color = (13, 25, 27)
MAP_ROUTE_TIE: Color = (119, 98, 57)
MAP_STATION_SHADOW: Color = (14, 27, 29)
MAP_LABEL_SHADOW: Color = (11, 20, 21)
LAND_ROUTE: Color = (180, 161, 93)
SEA_ROUTE: Color = (84, 180, 194)
CURRENT_CITY: Color = (255, 202, 86)
SELECTED_CITY: Color = (255, 255, 255)
NORMAL_CITY: Color = (218, 229, 218)
HIGH_CONSUMPTION_CITY: Color = (233, 146, 94)
TEXT_DARK: Color = (24, 24, 24)
TEXT_LIGHT: Color = (242, 242, 236)
POSITIVE: Color = (30, 115, 66)
NEGATIVE: Color = (176, 47, 42)
MUTED: Color = (95, 95, 91)
ACCENT_TEAL: Color = (23, 125, 127)
BUTTON_HOVER: Color = (218, 218, 214)
BUTTON_ACCENT: Color = (18, 104, 105)
BUTTON_ACCENT_HOVER: Color = (26, 133, 135)

FONT_NAME = "Microsoft YaHei UI"


def draw_raised_panel(rect: Rect, *, fill: Color = PANEL_FACE) -> None:
    """绘制带高光和阴影边缘的经典凸起面板。"""

    arcade.draw_lrbt_rectangle_filled(rect.left, rect.right, rect.bottom, rect.top, fill)
    arcade.draw_line(rect.left, rect.bottom, rect.left, rect.top, HIGHLIGHT, 2)
    arcade.draw_line(rect.left, rect.top, rect.right, rect.top, HIGHLIGHT, 2)
    arcade.draw_line(rect.left, rect.bottom, rect.right, rect.bottom, DARK_SHADOW, 2)
    arcade.draw_line(rect.right, rect.bottom, rect.right, rect.top, DARK_SHADOW, 2)


def draw_sunken_panel(rect: Rect, *, fill: Color = PANEL_INSET) -> None:
    """绘制信息区使用的凹陷面板。"""

    arcade.draw_lrbt_rectangle_filled(rect.left, rect.right, rect.bottom, rect.top, fill)
    arcade.draw_line(rect.left, rect.bottom, rect.left, rect.top, DARK_SHADOW, 2)
    arcade.draw_line(rect.left, rect.bottom, rect.right, rect.bottom, DARK_SHADOW, 2)
    arcade.draw_line(rect.left, rect.top, rect.right, rect.top, HIGHLIGHT, 2)
    arcade.draw_line(rect.right, rect.bottom, rect.right, rect.top, HIGHLIGHT, 2)


def draw_title_bar(rect: Rect, title: str) -> None:
    """绘制深蓝标题栏及其窗口名称。"""

    arcade.draw_lrbt_rectangle_filled(rect.left, rect.right, rect.bottom, rect.top, TITLE_BLUE)
    arcade.draw_text(
        title,
        rect.left + 8,
        rect.center_y,
        TEXT_LIGHT,
        14,
        font_name=FONT_NAME,
        bold=True,
        anchor_y="center",
    )


def draw_tab(rect: Rect, label: str, *, active: bool, hovered: bool) -> None:
    """绘制固定尺寸的页签，激活状态使用内凹效果。"""

    fill = PANEL_INSET if active else ((218, 218, 214) if hovered else PANEL_FACE)
    (draw_sunken_panel if active else draw_raised_panel)(rect, fill=fill)
    arcade.draw_text(
        label,
        rect.center_x,
        rect.center_y,
        TEXT_DARK,
        13,
        font_name=FONT_NAME,
        bold=active,
        anchor_x="center",
        anchor_y="center",
    )


def draw_navigation_item(rect: Rect, label: str, *, active: bool, hovered: bool, shortcut: int) -> None:
    """绘制左侧主导航；导航使用完整行高，避免中文页签横向拥挤。"""

    fill = (210, 224, 220) if active else (BUTTON_HOVER if hovered else PANEL_FACE)
    (draw_sunken_panel if active else draw_raised_panel)(rect, fill=fill)
    if active:
        arcade.draw_lrbt_rectangle_filled(rect.left + 5, rect.left + 9, rect.bottom + 5, rect.top - 5, ACCENT_TEAL)
    arcade.draw_text(
        label,
        rect.left + 18,
        rect.center_y,
        TEXT_DARK,
        14,
        font_name=FONT_NAME,
        bold=active,
        anchor_y="center",
    )
    arcade.draw_text(
        str(shortcut),
        rect.right - 12,
        rect.center_y,
        MUTED,
        11,
        font_name=FONT_NAME,
        anchor_x="right",
        anchor_y="center",
    )


def draw_command_button(
    rect: Rect,
    label: str,
    *,
    enabled: bool,
    hovered: bool,
    emphasis: bool = False,
) -> None:
    """绘制带禁用、悬停和强调状态的操作按钮。"""

    if not enabled:
        fill = PANEL_DISABLED
        text_color = MUTED
    elif emphasis:
        fill = BUTTON_ACCENT_HOVER if hovered else BUTTON_ACCENT
        text_color = TEXT_LIGHT
    else:
        fill = BUTTON_HOVER if hovered else PANEL_FACE
        text_color = TEXT_DARK
    draw_raised_panel(rect, fill=fill)
    arcade.draw_text(
        label,
        rect.center_x,
        rect.center_y,
        text_color,
        12,
        font_name=FONT_NAME,
        bold=emphasis,
        anchor_x="center",
        anchor_y="center",
    )


def draw_toggle(rect: Rect, label: str, *, checked: bool, hovered: bool, enabled: bool) -> None:
    """绘制用于路线加急等二元状态的复选开关。"""

    box = Rect(rect.left, rect.center_y - 8, rect.left + 16, rect.center_y + 8)
    draw_sunken_panel(box, fill=PANEL_INSET if enabled else PANEL_DISABLED)
    if checked:
        arcade.draw_line(box.left + 3, box.center_y, box.center_x - 1, box.bottom + 4, ACCENT_TEAL, 2)
        arcade.draw_line(box.center_x - 1, box.bottom + 4, box.right - 3, box.top - 4, ACCENT_TEAL, 2)
    text_color = TEXT_DARK if enabled else MUTED
    if hovered and enabled:
        text_color = ACCENT_TEAL
    arcade.draw_text(
        label,
        box.right + 7,
        rect.center_y,
        text_color,
        12,
        font_name=FONT_NAME,
        anchor_y="center",
    )


def draw_status_value(rect: Rect, label: str, value: str, *, color: Color = TEXT_DARK) -> None:
    """绘制状态条中的固定宽度指标。"""

    arcade.draw_text(
        label,
        rect.left + 8,
        rect.top - 7,
        MUTED,
        11,
        font_name=FONT_NAME,
        anchor_y="top",
    )
    value_size = 14 if len(value) > 13 else 16
    arcade.draw_text(
        value,
        rect.left + 8,
        rect.bottom + 7,
        color,
        value_size,
        font_name=FONT_NAME,
        bold=True,
        anchor_y="bottom",
    )
