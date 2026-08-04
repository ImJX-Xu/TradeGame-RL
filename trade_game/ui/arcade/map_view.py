"""绘制货运调度台中的路线、站点和当前位置标记。"""

from __future__ import annotations

from collections.abc import Mapping
from math import hypot

import arcade

from trade_game.core import Catalog, City, GameState, Route, TransportMode

from .theme import (
    CURRENT_CITY,
    HIGH_CONSUMPTION_CITY,
    LAND_ROUTE,
    MAP_LABEL_SHADOW,
    MAP_STATION_SHADOW,
    MAP_SURFACE,
    MAP_TICK,
    MAP_TRACK_SHADOW,
    MAP_ROUTE_TIE,
    NORMAL_CITY,
    Rect,
    SEA_ROUTE,
    SELECTED_CITY,
    TEXT_LIGHT,
)


# 这是调度示意图，而非等比例地理地图。固定的相对位置确保站点、站名和可达路线
# 在各个受支持的窗口尺寸下始终清晰。
_CITY_LAYOUT: Mapping[str, tuple[float, float]] = {
    "郑州": (0.34, 0.51),
    "石家庄": (0.24, 0.69),
    "太原": (0.11, 0.60),
    "北京": (0.39, 0.81),
    "沈阳": (0.57, 0.70),
    "长春": (0.73, 0.80),
    "哈尔滨": (0.88, 0.88),
    "上海": (0.68, 0.51),
    "福州": (0.64, 0.30),
    "广州": (0.34, 0.17),
    "深圳": (0.48, 0.12),
    "海南": (0.11, 0.13),
    "台北": (0.88, 0.31),
    "高雄": (0.81, 0.13),
}

_CITY_LABEL_OFFSETS: Mapping[str, tuple[float, float, str]] = {
    "郑州": (13, -18, "left"),
    "石家庄": (11, 12, "left"),
    "太原": (11, 12, "left"),
    "北京": (-11, 12, "right"),
    "沈阳": (11, 12, "left"),
    "长春": (11, -20, "left"),
    "哈尔滨": (-11, -20, "right"),
    "上海": (11, 12, "left"),
    "福州": (11, 12, "left"),
    "广州": (-11, 12, "right"),
    "深圳": (11, -20, "left"),
    "海南": (11, 12, "left"),
    "台北": (11, 12, "left"),
    "高雄": (11, -20, "left"),
}


def city_positions(catalog: Catalog, bounds: Rect) -> Mapping[str, tuple[float, float]]:
    """将目录中的城市映射到调度台的稳定坐标。"""

    return {
        city.name: (
            bounds.left + bounds.width * _CITY_LAYOUT[city.name][0],
            bounds.bottom + bounds.height * _CITY_LAYOUT[city.name][1],
        )
        for city in catalog.cities.values()
    }


def draw_map(
    catalog: Catalog,
    state: GameState,
    bounds: Rect,
    *,
    selected_city: str | None,
) -> Mapping[str, tuple[float, float]]:
    """绘制以当前调度焦点为中心的可达路线和站点。"""

    focus_city = selected_city or state.player.location
    positions = city_positions(catalog, bounds)
    _draw_dispatch_surface(bounds)
    _draw_connected_routes(catalog.routes, positions, focus_city)
    for city in catalog.cities.values():
        _draw_station(
            city,
            *positions[city.name],
            is_current=city.name == state.player.location,
            is_selected=city.name == focus_city,
        )
    _draw_header(bounds, state.day, state.player.location, focus_city)
    _draw_route_strip(catalog, bounds, state.player.location, focus_city)
    return positions


def _draw_dispatch_surface(bounds: Rect) -> None:
    """绘制简洁的调度台底面和边缘刻度，不使用泛用仪表盘网格。"""

    arcade.draw_lrbt_rectangle_filled(bounds.left, bounds.right, bounds.bottom, bounds.top, MAP_SURFACE)
    arcade.draw_lrbt_rectangle_outline(
        bounds.left + 1,
        bounds.right - 1,
        bounds.bottom + 1,
        bounds.top - 1,
        MAP_TICK,
        1,
    )
    spacing = 32
    for x in range(int(bounds.left) + spacing, int(bounds.right), spacing):
        arcade.draw_line(x, bounds.bottom + 1, x, bounds.bottom + 6, MAP_TICK, 1)
        arcade.draw_line(x, bounds.top - 1, x, bounds.top - 4, MAP_TICK, 1)
    for y in range(int(bounds.bottom) + spacing, int(bounds.top), spacing):
        arcade.draw_line(bounds.left + 1, y, bounds.left + 6, y, MAP_TICK, 1)
        arcade.draw_line(bounds.right - 1, y, bounds.right - 4, y, MAP_TICK, 1)


def _draw_connected_routes(
    routes: tuple[Route, ...],
    positions: Mapping[str, tuple[float, float]],
    focus_city: str,
) -> None:
    """只点亮调度焦点的直达路线，让地图直接对应当前决策。"""

    connected = tuple(route for route in routes if focus_city in (route.from_city, route.to_city))
    for route in connected:
        start, end = _parallel_route_endpoints(route, connected, positions)
        if route.mode is TransportMode.LAND:
            _draw_land_route(start, end)
        else:
            _draw_sea_route(start, end)


def _parallel_route_endpoints(
    route: Route,
    connected: tuple[Route, ...],
    positions: Mapping[str, tuple[float, float]],
) -> tuple[tuple[float, float], tuple[float, float]]:
    """为同一对站点的陆海双路线留出可辨识的平行间距。"""

    start = positions[route.from_city]
    end = positions[route.to_city]
    has_parallel_route = any(
        other is not route
        and {other.from_city, other.to_city} == {route.from_city, route.to_city}
        for other in connected
    )
    if not has_parallel_route:
        return start, end
    length = hypot(end[0] - start[0], end[1] - start[1])
    offset = 3 if route.mode is TransportMode.LAND else -3
    offset_x = -(end[1] - start[1]) / length * offset
    offset_y = (end[0] - start[0]) / length * offset
    return (
        (start[0] + offset_x, start[1] + offset_y),
        (end[0] + offset_x, end[1] + offset_y),
    )


def _draw_land_route(start: tuple[float, float], end: tuple[float, float]) -> None:
    """将陆路画成带轨枕的运输干线。"""

    arcade.draw_line(*start, *end, MAP_TRACK_SHADOW, 6)
    arcade.draw_line(*start, *end, LAND_ROUTE, 2)
    length = hypot(end[0] - start[0], end[1] - start[1])
    normal_x = -(end[1] - start[1]) / length
    normal_y = (end[0] - start[0]) / length
    for distance in range(14, int(length), 20):
        ratio = distance / length
        x = start[0] + (end[0] - start[0]) * ratio
        y = start[1] + (end[1] - start[1]) * ratio
        arcade.draw_line(
            x - normal_x * 3,
            y - normal_y * 3,
            x + normal_x * 3,
            y + normal_y * 3,
            MAP_ROUTE_TIE,
            1,
        )


def _draw_sea_route(start: tuple[float, float], end: tuple[float, float]) -> None:
    """将海路画成间断航线，以区别于货车运输干线。"""

    arcade.draw_line(*start, *end, MAP_TRACK_SHADOW, 5)
    length = hypot(end[0] - start[0], end[1] - start[1])
    segment_length = 10
    gap_length = 6
    for distance in range(0, int(length), segment_length + gap_length):
        segment_end = min(distance + segment_length, length)
        start_ratio = distance / length
        end_ratio = segment_end / length
        arcade.draw_line(
            start[0] + (end[0] - start[0]) * start_ratio,
            start[1] + (end[1] - start[1]) * start_ratio,
            start[0] + (end[0] - start[0]) * end_ratio,
            start[1] + (end[1] - start[1]) * end_ratio,
            SEA_ROUTE,
            2,
        )


def _draw_station(
    city: City,
    x: float,
    y: float,
    *,
    is_current: bool,
    is_selected: bool,
) -> None:
    """按城市能力绘制货运站、高消费商圈与当前货车。"""

    if is_selected:
        _draw_selection_brackets(x, y)
    if city.is_high_consumption:
        _draw_market_station(x, y)
    else:
        _draw_standard_station(x, y)
    if city.has_port:
        arcade.draw_line(x - 6, y - 9, x + 6, y - 9, SEA_ROUTE, 2)
    if city.has_bank:
        arcade.draw_circle_filled(x, y + 9, 2, LAND_ROUTE)
    if is_current:
        _draw_truck_marker(x, y)
    _draw_city_label(city.name, x, y)


def _draw_standard_station(x: float, y: float) -> None:
    arcade.draw_lrbt_rectangle_filled(x - 7, x + 7, y - 7, y + 7, MAP_STATION_SHADOW)
    arcade.draw_lrbt_rectangle_filled(x - 5, x + 5, y - 5, y + 5, NORMAL_CITY)
    arcade.draw_line(x - 3, y, x + 3, y, MAP_STATION_SHADOW, 1)


def _draw_market_station(x: float, y: float) -> None:
    arcade.draw_polygon_filled(
        ((x, y + 8), (x + 8, y), (x, y - 8), (x - 8, y)),
        MAP_STATION_SHADOW,
    )
    arcade.draw_polygon_filled(
        ((x, y + 6), (x + 6, y), (x, y - 6), (x - 6, y)),
        HIGH_CONSUMPTION_CITY,
    )


def _draw_selection_brackets(x: float, y: float) -> None:
    radius = 16
    length = 6
    arcade.draw_line(x - radius, y - radius, x - radius + length, y - radius, SELECTED_CITY, 2)
    arcade.draw_line(x - radius, y - radius, x - radius, y - radius + length, SELECTED_CITY, 2)
    arcade.draw_line(x + radius, y - radius, x + radius - length, y - radius, SELECTED_CITY, 2)
    arcade.draw_line(x + radius, y - radius, x + radius, y - radius + length, SELECTED_CITY, 2)
    arcade.draw_line(x - radius, y + radius, x - radius + length, y + radius, SELECTED_CITY, 2)
    arcade.draw_line(x - radius, y + radius, x - radius, y + radius - length, SELECTED_CITY, 2)
    arcade.draw_line(x + radius, y + radius, x + radius - length, y + radius, SELECTED_CITY, 2)
    arcade.draw_line(x + radius, y + radius, x + radius, y + radius - length, SELECTED_CITY, 2)


def _draw_truck_marker(x: float, y: float) -> None:
    """以小型货车标记当前位置，替代抽象的高亮圆环。"""

    truck_y = y + 16
    arcade.draw_lrbt_rectangle_filled(x - 12, x + 2, truck_y - 4, truck_y + 4, MAP_STATION_SHADOW)
    arcade.draw_lrbt_rectangle_filled(x - 10, x, truck_y - 2, truck_y + 3, CURRENT_CITY)
    arcade.draw_polygon_filled(
        ((x + 1, truck_y - 2), (x + 8, truck_y - 2), (x + 8, truck_y + 3), (x + 4, truck_y + 3)),
        CURRENT_CITY,
    )
    arcade.draw_circle_filled(x - 6, truck_y - 4, 2, MAP_STATION_SHADOW)
    arcade.draw_circle_filled(x + 5, truck_y - 4, 2, MAP_STATION_SHADOW)


def _draw_city_label(name: str, x: float, y: float) -> None:
    offset_x, offset_y, anchor_x = _CITY_LABEL_OFFSETS[name]
    label_x = x + offset_x
    label_y = y + offset_y
    arcade.draw_text(
        name,
        label_x + 1,
        label_y - 1,
        MAP_LABEL_SHADOW,
        12,
        font_name="Microsoft YaHei UI",
        anchor_x=anchor_x,
        anchor_y="center",
    )
    arcade.draw_text(
        name,
        label_x,
        label_y,
        TEXT_LIGHT,
        12,
        font_name="Microsoft YaHei UI",
        anchor_x=anchor_x,
        anchor_y="center",
    )


def _draw_header(bounds: Rect, day: int, current_city: str, focus_city: str) -> None:
    arcade.draw_text(
        "货运调度",
        bounds.left + 14,
        bounds.top - 12,
        TEXT_LIGHT,
        14,
        font_name="Microsoft YaHei UI",
        bold=True,
        anchor_y="top",
    )
    arcade.draw_text(
        f"D{day:03}  {current_city}站  {'本地待命' if current_city == focus_city else f'规划至 {focus_city}'}",
        bounds.right - 14,
        bounds.top - 14,
        TEXT_LIGHT,
        10,
        font_name="Microsoft YaHei UI",
        anchor_x="right",
        anchor_y="top",
    )


def _draw_route_strip(catalog: Catalog, bounds: Rect, current_city: str, focus_city: str) -> None:
    line_y = bounds.bottom + 31
    arcade.draw_line(bounds.left + 12, line_y, bounds.right - 12, line_y, MAP_TICK, 1)
    routes = tuple(
        route
        for route in catalog.routes
        if {route.from_city, route.to_city} == {current_city, focus_city}
    )
    if current_city == focus_city:
        summary = f"{current_city}站  |  直达路线 {sum(current_city in (route.from_city, route.to_city) for route in catalog.routes)} 条"
    else:
        route_text = "  /  ".join(
            f"{'陆运' if route.mode is TransportMode.LAND else '海运'} {route.distance_km} km"
            for route in routes
        )
        summary = f"调度计划  {current_city}  ->  {focus_city}  |  {route_text}"
    arcade.draw_text(
        summary,
        bounds.left + 14,
        bounds.bottom + 11,
        TEXT_LIGHT,
        11,
        font_name="Microsoft YaHei UI",
        anchor_y="bottom",
    )
