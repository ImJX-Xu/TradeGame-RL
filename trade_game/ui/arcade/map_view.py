"""绘制货运调度台中的路线、站点和当前位置标记。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import hypot

import arcade

from trade_game.core import Catalog, City, GameState, TransportMode

from .theme import (
    CURRENT_CITY,
    HIGH_CONSUMPTION_CITY,
    LAND_ROUTE,
    MAP_LABEL_SHADOW,
    MAP_REGION_BORDER,
    MAP_REGION_LABEL,
    MAP_REGION_SURFACE,
    MAP_STATION_SHADOW,
    MAP_SURFACE,
    MAP_TICK,
    MAP_TRACK_SHADOW,
    NORMAL_CITY,
    Rect,
    SEA_ROUTE,
    SELECTED_CITY,
    TEXT_LIGHT,
)


@dataclass(frozen=True, slots=True)
class ReachableRoute:
    """界面使用的一种可达运输方式及其最短总里程。"""

    mode: TransportMode
    distance_km: int


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
    "上海": (0.68, 0.43),
    "福州": (0.64, 0.30),
    "广州": (0.34, 0.17),
    "深圳": (0.48, 0.12),
    "海南": (0.11, 0.13),
    "台北": (0.88, 0.31),
    "高雄": (0.81, 0.13),
}

_REGION_ZONES: Mapping[str, tuple[tuple[float, float, float, float], ...]] = {
    "中原": ((0.04, 0.47, 0.47, 0.95),),
    "东北": ((0.49, 0.58, 0.96, 0.95),),
    "南方": ((0.25, 0.04, 0.74, 0.46),),
    "海岛": ((0.04, 0.04, 0.23, 0.28), (0.76, 0.04, 0.96, 0.41)),
}

_REGION_LABELS: Mapping[str, tuple[tuple[float, float], ...]] = {
    "中原": ((0.055, 0.49),),
    "东北": ((0.51, 0.91),),
    "南方": ((0.27, 0.43),),
    "海岛": ((0.055, 0.25), (0.78, 0.38)),
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
    reachable_routes: Mapping[str, tuple[ReachableRoute, ...]],
) -> Mapping[str, tuple[float, float]]:
    """绘制以当前位置为中心、不会随选择变化的可达调度图。"""

    focus_city = selected_city or state.player.location
    positions = city_positions(catalog, bounds)
    _draw_dispatch_surface(bounds)
    _draw_region_zones(catalog, bounds)
    _draw_reachable_routes(reachable_routes, positions, state.player.location)
    for city in catalog.cities.values():
        _draw_station(
            city,
            *positions[city.name],
            is_current=city.name == state.player.location,
            is_selected=city.name == focus_city,
            is_reachable=city.name == state.player.location or city.name in reachable_routes,
        )
    _draw_header(bounds, state.day, state.player.location, focus_city)
    _draw_route_strip(catalog, bounds, state.player.location, focus_city, reachable_routes)
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


def _draw_region_zones(catalog: Catalog, bounds: Rect) -> None:
    """用稳定分区标出每座城市所属的市场区域。"""

    catalog_regions = {city.region for city in catalog.cities.values()}
    if catalog_regions != set(_REGION_ZONES) or catalog_regions != set(_REGION_LABELS):
        raise ValueError("城市区域与调度图分区不一致")
    for zones in _REGION_ZONES.values():
        for left, bottom, right, top in zones:
            rect = Rect(
                bounds.left + bounds.width * left,
                bounds.bottom + bounds.height * bottom,
                bounds.left + bounds.width * right,
                bounds.bottom + bounds.height * top,
            )
            arcade.draw_lrbt_rectangle_filled(rect.left, rect.right, rect.bottom, rect.top, MAP_REGION_SURFACE)
            arcade.draw_lrbt_rectangle_outline(rect.left, rect.right, rect.bottom, rect.top, MAP_REGION_BORDER, 1)
    for region, labels in _REGION_LABELS.items():
        for x, y in labels:
            arcade.draw_text(
                region,
                bounds.left + bounds.width * x,
                bounds.bottom + bounds.height * y,
                MAP_REGION_LABEL,
                9,
                font_name="Microsoft YaHei UI",
                bold=True,
                anchor_y="bottom",
            )


def _draw_reachable_routes(
    routes: Mapping[str, tuple[ReachableRoute, ...]],
    positions: Mapping[str, tuple[float, float]],
    current_city: str,
) -> None:
    """始终从当前位置画出全部可达目的地，不让选择动作改变路网。"""

    for destination, options in routes.items():
        for option in options:
            start, end = _parallel_route_endpoints(
                positions[current_city],
                positions[destination],
                option.mode,
                options,
            )
            if option.mode is TransportMode.LAND:
                _draw_land_route(start, end)
            else:
                _draw_sea_route(start, end)


def _parallel_route_endpoints(
    start: tuple[float, float],
    end: tuple[float, float],
    mode: TransportMode,
    options: tuple[ReachableRoute, ...],
) -> tuple[tuple[float, float], tuple[float, float]]:
    """为同一对站点的陆海双路线留出可辨识的平行间距。"""

    if len(options) == 1:
        return start, end
    length = hypot(end[0] - start[0], end[1] - start[1])
    offset = 3 if mode is TransportMode.LAND else -3
    offset_x = -(end[1] - start[1]) / length * offset
    offset_y = (end[0] - start[0]) / length * offset
    return (
        (start[0] + offset_x, start[1] + offset_y),
        (end[0] + offset_x, end[1] + offset_y),
    )


def _draw_land_route(start: tuple[float, float], end: tuple[float, float]) -> None:
    """以简洁实线表示从本站可完成的陆运行程。"""

    arcade.draw_line(*start, *end, MAP_TRACK_SHADOW, 4)
    arcade.draw_line(*start, *end, LAND_ROUTE, 1)


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
    is_reachable: bool,
) -> None:
    """按城市能力绘制货运站、高消费商圈与当前货车。"""

    if is_selected:
        _draw_selection_brackets(x, y)
    if city.is_high_consumption:
        _draw_market_station(x, y, is_reachable=is_reachable)
    else:
        _draw_standard_station(x, y, is_reachable=is_reachable)
    if city.has_port and is_reachable:
        arcade.draw_line(x - 6, y - 9, x + 6, y - 9, SEA_ROUTE, 2)
    if city.has_bank and is_reachable:
        arcade.draw_circle_filled(x, y + 9, 2, LAND_ROUTE)
    if is_current:
        _draw_truck_marker(x, y)
    _draw_city_label(city.name, x, y, is_reachable=is_reachable)


def _draw_standard_station(x: float, y: float, *, is_reachable: bool) -> None:
    arcade.draw_lrbt_rectangle_filled(x - 7, x + 7, y - 7, y + 7, MAP_STATION_SHADOW)
    arcade.draw_lrbt_rectangle_filled(x - 5, x + 5, y - 5, y + 5, NORMAL_CITY if is_reachable else MAP_TICK)
    arcade.draw_line(x - 3, y, x + 3, y, MAP_STATION_SHADOW, 1)


def _draw_market_station(x: float, y: float, *, is_reachable: bool) -> None:
    arcade.draw_polygon_filled(
        ((x, y + 8), (x + 8, y), (x, y - 8), (x - 8, y)),
        MAP_STATION_SHADOW,
    )
    arcade.draw_polygon_filled(
        ((x, y + 6), (x + 6, y), (x, y - 6), (x - 6, y)),
        HIGH_CONSUMPTION_CITY if is_reachable else MAP_TICK,
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


def _draw_city_label(name: str, x: float, y: float, *, is_reachable: bool) -> None:
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
        TEXT_LIGHT if is_reachable else MAP_REGION_LABEL,
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


def _draw_route_strip(
    catalog: Catalog,
    bounds: Rect,
    current_city: str,
    focus_city: str,
    reachable_routes: Mapping[str, tuple[ReachableRoute, ...]],
) -> None:
    line_y = bounds.bottom + 31
    arcade.draw_line(bounds.left + 12, line_y, bounds.right - 12, line_y, MAP_TICK, 1)
    routes = reachable_routes.get(focus_city, ())
    if current_city == focus_city:
        summary = f"{current_city}站  |  {catalog.city(current_city).region}  |  可达城市 {len(reachable_routes)}"
    else:
        route_text = "  /  ".join(
            f"{'陆运' if route.mode is TransportMode.LAND else '海运'} {route.distance_km:,} km"
            for route in routes
        )
        summary = f"已选 {focus_city}  |  {catalog.city(focus_city).region}  |  {route_text}"
    arcade.draw_text(
        summary,
        bounds.left + 14,
        bounds.bottom + 11,
        TEXT_LIGHT,
        11,
        font_name="Microsoft YaHei UI",
        anchor_y="bottom",
    )
