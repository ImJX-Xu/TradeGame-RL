from __future__ import annotations

import csv
import heapq
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from .data import CITIES, TransportMode
from .game_config import LAND_SPEED_KM_PER_DAY, SEA_SPEED_KM_PER_DAY


@dataclass(frozen=True, slots=True)
class Edge:
    to_city: str
    km: int


Graph = Dict[str, List[Edge]]


def _add_undirected(g: Graph, a: str, b: str, km: int) -> None:
    g.setdefault(a, []).append(Edge(b, km))
    g.setdefault(b, []).append(Edge(a, km))


def _build_default_land_graph() -> Graph:
    """
    里程表（MVP 版）：用“近似真实”的干线距离，保证可玩闭环。
    后续你想更真实，我们可以把它换成 CSV 或更细的地图数据。
    """
    g: Graph = {}
    # 中原枢纽
    _add_undirected(g, "郑州", "石家庄", 420)
    _add_undirected(g, "郑州", "太原", 420)
    _add_undirected(g, "郑州", "北京", 690)
    _add_undirected(g, "郑州", "沈阳", 1100)
    _add_undirected(g, "郑州", "长春", 1400)
    _add_undirected(g, "郑州", "哈尔滨", 1700)
    _add_undirected(g, "郑州", "福州", 1400)
    _add_undirected(g, "郑州", "广州", 1600)
    _add_undirected(g, "郑州", "深圳", 1650)
    _add_undirected(g, "郑州", "上海", 900)

    # 华北/东北干线
    _add_undirected(g, "北京", "石家庄", 280)
    _add_undirected(g, "北京", "太原", 500)
    _add_undirected(g, "北京", "沈阳", 700)
    _add_undirected(g, "石家庄", "太原", 200)
    _add_undirected(g, "沈阳", "长春", 300)
    _add_undirected(g, "长春", "哈尔滨", 250)

    # 华东/华南干线
    _add_undirected(g, "上海", "福州", 820)
    _add_undirected(g, "福州", "广州", 850)
    _add_undirected(g, "广州", "深圳", 140)
    _add_undirected(g, "上海", "广州", 1500)

    return g


def _build_default_sea_graph() -> Graph:
    """
    默认海运图（向后兼容，当CSV不存在时使用）。
    """
    g: Graph = {}
    # 大陆港口
    # 上海/福州/广州/深圳（简化）
    _add_undirected(g, "上海", "福州", 700)
    _add_undirected(g, "福州", "广州", 800)
    _add_undirected(g, "广州", "深圳", 150)

    # 海南
    _add_undirected(g, "广州", "海南", 650)
    _add_undirected(g, "深圳", "海南", 670)

    # 台湾
    _add_undirected(g, "福州", "台北", 300)
    _add_undirected(g, "福州", "高雄", 430)
    _add_undirected(g, "台北", "高雄", 360)
    _add_undirected(g, "上海", "台北", 800)

    return g


def _load_routes_from_csv() -> tuple[Graph, Graph]:
    """
    从CSV文件加载路由数据。
    CSV格式：from_city,to_city,mode,distance_km
    mode: land 或 sea
    """
    import sys
    if getattr(sys, 'frozen', False):
        # PyInstaller 打包后：数据文件在临时目录
        BASE_DIR = Path(sys._MEIPASS) / "trade_game"
    else:
        # 开发模式：使用当前文件所在目录
        BASE_DIR = Path(__file__).resolve().parent
    ROUTES_CSV = BASE_DIR / "routes.csv"
    
    land_graph: Graph = {}
    sea_graph: Graph = {}
    
    # 如果CSV不存在，使用默认硬编码数据（向后兼容）
    if not ROUTES_CSV.exists():
        return _build_default_land_graph(), _build_default_sea_graph()
    
    try:
        with ROUTES_CSV.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                from_city = row["from_city"].strip()
                to_city = row["to_city"].strip()
                mode_str = row["mode"].strip().lower()
                try:
                    distance_km = int(row["distance_km"].strip())
                except (ValueError, KeyError):
                    continue
                
                if mode_str == "land":
                    _add_undirected(land_graph, from_city, to_city, distance_km)
                elif mode_str == "sea":
                    _add_undirected(sea_graph, from_city, to_city, distance_km)
    except Exception as e:
        # 如果加载失败，使用默认数据
        return _build_default_land_graph(), _build_default_sea_graph()
    
    return land_graph, sea_graph


# 从CSV加载路由数据
LAND_GRAPH, SEA_GRAPH = _load_routes_from_csv()


class RouteNotFound(Exception):
    pass


def shortest_distance_km(graph: Graph, start: str, goal: str) -> int:
    if start == goal:
        return 0
    if start not in graph or goal not in graph:
        raise RouteNotFound(f"no route: {start} -> {goal}")

    pq: List[Tuple[int, str]] = [(0, start)]
    dist: Dict[str, int] = {start: 0}
    while pq:
        d, u = heapq.heappop(pq)
        if u == goal:
            return d
        if d != dist.get(u, 10**18):
            continue
        for e in graph.get(u, []):
            nd = d + e.km
            if nd < dist.get(e.to_city, 10**18):
                dist[e.to_city] = nd
                heapq.heappush(pq, (nd, e.to_city))
    raise RouteNotFound(f"no route: {start} -> {goal}")


def base_travel_days(mode: TransportMode, km: int) -> int:
    """
    基于里程数计算基础天数，向上取整。
    - 陆运：每天约300公里
    - 海运：每天约250公里（海运较慢）
    """
    if mode == TransportMode.LAND:
        days = km / LAND_SPEED_KM_PER_DAY
    else:
        days = km / SEA_SPEED_KM_PER_DAY
    
    # 向上取整，最少1天
    return max(1, int(math.ceil(days)))


def sample_travel_days(mode: TransportMode, km: int, rng: random.Random) -> int:
    """
    实际耗时 = 基础耗时 * (1 ± μ)
    - 陆运 μ ~ N(0, 0.15)，上限 1.45 倍
    - 海运 μ ~ N(0, 0.20)，上限 1.60 倍
    最小 1 天；最终向上取整为“经过天数”。
    """
    base = base_travel_days(mode, km)
    if base <= 0:
        return 1
    if mode == TransportMode.LAND:
        mu = rng.gauss(0.0, 0.15)
        factor = 1.0 + mu
        factor = max(1.0 - 0.45, min(factor, 1.45))  # 大致范围
    else:
        mu = rng.gauss(0.0, 0.20)
        factor = 1.0 + mu
        factor = max(1.0 - 0.60, min(factor, 1.60))

    days = base * factor
    days = max(1.0, days)
    return int(math.ceil(days))


def route_km(mode: TransportMode, start: str, goal: str) -> int:
    if mode == TransportMode.LAND:
        return shortest_distance_km(LAND_GRAPH, start, goal)
    return shortest_distance_km(SEA_GRAPH, start, goal)


def route_km_any(start: str, goal: str) -> int:
    """
    两城间最短里程（陆运与海运取可通行方式中的较小值）。
    若两种方式均不可达则抛出 RouteNotFound。
    """
    if start == goal:
        return 0
    best: int | None = None
    for mode in (TransportMode.LAND, TransportMode.SEA):
        try:
            km = route_km(mode, start, goal)
            if best is None or km < best:
                best = km
        except RouteNotFound:
            continue
    if best is None:
        raise RouteNotFound(f"no route: {start} -> {goal}")
    return best


def get_route_km_range() -> Tuple[int, int]:
    """
    全图路线里程范围（仅不同城市对）：(最小里程, 最大里程)。
    用于异地售卖乘数在 [1, 2] 内按里程线性插值。
    """
    cities = list(CITIES.keys())
    distances: List[int] = []
    for i, a in enumerate(cities):
        for b in cities[i + 1 :]:
            try:
                distances.append(route_km_any(a, b))
            except RouteNotFound:
                pass
    if not distances:
        return (0, 0)
    return (min(distances), max(distances))


def validate_mode_allowed(mode: TransportMode, start: str, goal: str) -> None:
    if start not in CITIES or goal not in CITIES:
        raise RouteNotFound("unknown city")
    if mode not in CITIES[start].modes:
        raise RouteNotFound(f"{start} 不支持 {mode.value}")
    if mode not in CITIES[goal].modes:
        raise RouteNotFound(f"{goal} 不支持 {mode.value}")

