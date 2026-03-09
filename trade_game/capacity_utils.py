from __future__ import annotations

"""
总载重：仅保留由货车数量决定的货车总载重，与城市无关。
海运不单独提供载重；出海时费用乘数基于总载重计算（见 game_config / 出海费用）。
"""
from .data import CITIES, TransportMode
from .game_config import INITIAL_TRUCK_CAPACITY
from .inventory import cargo_used
from .state import Player


def is_sea_port(city: str) -> bool:
    """有海运能力的城市（大陆海港或海岛）。"""
    return TransportMode.SEA in CITIES[city].modes


def is_island_city(city: str) -> bool:
    """海岛/海运城市：不具备陆运能力（没有货车可用）。"""
    c = CITIES[city]
    return TransportMode.LAND not in c.modes


def effective_truck_capacity(player: Player, city: str) -> int:
    """
    当前城市可用的货车容量。海岛城市为 0（不能陆运，但总载重仍为货车载重，见 total_storage_capacity）。
    """
    if is_island_city(city):
        return 0
    return int(getattr(player, "truck_total_capacity", INITIAL_TRUCK_CAPACITY))


def total_storage_capacity(player: Player, city: str = "") -> int:
    """
    总载重：仅由货车数量决定（truck_total_capacity），与城市无关。
    海运不单独提供载重；各地显示的总载重一致。
    """
    return int(getattr(player, "truck_total_capacity", INITIAL_TRUCK_CAPACITY))


def current_cargo_units(player: Player) -> int:
    return cargo_used(player.cargo_lots)

