from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from .game_config import INITIAL_CASH, INITIAL_TRUCK_CAPACITY
from .inventory import CargoLot
from .loans import Loan


@dataclass(slots=True)
class Player:
    cash: float = INITIAL_CASH
    location: str = "郑州"
    day: int = 1
    # 车辆数量：每新增 1 辆车，总载重 +TRUCK_CAPACITY_PER_VEHICLE（见 game_config / 车厂）
    truck_count: int = 1
    # 初始货车总载重（单位），与 game_config.INITIAL_TRUCK_CAPACITY 一致
    truck_total_capacity: int = INITIAL_TRUCK_CAPACITY
    truck_durability: float = 100.0  # 0~100
    # 出海机制：在海港可“出海”往返海岛，无租船合同；单次出海载重按 ship_capacity_each 计
    ship_capacity_each: int = 200
    # 当玩家在海岛时，记录从大陆哪个海港出海的，返程只能回该港
    sea_departure_port: str = ""

    # 无仓库：货物完全依赖运输工具存储（先做“货物批次”模型，后续接入货车/货船拆分存放）
    cargo_lots: List[CargoLot] = field(default_factory=list)


@dataclass(slots=True)
class GameState:
    player: Player = field(default_factory=Player)

    # 游戏模式："challenge" 挑战模式（365天上限），"free" 自由模式（无时间限制）
    game_mode: str = "free"

    # 商品 id -> 当日 λ（价格波动值）
    daily_lambdas: Dict[str, float] = field(default_factory=dict)
    
    # 商品 id -> 昨日 λ（用于带惯性的波动计算）
    previous_lambdas: Dict[str, float] = field(default_factory=dict)

    # 商品 id@城市 -> 最近 7 天采购价序列
    price_history_buy_7d: Dict[str, List[float]] = field(default_factory=dict)
    # 商品 id@城市 -> 最近 7 天售卖价序列
    price_history_sell_7d: Dict[str, List[float]] = field(default_factory=dict)

    loans: List[Loan] = field(default_factory=list)

    # 商品累计运输损耗：product_id -> 已损数量（用于 UI 显示和利润估算）
    loss_by_product: Dict[str, int] = field(default_factory=dict)
