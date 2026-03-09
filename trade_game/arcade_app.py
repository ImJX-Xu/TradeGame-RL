"""
风物千程 — 唯一 Arcade 图形前端。

本模块为项目的图形界面实现，由 start_game.py 或 python -m trade_game.arcade_app 启动。
"""
from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import arcade

from trade_game.data import CITIES, HIGH_CONSUMPTION_CITIES, PRODUCTS, product_display_name
from trade_game.economy import can_sell_product_here, purchase_price, refresh_daily_lambdas, sell_unit_price
from trade_game.inventory import (
    CargoLot,
    add_lot,
    apply_transport_loss,
    cargo_used,
    expected_perishable_loss_details,
    expected_transport_loss_display,
)
from trade_game.loans import (
    Bankruptcy,
    borrow,
    estimated_assets,
    repay,
    total_outstanding_principal,
)
from trade_game.game_config import (
    DAILY_LABOR_PER_TRUCK,
    FAST_TRAVEL_COST_MULTIPLIER,
    FAST_TRAVEL_MIN_DAYS,
    FAST_TRAVEL_TIME_DIVISOR,
    INITIAL_CASH,
    LAND_COST_PER_KM,
    LOAN_DAILY_INTEREST_RATE,
    SEA_COST_PER_KM,
    TAIWAN_CUSTOMS,
    TRUCK_CAPACITY_PER_VEHICLE,
    TRUCK_DURABILITY_LOSS_PER_KM,
    TRUCK_MIN_DURABILITY_FOR_TRAVEL,
    TRUCK_PURCHASE_PRICE,
    TRUCK_REPAIR_COST_BASE,
    TRUCK_REPAIR_DAYS,
)
from trade_game.train_config import (
    AMOUNT_FRACTIONS,
    CHALLENGE_DAYS,
    DEMO_MODE_DAYS,
    compute_challenge_rating,
    compute_settlement_amount,
    get_max_days,
)
from trade_game.save_load import delete_game, load_game, save_game
from trade_game.timeflow import advance_one_day, DAILY_LABOR_PER_TRUCK
from trade_game.human_demo import HumanDemoRecorder, default_demo_path
from trade_game import api as api_actions
from trade_game.transport import (
    RouteNotFound,
    SEA_GRAPH,
    TransportMode,
    route_km,
    sample_travel_days,
    validate_mode_allowed,
)
from trade_game.state import GameState
from trade_game.capacity_utils import (
    current_cargo_units,
    total_storage_capacity,
)


SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
SCREEN_TITLE = "风物千程"


def _truck_damage_factors(durability: float) -> Tuple[float, float, float]:
    """
    根据当前货车耐久度，返回 (damage_ratio, time_factor, loss_factor)。
    - damage_ratio: 损坏比例，0~1（耐久 100% 为 0；耐久 30% 为 0.7）
    - time_factor: 运输时间倍率（最高约 +35%）
    - loss_factor: 货损率倍率（最高约 +70%）
    """
    durability = max(0.0, min(100.0, float(durability)))
    damage_ratio = (100.0 - durability) / 100.0
    # 出发门槛是 <=30% 禁止出车，因此 damage_ratio 实际上不会达到 1
    time_factor = 1.0 + 0.5 * damage_ratio
    loss_factor = 1.0 + damage_ratio
    return damage_ratio, time_factor, loss_factor

# --- 升级版布局神器：支持嵌套的 UI 盒子 ---
class UIBox:
    def __init__(self, x, y, width, height):
        """
        定义一个矩形区域
        :param x, y: 中心点坐标 (Center)
        """
        self.width = width
        self.height = height
        # 计算边界 (Left, Bottom, Right, Top)
        self.l = x - width / 2
        self.b = y - height / 2
        self.r = x + width / 2
        self.t = y + height / 2

    # --- 1. 快速获取坐标点 ---
    @property
    def center(self): return (self.l + self.width/2, self.b + self.height/2)
    @property
    def top_left(self): return (self.l, self.t)
    @property
    def bottom_left(self): return (self.l, self.b)
    
    # --- 2. 区域切分工具 (用于布局) ---
    
    def top_slice(self, height):
        """从顶部切下一块高度为 height 的区域"""
        new_y = self.t - height/2
        return UIBox(self.l + self.width/2, new_y, self.width, height)
    
    def bottom_slice(self, height):
        """从底部切下一块"""
        new_y = self.b + height/2
        return UIBox(self.l + self.width/2, new_y, self.width, height)

    def left_slice(self, width):
        """从左边切下一块"""
        new_x = self.l + width/2
        return UIBox(new_x, self.b + self.height/2, width, self.height)
        
    def right_slice(self, width):
        """从右边切下一块"""
        new_x = self.r - width/2
        return UIBox(new_x, self.b + self.height/2, width, self.height)

    def split_horizontal(self, ratio=0.5):
        """左右分栏 (ratio=0.5 表示对半分)"""
        w1 = self.width * ratio
        w2 = self.width - w1
        left = UIBox(self.l + w1/2, self.b + self.height/2, w1, self.height)
        right = UIBox(self.r - w2/2, self.b + self.height/2, w2, self.height)
        return left, right

    def split_vertical(self, top_height):
        """
        上下分栏：返回 (top_box, bottom_box)
        :param top_height: 上方区域的高度
        """
        # 上方区域：从当前 Box 顶部切下 top_height
        top = self.top_slice(top_height)
        # 下方区域：使用剩余高度
        bottom_height = max(0, self.height - top_height)
        center_x = self.l + self.width / 2
        bottom_center_y = self.b + bottom_height / 2
        bottom = UIBox(center_x, bottom_center_y, self.width, bottom_height)
        return top, bottom

    def pad(self, padding):
        """返回一个向内缩进了 padding 的新区域 (防止内容贴边)"""
        return UIBox(self.l + self.width/2, self.b + self.height/2, 
                     self.width - padding*2, self.height - padding*2)

    # --- 3. 在本区域内创建网格 (Grid) ---
    
    def make_grid(self, rows, cols, gap=10):
        """在这个Box内部创建一个网格，用于放按钮"""
        # 这里复用之前的逻辑，但绑定在Box内部
        return _MiniGrid(self, rows, cols, gap)

# 辅助类：在Box内部的网格
class _MiniGrid:
    def __init__(self, parent_box, rows, cols, gap):
        self.box = parent_box
        self.rows = rows
        self.cols = cols
        self.gap = gap
        # 计算格子大小
        inner_w = parent_box.width - gap * (cols + 1)
        inner_h = parent_box.height - gap * (rows + 1)
        self.cell_w = inner_w / cols
        self.cell_h = inner_h / rows
        
    def pos(self, row, col):
        """获取格子坐标 (row 0 为顶部)"""
        actual_row = (self.rows - 1) - row
        x = self.box.l + self.gap + col * (self.cell_w + self.gap) + self.cell_w/2
        y = self.box.b + self.gap + actual_row * (self.cell_h + self.gap) + self.cell_h/2
        return x, y
        
    def size(self, scale=0.9):
        return self.cell_w * scale, self.cell_h * scale

# --- 兼容旧代码的UIGrid类 ---
class UIGrid:
    def __init__(self, left, bottom, width, height, rows, cols, padding=10):
        """
        定义一个隐形表格区域
        :param left, bottom: 表格区域的左下角坐标
        :param width, height: 表格区域的总大小
        :param rows, cols: 你要把这个区域分成几行几列
        :param padding: 格子与格子之间的空隙
        """
        # 转换为UIBox的中心点坐标
        center_x = left + width / 2
        center_y = bottom + height / 2
        self.box = UIBox(center_x, center_y, width, height)
        self.left = self.box.l
        self.bottom = self.box.b
        self.width = self.box.width
        self.height = self.box.height
        self.rows = rows
        self.cols = cols
        self.padding = padding
        
        # 计算单个格子大小
        inner_w = width - padding * (cols + 1)
        inner_h = height - padding * (rows + 1)
        self.cell_w = inner_w / cols if cols > 0 else width
        self.cell_h = inner_h / rows if rows > 0 else height

    def pos(self, row, col, colspan=1):
        """
        获取指定格子的中心点坐标 (x, y)
        :param row: 第几行 (0为最上面一行)
        :param col: 第几列 (0为最左边一列)
        """
        # 自动处理Y轴翻转，让row=0在视觉上方
        actual_row = (self.rows - 1) - row
        
        x = self.box.l + self.padding + col * (self.cell_w + self.padding) + (self.cell_w * colspan) / 2
        y = self.box.b + self.padding + actual_row * (self.cell_h + self.padding) + self.cell_h / 2
        return x, y

    def size(self, scale_w=0.8, scale_h=0.8):
        """获取建议的按钮大小 (按比例缩放，防止贴边)"""
        return self.cell_w * scale_w, self.cell_h * scale_h

# Win98风格颜色常量
WIN98_BG_DARK = (16, 16, 20)  # #101014
WIN98_BG_LIGHT = (28, 32, 40)  # #1C2028
WIN98_BUTTON_FACE = (192, 192, 192)  # #C0C0C0
WIN98_BUTTON_HIGHLIGHT = (255, 255, 255)  # #FFFFFF
WIN98_BUTTON_SHADOW = (64, 64, 64)  # #404040
WIN98_BUTTON_DARK = (128, 128, 128)  # #808080
WIN98_TITLE_BAR = (0, 0, 128)  # #000080
WIN98_CASH_HIGHLIGHT = (255, 211, 122)  # #FFD37A
WIN98_TEXT_LIGHT = (224, 224, 224)  # #E0E0E0
WIN98_TEXT_DARK = (32, 32, 32)  # #202020

# --- 城市分类常量：集中维护，避免多处硬编码不一致 ---
# 说明：
# - LAND_CITIES：仅陆运城市
# - LAND_SEA_CITIES：陆+海运城市
# - SEA_CITIES：仅海运城市（海岛）
# - PORT_CITIES：所有有港口的城市（用于租船等）
# - TRAVEL_REGIONS：仅用于 UI 分组展示
LAND_CITIES = {
    "郑州",
    "石家庄",
    "太原",
    "沈阳",
    "长春",
    "哈尔滨",
    "北京",
}

LAND_SEA_CITIES = {
    "广州",
    "深圳",
    "福州",
    "上海",
}

SEA_CITIES = {
    "海南",
    "台北",
    "高雄",
}

# 有港口的城市（大陆海港 + 海岛）：仅在这些城市可「出海」
PORT_CITIES = LAND_SEA_CITIES | SEA_CITIES

PORT_CITIES = {
    "广州",
    "深圳",
    "福州",
    "上海",
    "海南",
    "台北",
    "高雄",
}

TRAVEL_REGIONS: Dict[str, List[str]] = {
    "中原地区": ["郑州", "石家庄", "太原", "北京"],
    "东北地区": ["沈阳", "长春", "哈尔滨"],
    "南方地区": ["广州", "深圳", "福州", "上海"],
    "海岛地区": ["海南", "台北", "高雄"],
}

# 各功能窗口的帮助说明（面向玩家，笼统介绍）
_travel_help = (
    "选择目的地城市进行运输。\n\n陆运：需有车辆，大陆城市间通行。\n海运：在海港使用「出海」往返海岛。\n\n"
    "左侧选择目的地，右侧可查看预计耗时和费用。\n\n"
    "快速出行费用更高但耗时更短，适合生鲜等有时效性的商品。\n\n"
    "注意：出行会消耗时间，游戏将自动度过运输所需的天数。"
    "\n\n【城市特性】\n"
    "高消费城市：商品售价高于其他城市。\n"
    "台湾：航线按趟收费，采购按价收税。\n"
    "异地售价：商品在距离原产地越远的地区售价越高。"
)
DIALOG_HELP_TEXTS: Dict[str, str] = {
    "travel": _travel_help,
    "market": "采购和售卖商品。\n\n采购：仅能在商品产地采购。\n售卖：异地售卖才有利润，本地售卖仅能回收成本。\n\n商品标签说明：\n【城】城市特产，仅在特定城市可采购\n【区】区域特产，同区域内多个城市可采购\n【民】【轻】普通类商品，价格较稳定\n【损】易损商品，运输损耗较大\n【鲜】生鲜特产，有保质期，运输损耗与时间相关\n\n价格每日波动，可关注低价采购、高价售出。\n\n注意：每次采购或售出操作会消耗一天时间。",
    "loans": "在银行借贷，扩大经营资金。\n\n可借额度与您的资产相关，按日计息。\n可随时还款，无固定期限。\n\n注意：仅大陆城市有银行，海岛城市无法借贷或还款。",
    "repair": "车厂：查看车辆状态、维修货车、购买新车。\n\n耐久度会随运输里程自然损耗。\n耐久度过低时需维修才能继续出行。\n\n每增加一辆车，每天会有固定人力成本。\n\n注意：维修货车会消耗多天时间；购买新车不消耗时间。",
    "sail": "出海：在海港选择海岛目的地，或在海岛返回出发港。\n\n大陆海港仅显示海岛目的地；海岛仅显示「返回出发港」一条航线。\n费用与运输损耗按里程计算，与陆运规则一致。",
    "save": "将当前游戏进度保存到选定的存档槽位。\n\n若该槽位已有存档，覆盖前会提示确认。\n\n存档不消耗游戏时间。",
    "load": "从选定的存档槽位读取游戏进度。\n\n会覆盖当前进度，请确认后再操作。\n\n读档不消耗游戏时间。",
}


class TradeGameWindow(arcade.Window):
    def __init__(self, *, demo_autorecord: bool = False) -> None:
        super().__init__(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE, resizable=False)

        self.background = None
        # 支持打包后的资源路径
        import sys
        if getattr(sys, 'frozen', False):
            # PyInstaller 打包后：资源文件在临时目录
            assets_dir = Path(sys._MEIPASS) / "trade_game" / "assets"
        else:
            # 开发模式：使用当前文件所在目录
            assets_dir = Path(__file__).resolve().parent / "assets"
        china_map = assets_dir / "china_map.png"
        if china_map.exists():
            self.background = arcade.load_texture(china_map)

        self.rng = random.Random()
        self.state = GameState()
        # 开局Day1：所有商品λ=0（previous_lambdas=None）
        self.state.daily_lambdas = refresh_daily_lambdas(self.rng, None)
        self.log: List[str] = []
        # 界面状态："mode_select" 模式选择 | "playing" 游戏中 | "challenge_end" 挑战结算
        # 玩家演示模式：直接进入自由模式游戏中；正常启动：从模式选择开始。
        self.current_screen: str = "playing" if demo_autorecord else "mode_select"
        # 挑战结束数据：{"days": int, "total_assets": float, "rating": str, "bankrupt": bool}
        self.challenge_end_data: Optional[Dict[str, Any]] = None
        # 按钮 & 城市行可点击区域
        self.button_regions: Dict[str, Tuple[float, float, float, float]] = {}
        # 允许同一城市在多个分组中重复出现（例如“高消费经济区”与其他地区重叠）
        self.city_row_regions: List[Tuple[str, Tuple[float, float, float, float]]] = []

        # 简化海运租船：仅记录“是否已租船”，按里程计费一次
        self.has_sea_rental: bool = False
        self.land_cost_per_km: float = LAND_COST_PER_KM
        self.sea_cost_per_km: float = SEA_COST_PER_KM

        # 弹窗系统
        self.active_dialog: Optional[str] = None  # 'travel', 'market', 'loans', 'repair', 'sail', 'save', 'load'
        self.dialog_data: Dict[str, Any] = {}  # 存储各弹窗的状态数据
        self.help_popup_text: Optional[str] = None  # 帮助弹窗内容，非空时显示

        # 自由模式开局金额设置（仅在模式选择界面使用）
        self.start_cash_text: str = str(int(INITIAL_CASH))
        self.start_cash_focused: bool = True
        
        # 市场面板数据
        self.market_index: int = 0
        self.market_qty: int = 1
        self.market_is_buy: bool = True
        self.market_tab: str = "buy"  # 'buy' or 'sell'
        # 市场小弹窗：输入具体采购/售出数量
        # 结构：{"mode": "buy"|"sell", "pid": str, "qty": int, "text": str}
        self.market_order_dialog: Optional[Dict[str, Any]] = None
        self.market_order_focused: bool = False
        
        # 借贷面板数据（已统一为简单利率模式，不再区分 daily/lump）
        self.loans_amount: float = 0.0
        self.loans_amount_text: str = "0"  # 借贷金额输入框内容，默认显示 0
        # 右侧：还款金额
        self.loans_repay_amount: float = 0.0
        self.loans_repay_amount_text: str = "0"
        
        # 价格分析文本（在价格信息弹窗中显示）
        self.price_info_lines: List[str] = []
        self.price_info_scroll: int = 0
        # 笔记框中的价格建议（连续降价/涨价等）
        self.price_note_lines: List[str] = []
        # 价格信息滚动条拖动状态
        self.price_info_dragging: bool = False
        # (track_x, track_y, track_h, handle_h, max_visible, total_lines)
        self.price_info_scrollbar_meta: Optional[Tuple[float, float, float, float, int, int]] = None
        self.loans_focused_input: Optional[str] = None  # None | "borrow" | "repay" 两个输入框完全独立
        self.loans_amount_hint: str = ""  # 如 "已自动调整为可借上限 xxx 元"
        
        # 出行面板数据
        self.travel_target: Optional[str] = None
        self.travel_region_expanded: Dict[str, bool] = {}  # 区域折叠状态
        self.sail_target: Optional[str] = None  # 出海目的地
        self.sail_row_regions: List[Tuple[str, Tuple[float, float, float, float]]] = []  # (city, (x1,y1,x2,y2))
        
        # 车厂购车数量输入（与市场/借贷类似的输入框）
        self.factory_buy_qty: int = 1
        self.factory_buy_text: str = "1"
        self.factory_buy_focused: bool = False

        # 存档面板数据
        self.save_slot_selected: Optional[int] = None
        self.load_slot_selected: Optional[int] = None
        
        # 鼠标hover状态
        self.hover_button: Optional[str] = None
        self.hover_city: Optional[str] = None

        # ---- 人类玩家轨迹录制（用于 PPO warmstart）----
        # F8 开始/停止录制；停止时保存到 runs/demos/*.npz
        self.demo_recording: bool = False
        self.demo_recorder: Optional[HumanDemoRecorder] = None
        self._demo_autorecord: bool = bool(demo_autorecord)
        self._demo_saved_path: Optional[Path] = None

        if self._demo_autorecord:
            # 对齐 CLI 规则：直接开一局，初始现金为 INITIAL_CASH；演示模式 90 天上限。
            self.state = GameState()
            self.state.game_mode = "demo"
            self.state.player.cash = float(INITIAL_CASH)
            self.state.daily_lambdas = refresh_daily_lambdas(self.rng, None)
            self.current_screen = "playing"

            self.demo_recorder = HumanDemoRecorder()
            self.demo_recording = True
            self._log("【玩家演示模式】已自动开始录制轨迹；退出窗口将自动保存。")

    def _log(self, msg: str) -> None:
        self.log.append(msg)
        self.log = self.log[-12:]

    def on_close(self) -> None:  # type: ignore[override]
        # 自动保存 demo（玩家演示模式 / 手动录制均适用）
        if self.demo_recording and self.demo_recorder is not None:
            try:
                out = default_demo_path(prefix="human_demo")
                self.demo_recorder.save_npz(out)
                self._demo_saved_path = out
            except Exception:
                pass
        try:
            super().on_close()
        except Exception:
            # arcade 版本差异下，super().on_close 可能不存在
            try:
                arcade.close_window()
            except Exception:
                pass

    def _do_advance_day(self, *, skip_day_limit_check: bool = False) -> Optional[Tuple[List[str], bool]]:
        """
        推进一天。若有天数上限的模式达到上限或破产，返回 None，调用方应终止。
        否则返回 (msgs, False)。skip_day_limit_check=True 时（运输途中）不检查天数上限。
        """
        try:
            self.state, msgs = advance_one_day(self.state, self.rng)
            if not skip_day_limit_check:
                max_days = get_max_days(getattr(self.state, "game_mode", "free"))
                if max_days is not None and self.state.player.day > max_days:
                    p = self.state.player
                    settlement = compute_settlement_amount(
                        p.cash, p.cargo_lots,
                        max(1, int(getattr(p, "truck_count", 1))),
                        self.state.loans,
                    )
                    self.challenge_end_data = {
                        "days": max_days,
                        "total_assets": settlement,
                        "rating": compute_challenge_rating(settlement, bankrupt=False),
                        "bankrupt": False,
                    }
                    self.current_screen = "challenge_end"
                    return None
            return (msgs, False)
        except Bankruptcy as e:
            gm = getattr(self.state, "game_mode", "free")
            if gm in ("challenge", "demo"):
                max_days = get_max_days(gm)
                self.challenge_end_data = {
                    "days": max_days or self.state.player.day,
                    "total_assets": 0.0,
                    "rating": "拉完了",
                    "bankrupt": True,
                }
                self.current_screen = "challenge_end"
                return None
            self._log(f"判定破产：{e}")
            return None

    def _check_and_settle_if_day_limit(self) -> None:
        """运输到达后检查是否已超天数上限，若超则结算。"""
        max_days = get_max_days(getattr(self.state, "game_mode", "free"))
        if max_days is None:
            return
        if self.state.player.day <= max_days:
            return
        p = self.state.player
        settlement = compute_settlement_amount(
            p.cash, p.cargo_lots,
            max(1, int(getattr(p, "truck_count", 1))),
            self.state.loans,
        )
        self.challenge_end_data = {
            "days": max_days,
            "total_assets": settlement,
            "rating": compute_challenge_rating(settlement, bankrupt=False),
            "bankrupt": False,
        }
        self.current_screen = "challenge_end"

    # --- Win98风格UI绘制辅助函数 ---
    
    def _draw_win98_3d_border(
        self, x: float, y: float, width: float, height: float, raised: bool = True
    ) -> None:
        """绘制Win98风格的3D边框（凸起或凹陷）"""
        from arcade import draw_lbwh_rectangle_filled, draw_lbwh_rectangle_outline
        
        if raised:
            # 凸起：左上亮，右下暗
            # 外边框
            draw_lbwh_rectangle_outline(x, y, width, height, WIN98_BUTTON_HIGHLIGHT, 1)
            # 内边框（右下）
            draw_lbwh_rectangle_outline(x + 1, y + 1, width - 2, height - 2, WIN98_BUTTON_SHADOW, 1)
        else:
            # 凹陷：左上暗，右下亮
            # 外边框
            draw_lbwh_rectangle_outline(x, y, width, height, WIN98_BUTTON_SHADOW, 1)
            # 内边框（左上）
            draw_lbwh_rectangle_outline(x + 1, y + 1, width - 2, height - 2, WIN98_BUTTON_HIGHLIGHT, 1)
    
    def _draw_win98_button(
        self,
        name: str,
        x: float,
        y: float,
        width: float,
        height: float,
        label: str,
        enabled: bool = True,
        pressed: bool = False,
    ) -> None:
        """绘制Win98风格的立体按钮（x,y为按钮中心坐标）"""
        from arcade import draw_lbwh_rectangle_filled
        
        x1 = x - width / 2
        y1 = y - height / 2
        x2 = x + width / 2
        y2 = y + height / 2
        
        # 记录按钮区域（禁用按钮不注册，避免误触）
        if enabled:
            self.button_regions[name] = (x1, y1, x2, y2)
        
        if not enabled:
            # 禁用状态：灰色背景，文字变浅
            draw_lbwh_rectangle_filled(x1, y1, width, height, WIN98_BUTTON_DARK)
            self._draw_win98_3d_border(x1, y1, width, height, raised=False)
            arcade.draw_text(
                label, x1 + width / 2, y, WIN98_TEXT_DARK, 12, anchor_x="center", anchor_y="center"
            )
            return
        
        # 按下状态：边框反转
        if pressed or (self.hover_button == name):
            # 按下或hover：稍微变亮
            draw_lbwh_rectangle_filled(x1, y1, width, height, WIN98_BUTTON_FACE)
            self._draw_win98_3d_border(x1, y1, width, height, raised=not pressed)
        else:
            # 正常状态
            draw_lbwh_rectangle_filled(x1, y1, width, height, WIN98_BUTTON_FACE)
            self._draw_win98_3d_border(x1, y1, width, height, raised=True)
        
        # 文字（按下时稍微下移）
        text_y = y - (1 if pressed else 0)
        arcade.draw_text(
            label, x1 + width / 2, text_y, WIN98_TEXT_DARK, 12, anchor_x="center", anchor_y="center"
        )
    
    def _draw_progress_bar(
        self, x: float, y: float, width: float, height: float, value: float, max_value: float
    ) -> None:
        """绘制进度条（带Win98风格边框，x,y为进度条左下角坐标）"""
        from arcade import draw_lbwh_rectangle_filled
        
        # 背景（凹陷边框）
        self._draw_win98_3d_border(x, y, width, height, raised=False)
        draw_lbwh_rectangle_filled(x + 2, y + 2, width - 4, height - 4, WIN98_BG_DARK)
        
        # 填充
        if max_value > 0:
            fill_width = max(0, min(width - 4, (value / max_value) * (width - 4)))
            if fill_width > 0:
                # 根据百分比选择颜色
                percent = value / max_value
                if percent > 0.5:
                    fill_color = (76, 175, 80)  # 绿色
                elif percent > 0.3:
                    fill_color = (255, 193, 7)  # 橙黄色
                else:
                    fill_color = (229, 57, 53)  # 红色
                draw_lbwh_rectangle_filled(x + 2, y + 2, fill_width, height - 4, fill_color)

    # --- 逻辑辅助 ---

    def _reachable_cities(self) -> List[str]:
        """出行（陆运）可达城市：仅陆路，不含海运；海岛需通过「出海」前往。"""
        p = self.state.player
        start = p.location
        reachable: List[str] = []
        for name in CITIES.keys():
            if name == start:
                continue
            if name in LAND_CITIES or name in LAND_SEA_CITIES:
                try:
                    validate_mode_allowed(TransportMode.LAND, start, name)
                    route_km(TransportMode.LAND, start, name)
                    reachable.append(name)
                except RouteNotFound:
                    continue
        return sorted(reachable)

    def _cli_can_loan(self) -> bool:
        """录制模式下对齐 CLI：仅在有银行的城市可借贷/还款。"""
        return CITIES[self.state.player.location].has_bank

    def _cli_can_sail(self) -> bool:
        """录制模式下对齐 CLI：仅在港口城市可出海。"""
        return self.state.player.location in PORT_CITIES

    def _cli_reachable_land_cities(self) -> List[str]:
        """录制模式下对齐 CLI：陆运可达城市（耐久不足时返回空）。"""
        if not self.demo_recording:
            return self._reachable_cities()
        p = self.state.player
        if p.truck_durability <= TRUCK_MIN_DURABILITY_FOR_TRAVEL:
            return []
        return self._reachable_cities()

    def _sail_destinations(self) -> List[str]:
        """出海可选目的地：大陆海港时仅海岛；海岛时可到其他海岛海港 + 返回大陆出海港。"""
        p = self.state.player
        loc = p.location
        if loc in LAND_SEA_CITIES:
            return sorted(SEA_CITIES)
        if loc in SEA_CITIES:
            # 其他海岛 + 返程大陆港（若有）
            others = sorted(c for c in SEA_CITIES if c != loc)
            if p.sea_departure_port:
                return others + [p.sea_departure_port]
            return others
        return []

    def _estimate_trip_profit(self, target: str) -> Optional[float]:
        """估算一次完整行程的大致利润（仅供 UI 参考；出行仅陆运，海运走出海）。"""
        p = self.state.player
        from_city = p.location
        if target in SEA_CITIES:
            return None
        mode = TransportMode.LAND
        try:
            validate_mode_allowed(mode, from_city, target)
            km = route_km(mode, from_city, target)
        except RouteNotFound:
            return None

        if mode == TransportMode.LAND:
            truck_count = max(1, int(getattr(p, "truck_count", 1)))
            cost = km * self.land_cost_per_km * truck_count
        else:
            # 海运：基于里程数计算费用，大陆↔台湾有额外关税
            is_taiwan_route = (from_city in ("台北", "高雄")) ^ (target in ("台北", "高雄"))
            customs = TAIWAN_CUSTOMS if is_taiwan_route else 0.0
            cost = km * self.sea_cost_per_km + customs

        profit = 0.0
        for lot in p.cargo_lots:
            prod = PRODUCTS[lot.product_id]
            shelf = lot.shelf_life_remaining_days
            here = sell_unit_price(
                prod,
                from_city,
                self.state.daily_lambdas,
                quantity_sold=lot.quantity,
                shelf_life_remaining_days=shelf,
            )
            there = sell_unit_price(
                prod,
                target,
                self.state.daily_lambdas,
                quantity_sold=lot.quantity,
                shelf_life_remaining_days=shelf,
            )
            profit += (there - here) * lot.quantity

        return round(profit - cost, 2)

    def _preview_travel_plan(
        self, target: str, *, allow_unreachable: bool
    ) -> Optional[Tuple[TransportMode, float, int]]:
        """给 UI 预览用的“稳定版”路线（含模式/里程/耗时）。

        - allow_unreachable=False：仅陆运可达（海岛走出海，不在此返回）
        - allow_unreachable=True：仍仅尝试陆运
        """
        p = self.state.player
        start = p.location
        if target in SEA_CITIES:
            return None
        try:
            validate_mode_allowed(TransportMode.LAND, start, target)
            km = route_km(TransportMode.LAND, start, target)
        except RouteNotFound:
            return None
        seed = hash((TransportMode.LAND.value, start, target)) & 0xFFFFFFFF
        preview_rng = random.Random(seed)
        base_days = sample_travel_days(TransportMode.LAND, km, preview_rng)
        _damage_ratio, time_factor, _loss_factor = _truck_damage_factors(p.truck_durability)
        days = max(1, int(round(base_days * time_factor)))
        return TransportMode.LAND, float(km), int(days)

    def _preview_travel_days(self, target: str) -> Optional[Tuple[float, int]]:
        """给 UI 预览用的“稳定版”里程与耗时，避免每帧随机闪动。"""
        plan = self._preview_travel_plan(target, allow_unreachable=False)
        if plan is None:
            return None
        _mode, km, days = plan
        return km, days

    def _apply_additional_truck_damage_loss(
        self,
        lots: List[CargoLot],
        *,
        origin_city: str,
        target_city: str,
        damage_ratio: float,
        loss_stats: Dict[str, int] | None = None,
    ) -> int:
        """
        仅陆运使用：在基础运输损耗之外，根据货车损坏程度额外增加一部分损耗。
        海运不调用此函数，车辆损坏不影响海运。
        damage_ratio 取自 _truck_damage_factors，范围约 0~0.7。
        """
        if damage_ratio <= 0.0:
            return 0

        extra_lost = 0
        for lot in list(lots):
            if lot.quantity <= 0:
                continue
            prod = PRODUCTS[lot.product_id]
            base_rate = prod.transport_loss_rate
            if base_rate <= 0.0:
                continue

            # 额外损耗率：与基础损耗率成正比，最多额外 +70%
            extra_rate = base_rate * damage_ratio
            if origin_city in ("台北", "高雄") or target_city in ("台北", "高雄"):
                extra_rate += 0.01 * damage_ratio

            delta = int(lot.quantity * extra_rate)
            if delta <= 0:
                continue
            lot.quantity -= delta
            extra_lost += delta
            if loss_stats is not None:
                loss_stats[prod.id] = loss_stats.get(prod.id, 0) + delta

        # 清理 0 数量批次
        lots[:] = [l for l in lots if l.quantity > 0]
        return extra_lost

    def _travel(self, target: str, *, fast: bool = False) -> None:
        """陆运出行（海岛需走「出海」）。"""
        p = self.state.player
        start = p.location
        if start == target:
            self._log("你已经在该城市。")
            return
        if target in SEA_CITIES:
            self._log("海岛需使用「出海」前往。")
            return
        mode = TransportMode.LAND
        try:
            validate_mode_allowed(mode, start, target)
            km = route_km(mode, start, target)
        except RouteNotFound as e:
            self._log(f"无法到达：{e}")
            return
        damage_ratio, time_factor, loss_factor = _truck_damage_factors(p.truck_durability)
        if p.truck_durability <= TRUCK_MIN_DURABILITY_FOR_TRAVEL:
            self._log(f"货车耐久≤{TRUCK_MIN_DURABILITY_FOR_TRAVEL:.0f}%，必须先维修。")
            return
        base_days = sample_travel_days(mode, km, self.rng)
        days = max(1, int(round(base_days * time_factor)))
        if fast:
            days = max(FAST_TRAVEL_MIN_DAYS, int(days // FAST_TRAVEL_TIME_DIVISOR))
            cost = km * self.land_cost_per_km * max(1, int(getattr(p, "truck_count", 1))) * FAST_TRAVEL_COST_MULTIPLIER
        else:
            cost = km * self.land_cost_per_km * max(1, int(getattr(p, "truck_count", 1)))
        if p.cash < cost:
            self._log(f"现金不足：本次运输成本 {cost:.0f} 元。")
            return
        # 录制：出行动作（扣钱/推进天数前记录 pre-action 状态）
        if self.demo_recording and self.demo_recorder is not None:
            snap = copy.deepcopy(self.state)
            self.demo_recorder.record(snap, self.rng, api_actions.ActionTravel(city=target, mode="land"))
        p.cash = round(p.cash - cost, 2)
        p.truck_durability = max(0.0, round(p.truck_durability - km * TRUCK_DURABILITY_LOSS_PER_KM, 2))
        for _ in range(days):
            result = self._do_advance_day(skip_day_limit_check=True)
            if result is None:
                return
            msgs, _ = result
            for m in msgs:
                self._log(m)
        per_trip_loss: Dict[str, int] = {}
        lost = apply_transport_loss(p.cargo_lots, origin_city=start, target_city=target, km=float(km), days=days, rng=self.rng, loss_stats=per_trip_loss)
        if damage_ratio > 0:
            lost += self._apply_additional_truck_damage_loss(p.cargo_lots, origin_city=start, target_city=target, damage_ratio=damage_ratio, loss_stats=per_trip_loss)
        if lost > 0:
            self._log(f"运输损耗：损失 {lost} 单位货物。")
        if per_trip_loss:
            acc = self.state.loss_by_product
            for pid, n in per_trip_loss.items():
                acc[pid] = acc.get(pid, 0) + n
        p.location = target
        self._log(f"到达 {target}（陆运，{km}km，用时 {days} 天，成本 {cost:.0f}）。")
        self._check_and_settle_if_day_limit()
        try:
            save_game(self.state, "autosave")
        except Exception:
            pass

    def _sail(self, target: str, *, fast: bool = False) -> None:
        """出海：大陆海港→海岛、海岛→其他海岛、或海岛→大陆出海港。费用与损耗同原海运逻辑。"""
        p = self.state.player
        start = p.location
        if start == target:
            self._log("你已经在该城市。")
            return
        dests = self._sail_destinations()
        if target not in dests:
            self._log("该目的地不在当前出海航线内。")
            return
        try:
            validate_mode_allowed(TransportMode.SEA, start, target)
            km = route_km(TransportMode.SEA, start, target)
        except RouteNotFound as e:
            self._log(f"无法到达：{e}")
            return
        units = current_cargo_units(p)
        if start not in LAND_SEA_CITIES and start not in SEA_CITIES:
            self._log("当前城市无法出海。")
            return
        base_days = sample_travel_days(TransportMode.SEA, km, self.rng)
        days = base_days
        if fast:
            days = max(FAST_TRAVEL_MIN_DAYS, int(days // FAST_TRAVEL_TIME_DIVISOR))
        is_taiwan = (start in ("台北", "高雄")) ^ (target in ("台北", "高雄"))
        customs = TAIWAN_CUSTOMS if is_taiwan else 0.0
        base_cost = km * self.sea_cost_per_km + customs
        total_cap = total_storage_capacity(p, p.location)
        load_mult = 1.0 + (units / max(1, total_cap))
        cost = base_cost * load_mult
        if fast:
            cost *= FAST_TRAVEL_COST_MULTIPLIER
        if p.cash < cost:
            self._log(f"现金不足：本次出海成本 {cost:.0f} 元。")
            return
        # 录制：出海动作（扣钱/推进天数前记录 pre-action 状态）
        if self.demo_recording and self.demo_recorder is not None:
            snap = copy.deepcopy(self.state)
            self.demo_recorder.record(snap, self.rng, api_actions.ActionTravel(city=target, mode="sea"))
        p.cash = round(p.cash - cost, 2)
        if start in LAND_SEA_CITIES and target in SEA_CITIES:
            p.sea_departure_port = start
        elif start in SEA_CITIES and target in LAND_SEA_CITIES:
            p.sea_departure_port = ""
        for _ in range(days):
            result = self._do_advance_day(skip_day_limit_check=True)
            if result is None:
                return
            msgs, _ = result
            for m in msgs:
                self._log(m)
        per_trip_loss_sea: Dict[str, int] = {}
        lost = apply_transport_loss(p.cargo_lots, origin_city=start, target_city=target, km=float(km), days=days, rng=self.rng, loss_stats=per_trip_loss_sea)
        if lost > 0:
            self._log(f"运输损耗：损失 {lost} 单位货物。")
        if per_trip_loss_sea:
            acc = self.state.loss_by_product
            for pid, n in per_trip_loss_sea.items():
                acc[pid] = acc.get(pid, 0) + n
        p.location = target
        self._log(f"到达 {target}（海运，{km}km，用时 {days} 天，成本 {cost:.0f}）。")
        self._check_and_settle_if_day_limit()
        try:
            save_game(self.state, "autosave")
        except Exception:
            pass

    # --- 市场 / 借贷 操作 ---

    def _buy_in_ui(self, pid: str, qty: int) -> Tuple[bool, str]:
        p = self.state.player
        city = p.location
        if pid not in PRODUCTS:
            return False, "未知商品。"
        if qty <= 0:
            return False, "数量必须>0。"
        prod = PRODUCTS[pid]
        unit = purchase_price(prod, city, self.state.daily_lambdas)
        if unit is None:
            return False, "该城市无法采购此商品（仅产地可买，基础品除外）。"
        used = cargo_used(p.cargo_lots)
        cap = total_storage_capacity(p, city)
        if used + qty > cap:
            return False, f"容量不足：当前 {used}/{cap}，本次需要 {qty}。"
        cost = round(unit * qty, 2)
        if p.cash < cost:
            return False, f"现金不足：需要 {cost:.2f}，当前 {p.cash:.2f}。"
        p.cash = round(p.cash - cost, 2)
        add_lot(
            p.cargo_lots,
            CargoLot(
                product_id=pid,
                quantity=qty,
                origin_city=city,
                shelf_life_remaining_days=prod.perishable_shelf_life_days,
            ),
        )
        return True, f"已采购 {prod.name} x{qty}，单价 {unit:.2f}，共 {cost:.2f}。"

    def _sell_in_ui(self, pid: str, qty: int) -> Tuple[bool, str]:
        from trade_game.inventory import remove_quantity_fifo

        p = self.state.player
        city = p.location
        if pid not in PRODUCTS:
            return False, "未知商品。"
        if qty <= 0:
            return False, "数量必须>0。"
        prod = PRODUCTS[pid]

        # 特产售卖限制：城市特产不能在产地售卖；区域特产不能在同一区域售卖
        if not can_sell_product_here(prod, city):
            return False, "特产必须运出本地/本区域后才能售卖。"

        have = sum(l.quantity for l in p.cargo_lots if l.product_id == pid)
        if have <= 0:
            return False, "你没有该商品。"
        sell_qty = min(qty, have)
        actual, removed = remove_quantity_fifo(p.cargo_lots, pid, sell_qty)
        if actual <= 0:
            return False, "售卖失败。"
        min_shelf = None
        for lot in removed:
            if lot.shelf_life_remaining_days is None:
                continue
            min_shelf = (
                lot.shelf_life_remaining_days
                if min_shelf is None
                else min(min_shelf, lot.shelf_life_remaining_days)
            )
        unit = sell_unit_price(
            prod,
            city,
            self.state.daily_lambdas,
            quantity_sold=actual,
            shelf_life_remaining_days=min_shelf,
        )
        revenue = round(unit * actual, 2)
        p.cash = round(p.cash + revenue, 2)
        return True, f"已售卖 {prod.name} x{actual}，单价 {unit:.2f}，共 {revenue:.2f}。"

    def _do_borrow_ui(self) -> None:
        p = self.state.player
        loc = p.location
        if not CITIES[loc].has_bank:
            self._log("当前城市没有银行，无法借贷。")
            return
        amount = max(100.0, self.loans_amount)
        # 净资产 = 总现金 - 总债务，可借金额 ≤ 净资产（可为负）
        total_debt_amount = sum(l.debt_total() for l in self.state.loans)
        net_assets = p.cash - total_debt_amount
        principal_total = total_outstanding_principal(self.state.loans)
        if principal_total + amount > net_assets:
            self._log(f"超出额度：净资产 {net_assets:.0f}，已借本金 {principal_total:.0f}。")
            return
        # 录制：借贷（pre-action 状态）
        snap = copy.deepcopy(self.state) if (self.demo_recording and self.demo_recorder is not None) else None
        # 简化后：统一按日利率计息，不再区分模式
        loan = borrow(self.state.loans, amount=amount, day=p.day, interest_mode="simple")
        p.cash = round(p.cash + amount, 2)
        rate_pct = LOAN_DAILY_INTEREST_RATE * 100
        self._log(f"借款成功：{amount:.0f} 元（日利率 {rate_pct:.1f}%，可随时还款）。")
        if snap is not None:
            self.demo_recorder.record(snap, self.rng, api_actions.ActionBorrow(amount=float(amount)))
        # 借贷成功也消耗 1 天
        result = self._do_advance_day()
        if result is not None:
            msgs, _ = result
            for m in msgs:
                self._log(m)

    def _do_repay_ui(self) -> None:
        p = self.state.player
        loc = p.location
        if not CITIES[loc].has_bank:
            self._log("当前城市没有银行，无法还款。")
            return
        if not self.state.loans:
            self._log("当前无借贷。")
            return

        # 使用借贷输入框中的金额作为还款金额
        amount = max(100.0, self.loans_amount)
        before = p.cash
        p.cash = repay(self.state.loans, cash=p.cash, amount=amount)
        spent = before - p.cash
        if spent <= 0:
            self._log("现金不足，无法还款。")
            return
        self._log(f"已还款：{spent:.0f} 元。")

    def _update_price_info_text(self) -> None:
        """
        价格信息（趋势分析）：
        - 哪些售卖价处于“上涨趋势”
        - 哪些采购价处于“下降趋势”
        同时生成：
        - 售出价连续 N 天涨价：售卖机会
        - 采购价连续 N 天降价：采购机会
        - 采购价连续 N 天涨价：采购预警
        - 售出价连续 N 天降价：售卖预警
        """
        from trade_game.data import PRODUCTS, CITIES

        # 使用最近 N 天来判断趋势（默认 5 天）
        WINDOW_DAYS = 5

        def _trend_pct(values: List[float]) -> float:
            if not values:
                return 0.0
            first = float(values[0])
            last = float(values[-1])
            if first == 0:
                return 0.0
            return (last - first) / abs(first) * 100.0

        def _is_monotonic(values: List[float], *, direction: str) -> bool:
            if len(values) < WINDOW_DAYS:
                return False
            last = values[-WINDOW_DAYS:]
            if direction == "down":
                return all(last[i] < last[i - 1] for i in range(1, WINDOW_DAYS))
            return all(last[i] > last[i - 1] for i in range(1, WINDOW_DAYS))

        sell_up: List[Tuple[float, str, str, str, List[float]]] = []
        buy_down: List[Tuple[float, str, str, str, List[float]]] = []
        buy_down_n: List[str] = []
        sell_up_n: List[str] = []
        buy_up_n: List[str] = []
        sell_down_n: List[str] = []

        # 售卖价上涨趋势
        for key, hist in self.state.price_history_sell_7d.items():
            if len(hist) < WINDOW_DAYS:
                continue
            city, pid = key.split("|", 1)
            prod = PRODUCTS.get(pid)
            if not prod:
                continue
            pct = _trend_pct(hist[-WINDOW_DAYS:])
            if pct > 0:
                sell_up.append((pct, city, pid, prod.name, hist[-7:]))
            if _is_monotonic(hist, direction="up"):
                sell_up_n.append(f"{city} - {prod.name}")
            if _is_monotonic(hist, direction="down"):
                sell_down_n.append(f"{city} - {prod.name}")

        # 采购价下降趋势（只有有采购记录的才会出现）
        for key, hist in self.state.price_history_buy_7d.items():
            if len(hist) < WINDOW_DAYS:
                continue
            city, pid = key.split("|", 1)
            prod = PRODUCTS.get(pid)
            if not prod:
                continue
            pct = _trend_pct(hist[-WINDOW_DAYS:])
            if pct < 0:
                buy_down.append((pct, city, pid, prod.name, hist[-7:]))
            if _is_monotonic(hist, direction="down"):
                buy_down_n.append(f"{city} - {prod.name}")
            if _is_monotonic(hist, direction="up"):
                buy_up_n.append(f"{city} - {prod.name}")

        sell_up.sort(key=lambda x: x[0], reverse=True)
        buy_down.sort(key=lambda x: x[0])  # pct 越小降得越多

        # 弹窗文本（表格形式）
        lines: List[str] = []
        lines.append(f"【售卖价上涨趋势 Top20】（按近 {WINDOW_DAYS} 天涨幅%排序）")
        if not sell_up:
            lines.append("  （暂无上涨趋势数据）")
        else:
            lines.append("  城市 | 商品 | 7日涨幅% | 序列(近7天)")
            for pct, city, _pid, name, seq in sell_up[:20]:
                series = ",".join(f"{v:.2f}" for v in seq)
                lines.append(f"  {city} | {name} | +{pct:.1f}% | {series}")

        lines.append("")
        lines.append(f"【采购价下降趋势 Top20】（按近 {WINDOW_DAYS} 天跌幅%排序）")
        if not buy_down:
            lines.append("  （暂无下降趋势数据）")
        else:
            lines.append("  城市 | 商品 | 7日跌幅% | 序列(近7天)")
            for pct, city, _pid, name, seq in buy_down[:20]:
                series = ",".join(f"{v:.2f}" for v in seq)
                lines.append(f"  {city} | {name} | {pct:.1f}% | {series}")

        lines.append("")
        lines.append(f"【连续 {WINDOW_DAYS} 天机会/预警】")
        lines.append(f"  售出价连续 {WINDOW_DAYS} 天涨价：【售卖机会】")
        if sell_up_n:
            for s in sell_up_n[:30]:
                lines.append(f"    {s}")
        else:
            lines.append("    （暂无）")
        lines.append(f"  采购价连续 {WINDOW_DAYS} 天降价：【采购机会】")
        if buy_down_n:
            for s in buy_down_n[:30]:
                lines.append(f"    {s}")
        else:
            lines.append("    （暂无）")
        lines.append(f"  采购价连续 {WINDOW_DAYS} 天涨价：【采购预警】")
        if buy_up_n:
            for s in buy_up_n[:30]:
                lines.append(f"    {s}")
        else:
            lines.append("    （暂无）")
        lines.append(f"  售出价连续 {WINDOW_DAYS} 天降价：【售卖预警】")
        if sell_down_n:
            for s in sell_down_n[:30]:
                lines.append(f"    {s}")
        else:
            lines.append("    （暂无）")

        self.price_info_lines = lines

        # 笔记框建议（精简）
        # 笔记框建议：若没有任何机会则不显示
        note: List[str] = []
        if sell_up_n:
            note.append(f"【售卖机会】连续涨价{WINDOW_DAYS}天：")
            for s in sell_up_n[:5]:
                note.append(f"- {s}")
        if buy_down_n:
            note.append(f"【采购机会】连续降价{WINDOW_DAYS}天：")
            for s in buy_down_n[:5]:
                note.append(f"- {s}")
        if buy_up_n:
            note.append(f"【采购预警】连续涨价{WINDOW_DAYS}天：")
            for s in buy_up_n[:5]:
                note.append(f"- {s}")
        if sell_down_n:
            note.append(f"【售卖预警】连续降价{WINDOW_DAYS}天：")
            for s in sell_down_n[:5]:
                note.append(f"- {s}")
        self.price_note_lines = note

    def _update_price_notes_only(self) -> None:
        """
        仅用于主页右侧笔记框的简要机会提示：
        - 售出价连续 N 天涨价：售卖机会
        - 采购价连续 N 天降价：采购机会
        - 采购价连续 N 天涨价：采购预警
        - 售出价连续 N 天降价：售卖预警
        不做复杂统计，只基于价格历史判断是否存在这样的商品。
        """
        from trade_game.data import PRODUCTS

        WINDOW_DAYS = 5
        buy_down_n: List[str] = []
        sell_up_n: List[str] = []
        buy_up_n: List[str] = []
        sell_down_n: List[str] = []

        # 连续 N 天采购价下降
        for key, hist in self.state.price_history_buy_7d.items():
            if len(hist) < WINDOW_DAYS:
                continue
            last = hist[-WINDOW_DAYS:]
            if all(last[i] < last[i - 1] for i in range(1, WINDOW_DAYS)):
                city, pid = key.split("|", 1)
                prod = PRODUCTS.get(pid)
                if prod:
                    buy_down_n.append(f"{city} - {prod.name}")
            if all(last[i] > last[i - 1] for i in range(1, WINDOW_DAYS)):
                city, pid = key.split("|", 1)
                prod = PRODUCTS.get(pid)
                if prod:
                    buy_up_n.append(f"{city} - {prod.name}")

        # 连续 N 天售卖价上涨
        for key, hist in self.state.price_history_sell_7d.items():
            if len(hist) < WINDOW_DAYS:
                continue
            last = hist[-WINDOW_DAYS:]
            if all(last[i] > last[i - 1] for i in range(1, WINDOW_DAYS)):
                city, pid = key.split("|", 1)
                prod = PRODUCTS.get(pid)
                if prod:
                    sell_up_n.append(f"{city} - {prod.name}")
            if all(last[i] < last[i - 1] for i in range(1, WINDOW_DAYS)):
                city, pid = key.split("|", 1)
                prod = PRODUCTS.get(pid)
                if prod:
                    sell_down_n.append(f"{city} - {prod.name}")

        # 若没有任何机会则不显示
        note: List[str] = []
        if sell_up_n:
            note.append(f"【售卖机会】连续涨价{WINDOW_DAYS}天：")
            for s in sell_up_n[:5]:
                note.append(f"- {s}")
        if buy_down_n:
            note.append(f"【采购机会】连续降价{WINDOW_DAYS}天：")
            for s in buy_down_n[:5]:
                note.append(f"- {s}")
        if buy_up_n:
            note.append(f"【采购预警】连续涨价{WINDOW_DAYS}天：")
            for s in buy_up_n[:5]:
                note.append(f"- {s}")
        if sell_down_n:
            note.append(f"【售卖预警】连续降价{WINDOW_DAYS}天：")
            for s in sell_down_n[:5]:
                note.append(f"- {s}")
        self.price_note_lines = note

    def _draw_price_info_dialog(self) -> None:
        """【价格信息】弹窗：使用文本框 + 简易滚动按钮展示 7 日价格分析结果。"""
        from arcade import draw_lbwh_rectangle_filled

        # 放大显示（约 2 倍），但不超过屏幕可用范围
        width = min(SCREEN_WIDTH - 80, 1400)
        height = min(SCREEN_HEIGHT - 80, 1040)

        # 外框
        x1, y1, x2, y2 = self._draw_dialog_window("价格信息", width, height)

        dialog_box = UIBox(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2, width, height)
        content_box = dialog_box.pad(30)

        # 上方标题与说明
        header_box, main_box = content_box.split_vertical(60)
        arcade.draw_text(
            "最近 7 日各城市进货价 / 售卖价统计（仅供参考）。",
            header_box.l + 10,
            header_box.t - 20,
            WIN98_TEXT_DARK,
            13,
            anchor_x="left",
            anchor_y="top",
        )

        # 主体：文本框 + 滚动按钮区域
        text_box, control_box = main_box.split_vertical(main_box.height - 40)

        draw_lbwh_rectangle_filled(text_box.l, text_box.b, text_box.width, text_box.height, WIN98_BG_LIGHT)
        self._draw_win98_3d_border(text_box.l, text_box.b, text_box.width, text_box.height, raised=False)

        inner = text_box.pad(15)
        line_height = 20

        # 计算当前可显示的最大行数
        max_visible = max(1, int(inner.height // line_height) - 1)
        total_lines = len(self.price_info_lines)
        start = max(0, min(self.price_info_scroll, max(0, total_lines - max_visible)))
        end = start + max_visible

        # 绘制文本
        y = inner.t - 5
        arcade.draw_text("【价格信息】", inner.l, y, WIN98_TEXT_LIGHT, 13, anchor_y="top", bold=True)
        y -= line_height
        if not self.price_info_lines:
            arcade.draw_text("暂无数据，请先在市场中多进行几天交易。", inner.l, y, WIN98_TEXT_LIGHT, 12, anchor_y="top")
        else:
            for line in self.price_info_lines[start:end]:
                arcade.draw_text(line, inner.l, y, WIN98_TEXT_LIGHT, 12, anchor_y="top")
                y -= line_height

        # 绘制右侧滚动条轨道和滑块（仅在内容超过一页时显示）
        if total_lines > max_visible:
            scrollbar_w = 10
            track_x = inner.r - scrollbar_w - 2
            track_y = inner.b
            track_h = inner.height
            from arcade import draw_lbwh_rectangle_filled
            # 轨道
            draw_lbwh_rectangle_filled(track_x, track_y, scrollbar_w, track_h, WIN98_BG_DARK)
            self._draw_win98_3d_border(track_x, track_y, scrollbar_w, track_h, raised=False)

            # 滑块高度与位置
            handle_min_h = 20
            handle_h = max(handle_min_h, int(track_h * (max_visible / total_lines)))
            max_scroll = max(0, total_lines - max_visible)
            if max_scroll > 0:
                scroll_ratio = start / max_scroll
            else:
                scroll_ratio = 0.0
            handle_y = track_y + int((track_h - handle_h) * scroll_ratio)
            draw_lbwh_rectangle_filled(track_x + 1, handle_y + 1, scrollbar_w - 2, handle_h - 2, WIN98_BUTTON_HIGHLIGHT)

            # 记录滚动条几何信息供拖动使用
            self.price_info_scrollbar_meta = (
                float(track_x),
                float(track_y),
                float(track_h),
                float(handle_h),
                int(max_visible),
                int(total_lines),
            )
        else:
            self.price_info_scrollbar_meta = None

        # 滚动控制按钮
        ctrl_grid = UIGrid(
            left=control_box.l,
            bottom=control_box.b,
            width=control_box.width,
            height=control_box.height,
            rows=1,
            cols=3,
            padding=10,
        )
        up_x, up_y = ctrl_grid.pos(0, 0)
        down_x, down_y = ctrl_grid.pos(0, 1)
        close_x, close_y = ctrl_grid.pos(0, 2)

        self._draw_win98_button("price_info_up", up_x, up_y, 80, 26, "向上", pressed=False)
        self._draw_win98_button("price_info_down", down_x, down_y, 80, 26, "向下", pressed=False)
        self._draw_win98_button("price_info_close", close_x, close_y, 90, 28, "关闭", pressed=False)

    # --- 弹窗绘制函数（Win98风格） ---
    
    def _draw_dialog_window(
        self, title: str, width: float, height: float, has_close: bool = True,
        help_key: Optional[str] = None,
    ) -> Tuple[float, float, float, float]:
        """绘制Win98风格的对话框窗口，返回窗口区域 (x1, y1, x2, y2)。
        help_key: 若提供，在关闭按钮旁绘制「?」帮助按钮，点击后显示对应说明。
        """
        from arcade import draw_lbwh_rectangle_filled
        
        x = SCREEN_WIDTH / 2
        y = SCREEN_HEIGHT / 2
        x1 = x - width / 2
        y1 = y - height / 2
        x2 = x + width / 2
        y2 = y + height / 2
        
        # 窗口主体背景
        draw_lbwh_rectangle_filled(x1, y1, width, height, WIN98_BUTTON_FACE)
        self._draw_win98_3d_border(x1, y1, width, height, raised=True)
        
        # 标题栏
        title_bar_height = 26
        title_bar_y = y2 - title_bar_height
        draw_lbwh_rectangle_filled(x1, title_bar_y, width, title_bar_height, WIN98_TITLE_BAR)
        arcade.draw_text(
            title, x1 + 10, title_bar_y + 8, WIN98_BUTTON_HIGHLIGHT, 13, anchor_y="bottom"
        )
        
        # 右上角按钮区：帮助(?)、关闭(×)
        btn_size = 22
        btn_gap = 4
        close_x = x2 - btn_size - 6
        close_y = title_bar_y + title_bar_height / 2
        
        if help_key and help_key in DIALOG_HELP_TEXTS:
            help_btn_x = close_x - btn_size - btn_gap
            self._draw_win98_button(
                "dialog_help", help_btn_x, close_y, btn_size, btn_size - 4, "?", pressed=False
            )
        
        if has_close:
            self._draw_win98_button(
                "dialog_close", close_x, close_y, btn_size, btn_size - 4, "×", pressed=False
            )
        
        return (x1, y1, x2, y2)
    
    def _draw_travel_dialog(self) -> None:
        """【出行】弹窗 - 外框+切片+网格化布局"""
        from arcade import draw_lbwh_rectangle_filled
        
        width = 950
        height = 620
        
        # 1. 外框：绘制居中的大框
        x1, y1, x2, y2 = self._draw_dialog_window("出行选择", width, height, help_key="travel")
        
        # 2. 切片：使用UIBox进行区域切分
        dialog_box = UIBox(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2, width, height)
        
        # 内容区域
        content_box = dialog_box.pad(40)
        
        # 左右分栏：左侧为城市列表，右侧再细分为「普通出行」与「快速出行」两列
        left_box, right_outer_box = content_box.split_horizontal(0.45)
        
        # 3. 网格化：在每个区域内建立局部网格
        
        # 左侧城市列表网格：行数自适应，避免“列不全”
        # 需要显示：4 个地区 + 1 个高消费经济区（可与其他组重叠），并列出除所在地外所有城市
        p = self.state.player
        regions_ordered: List[Tuple[str, List[str]]] = []
        regions_ordered.extend(list(TRAVEL_REGIONS.items()))
        regions_ordered.append(("高消费经济区", sorted(HIGH_CONSUMPTION_CITIES)))

        # 估算最坏情况下（全部展开）的行数：每组 1 行标题 + 城市行
        worst_rows = 1  # 预留标题后第一行
        for region_name, cities in regions_ordered:
            worst_rows += 1  # 分组按钮
            worst_rows += sum(1 for c in cities if c != p.location)

        city_grid_rows = max(15, min(28, worst_rows + 1))
        city_grid = left_box.make_grid(city_grid_rows, 1, gap=8)
        
        # 右侧路线预览区域再细分为两列：中间普通出行 + 右侧快速出行
        normal_box, fast_box = right_outer_box.split_horizontal(0.5)
        # 各自使用独立的网格
        normal_grid = normal_box.make_grid(12, 1, gap=15)
        fast_grid = fast_box.make_grid(12, 1, gap=15)
        
        # === 左侧：城市列表区域 ===
        # 标题
        arcade.draw_text("选择目的地", left_box.l + 10, left_box.t - 20, WIN98_TEXT_DARK, 14, anchor_y="top", bold=True)
        
        # 使用统一维护的区域分组 + “高消费经济区”（允许重叠）
        reachable = self._cli_reachable_land_cities()
        if self.demo_recording and p.truck_durability <= TRUCK_MIN_DURABILITY_FOR_TRAVEL and not reachable:
            arcade.draw_text("货车耐久过低，需先维修才能陆运（CLI 规则）", left_box.l + 10, left_box.t - 45, WIN98_BUTTON_DARK, 11, anchor_y="top")
        
        current_row = 1
        for region_name, cities in regions_ordered:
            if current_row >= city_grid.rows:
                break
            
            # 区域折叠按钮
            expanded = self.travel_region_expanded.get(region_name, True)
            marker = "[-]" if expanded else "[+]"
            
            # 使用网格定位区域按钮
            btn_x, btn_y = city_grid.pos(current_row, 0)
            self._draw_win98_button(
                f"region_{region_name}", btn_x, btn_y, 130, 24, f"{marker} {region_name}", pressed=False
            )
            current_row += 1
            
            if expanded:
                for city in cities:
                    if current_row >= city_grid.rows - 1:
                        break
                    if city == p.location:
                        continue
                    can_reach = city in reachable
                    city_color = WIN98_TEXT_DARK if can_reach else WIN98_BUTTON_DARK

                    # 计算预计信息（不可达也展示耗时/费用，只是保持灰色）
                    plan = self._preview_travel_plan(city, allow_unreachable=False)
                    if plan is None:
                        plan = self._preview_travel_plan(city, allow_unreachable=True)

                    if plan is None:
                        info_text = "  耗时: --天  费用: --元"
                    else:
                        mode, km, days = plan
                        truck_count = max(1, int(getattr(p, "truck_count", 1)))
                        if mode == TransportMode.LAND:
                            # 陆运：里程费用 × 车辆数
                            cost = km * self.land_cost_per_km * truck_count
                        else:
                            # 海运：基于里程数计算费用，大陆↔台湾有额外关税
                            is_taiwan = (p.location in ("台北", "高雄")) ^ (city in ("台北", "高雄"))
                            customs = TAIWAN_CUSTOMS if is_taiwan else 0.0
                            cost = km * self.sea_cost_per_km + customs
                        info_text = f"  耗时: {days}天  费用: {cost:.0f}元"
                    
                    # 城市行
                    text_x, text_y = city_grid.pos(current_row, 0)
                    if can_reach:
                        self.city_row_regions.append(
                            (city, (left_box.l, text_y - 14, left_box.l + left_box.width, text_y + 14))
                        )
                    
                    arcade.draw_text(
                        city + info_text, left_box.l + 10, text_y, city_color, 11, anchor_y="center"
                    )
                    current_row += 1
        
        # === 右侧：路线预览（普通出行 + 快速出行） ===
        if self.travel_target:
            target = self.travel_target
            # 目的地标题（放在普通出行列顶部）
            title_x, title_y = normal_grid.pos(0, 0)
            arcade.draw_text(f"目的地：{target}", normal_box.l + 10, normal_box.t - 40, WIN98_TEXT_DARK, 16, anchor_y="top", bold=True)
            
            try:
                preview = self._preview_travel_days(target)
                if preview is None:
                    raise RouteNotFound("unreachable")
                km, days = preview

                # 出行弹窗仅显示陆运（海岛走「出海」）
                mode = TransportMode.LAND
                mode_text = "陆运"
                truck_count = max(1, int(getattr(p, "truck_count", 1)))
                cost_normal = km * self.land_cost_per_km * truck_count
                
                info_items = [
                    ("出行方式", mode_text),
                    ("预计里程", f"{km} km"),
                    ("预计耗时", f"{days} 天"),
                    ("基础运输成本", f"{cost_normal:.0f} 元"),
                ]
                
                for i, (label, value) in enumerate(info_items):
                    text_x, text_y = normal_grid.pos(i + 2, 0)
                    arcade.draw_text(f"{label}：{value}", normal_box.l + 10, text_y, WIN98_TEXT_DARK, 13, anchor_y="center")
                
                # 损耗提示（预计运损：生鲜按商品区分，易损与里程成正比；实际再×0.9～1.1）
                loss_x, loss_y = normal_grid.pos(6, 0)
                # 标题行
                arcade.draw_text(
                    "预计运输损耗：",
                    normal_box.l + 10,
                    loss_y,
                    WIN98_TEXT_DARK,
                    11,
                    anchor_y="center",
                )
                # 易损 / 轻工 / 民用 三行纵向排列
                loss_lines = expected_transport_loss_display(km, days).splitlines()
                cat_y = loss_y - 16
                for line in loss_lines:
                    arcade.draw_text(
                        f"  {line}",
                        normal_box.l + 20,
                        cat_y,
                        WIN98_TEXT_DARK,
                        11,
                        anchor_y="center",
                    )
                    cat_y -= 14

                # 生鲜逐项明细：整体下移，放在大类损耗之后
                perishable_details = expected_perishable_loss_details(km, days)
                if perishable_details:
                    detail_y = cat_y - 18
                    for name, pct in perishable_details:
                        arcade.draw_text(
                            f"  - {name}：约 {pct:.1f}%",
                            normal_box.l + 20,
                            detail_y,
                            WIN98_TEXT_DARK,
                            11,
                            anchor_y="center",
                        )
                        detail_y -= 16

                # === 快速出行信息（费用/时间由 game_config 参数决定） ===
                fast_days = max(FAST_TRAVEL_MIN_DAYS, int(days // FAST_TRAVEL_TIME_DIVISOR))
                fast_cost = cost_normal * FAST_TRAVEL_COST_MULTIPLIER
                f_items = [
                    ("快速出行", mode_text),
                    ("预计里程", f"{km} km"),
                    ("预计耗时", f"{fast_days} 天"),
                    ("快速运输成本", f"{fast_cost:.0f} 元"),
                ]
                for i, (label, value) in enumerate(f_items):
                    fx, fy = fast_grid.pos(i + 2, 0)
                    arcade.draw_text(f"{label}：{value}", fast_box.l + 10, fy, WIN98_TEXT_DARK, 13, anchor_y="center")

                # 快速出行的损耗提示（使用 fast_days 重新计算）
                f_loss_x, f_loss_y = fast_grid.pos(6, 0)
                arcade.draw_text(
                    "预计运输损耗：",
                    fast_box.l + 10,
                    f_loss_y,
                    WIN98_TEXT_DARK,
                    11,
                    anchor_y="center",
                )
                f_loss_lines = expected_transport_loss_display(km, fast_days).splitlines()
                f_cat_y = f_loss_y - 16
                for line in f_loss_lines:
                    arcade.draw_text(
                        f"  {line}",
                        fast_box.l + 20,
                        f_cat_y,
                        WIN98_TEXT_DARK,
                        11,
                        anchor_y="center",
                    )
                    f_cat_y -= 14

                f_perishable = expected_perishable_loss_details(km, fast_days)
                if f_perishable:
                    f_detail_y = f_cat_y - 18
                    for name, pct in f_perishable:
                        arcade.draw_text(
                            f"  - {name}：约 {pct:.1f}%",
                            fast_box.l + 20,
                            f_detail_y,
                            WIN98_TEXT_DARK,
                            11,
                            anchor_y="center",
                        )
                        f_detail_y -= 16

                # 确认按钮：普通出行 + 快速出行
                confirm_x, confirm_y = normal_grid.pos(10, 0)
                self._draw_win98_button("travel_confirm_normal", confirm_x, confirm_y, 130, 32, "普通出行", pressed=False)
                f_confirm_x, f_confirm_y = fast_grid.pos(10, 0)
                self._draw_win98_button("travel_confirm_fast", f_confirm_x, f_confirm_y, 130, 32, "快速出行", pressed=False)
            except RouteNotFound:
                error_x, error_y = normal_grid.pos(2, 0)
                arcade.draw_text(
                    "当前模式下无法到达该城市。", normal_box.l + 10, error_y, (229, 57, 53), 13, anchor_y="center"
                )
        else:
            prompt_x, prompt_y = normal_grid.pos(2, 0)
            arcade.draw_text(
                "请在左侧城市列表中选择目的地。", normal_box.l + 10, prompt_y, WIN98_TEXT_DARK, 13, anchor_y="center"
            )
        
        # 取消按钮（放在普通出行列底部）
        cancel_x, cancel_y = normal_grid.pos(11, 0)
        self._draw_win98_button("travel_cancel", cancel_x, cancel_y, 90, 28, "取消", pressed=False)
    
    def _draw_market_dialog(self) -> None:
        """【市场】弹窗 - 外框+切片+网格化布局"""
        width = 950
        height = 680
        city = self.state.player.location
        
        # 1. 外框：绘制居中的大框
        x1, y1, x2, y2 = self._draw_dialog_window(f"{city}市场", width, height, help_key="market")
        
        # 2. 切片：使用UIBox进行区域切分
        dialog_box = UIBox(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2, width, height)
        
        # 内容区域
        content_box = dialog_box.pad(40)

        # 垂直方向：状态条（40） + 标签条（40） + 剩余作为内容区
        status_box, below_status = content_box.split_vertical(40)
        tab_box, content_main_box = below_status.split_vertical(40)
        
        p = self.state.player
        
        # 3. 网格化：在每个区域内建立局部网格
        
        # 顶部状态区网格
        status_grid = status_box.make_grid(1, 1, gap=10)
        
        # 绘制状态信息
        status_x, status_y = status_grid.pos(0, 0)
        arcade.draw_text(
            f"现金：{p.cash:,.0f} 元  |  载重：{cargo_used(p.cargo_lots)} / {total_storage_capacity(p, p.location)}",
            status_box.l + 10, status_box.t - 15, WIN98_TEXT_DARK, 13, anchor_y="top"
        )
        
        # 标签页区网格
        tab_grid = tab_box.make_grid(1, 2, gap=10)
        
        # 采购标签
        buy_x, buy_y = tab_grid.pos(0, 0)
        buy_pressed = self.market_tab == "buy"
        self._draw_win98_button("market_tab_buy", buy_x, buy_y, 120, 34, "采购", pressed=buy_pressed)
        
        # 售卖标签
        sell_x, sell_y = tab_grid.pos(0, 1)
        sell_pressed = self.market_tab == "sell"
        self._draw_win98_button("market_tab_sell", sell_x, sell_y, 120, 34, "售卖", pressed=sell_pressed)
        
        # 中间内容区
        if self.market_tab == "buy":
            self._draw_market_buy_tab(content_main_box.l, content_main_box.b, content_main_box.width, content_main_box.height)
        else:
            self._draw_market_sell_tab(content_main_box.l, content_main_box.b, content_main_box.width, content_main_box.height)

        # 若有二级“小确认弹窗”，在市场窗口之上再绘制一层
        if self.market_order_dialog is not None:
            # 仅保留小弹窗的按钮可点击，屏蔽底层市场按钮
            self.button_regions.clear()
            self._draw_market_order_dialog()
    
    def _draw_market_buy_tab(self, x1: float, y1: float, width: float, height: float) -> None:
        """市场-采购标签页 - 使用UIGrid布局"""
        p = self.state.player
        city = p.location
        
        # 创建商品列表网格
        # 行数根据商品数量自适应，保证所有商品都能显示出来（至少 12 行）
        total_products = len(PRODUCTS)
        rows = max(12, total_products + 2)  # 预留表头 + 汇总行
        buy_grid = UIGrid(left=x1, bottom=y1, width=width, height=height, rows=rows, cols=5, padding=15)
        
        # 表头：统一使用网格的 pos() 方法定位，确保与内容对齐
        headers = ["商品名", "采购价", "7日均价", "数量", "操作"]
        for col, header in enumerate(headers):
            header_x, header_y = buy_grid.pos(0, col)
            # 商品名列左对齐，其他列居中对齐，与内容保持一致
            if col == 0:
                name_left = buy_grid.left + buy_grid.padding + (buy_grid.cell_w + buy_grid.padding) * 0 + 10
                arcade.draw_text(header, name_left, header_y, WIN98_TEXT_DARK, 14, anchor_x="left", anchor_y="center", bold=True)
            else:
                arcade.draw_text(header, header_x, header_y, WIN98_TEXT_DARK, 14, anchor_x="center", anchor_y="center", bold=True)
        
        # 商品列表：显示当前版本的所有商品
        products = sorted(PRODUCTS.keys())
        visible_count = len(products)

        # 当前选中的行索引（在全部商品范围内循环）
        current_index = self.market_index % visible_count if visible_count > 0 else 0
        
        for i, pid in enumerate(products):
            if i >= buy_grid.rows - 2:
                break
            prod = PRODUCTS[pid]
            buy_price = purchase_price(prod, city, self.state.daily_lambdas)
            
            # 商品名：左对齐（距离单元格左边缘10像素）
            name_x, name_y = buy_grid.pos(i + 1, 0)
            name_left = buy_grid.left + buy_grid.padding + (buy_grid.cell_w + buy_grid.padding) * 0 + 10
            if buy_price is None:
                arcade.draw_text(product_display_name(prod), name_left, name_y, WIN98_BUTTON_DARK, 12, anchor_x="left", anchor_y="center")
            else:
                arcade.draw_text(product_display_name(prod), name_left, name_y, WIN98_TEXT_DARK, 12, anchor_x="left", anchor_y="center")
            
            # 采购价：居中对齐
            price_x, price_y = buy_grid.pos(i + 1, 1)
            if buy_price is None:
                arcade.draw_text("--", price_x, price_y, WIN98_BUTTON_DARK, 12, anchor_x="center", anchor_y="center")
            else:
                # 显示当日实时采购价（保留 economy 中的两位小数）
                arcade.draw_text(f"{buy_price:.2f}", price_x, price_y, WIN98_TEXT_DARK, 12, anchor_x="center", anchor_y="center")

            # 7 日均采购价：使用 GameState.price_history_buy_7d（仅对产地城市有效）
            avg_x, avg_y = buy_grid.pos(i + 1, 2)
            key = f"{city}|{pid}"
            hist_buy = self.state.price_history_buy_7d.get(key, [])
            if hist_buy:
                avg_val = sum(hist_buy) / len(hist_buy)
                arcade.draw_text(f"{avg_val:.2f}", avg_x, avg_y, WIN98_TEXT_DARK, 12, anchor_x="center", anchor_y="center")
            else:
                arcade.draw_text("--", avg_x, avg_y, WIN98_BUTTON_DARK, 12, anchor_x="center", anchor_y="center")
            
            # 数量显示：居中对齐
            qty_x, qty_y = buy_grid.pos(i + 1, 3)
            qty_text = str(self.market_qty) if i == current_index else "1"
            arcade.draw_text(qty_text, qty_x, qty_y, WIN98_TEXT_DARK, 12, anchor_x="center", anchor_y="center")
            
            # 采购按钮
            if buy_price is not None:
                btn_x, btn_y = buy_grid.pos(i + 1, 4)
                # 主操作：采购
                self._draw_win98_button(f"buy_{pid}", btn_x - 40, btn_y, 80, 30, "采购", pressed=False)
                # 查看 7 日价格：记录最近 7 天采购价
                self._draw_win98_button(f"price_hist_buy_{pid}", btn_x + 70, btn_y, 60, 24, "7日", pressed=False)
        
        # 底部说明
        summary_x, summary_y = buy_grid.pos(buy_grid.rows - 1, 0)
        arcade.draw_text(
            "提示：每次完成采购或售卖操作都会消耗 1 天时间。",
            x1 + 20,
            summary_y,
            WIN98_BUTTON_DARK,
            12,
            anchor_y="center",
        )
    
    def _draw_market_sell_tab(self, x1: float, y1: float, width: float, height: float) -> None:
        """市场-售卖标签页 - 使用UIGrid布局"""
        p = self.state.player
        city = p.location
        
        # 库存列表
        inv: dict[str, int] = {}
        for lot in p.cargo_lots:
            inv[lot.product_id] = inv.get(lot.product_id, 0) + lot.quantity
        
        # 创建商品列表网格
        sell_grid = UIGrid(
            left=x1,
            bottom=y1,
            width=width,
            height=height,
            rows=12, cols=6,
            padding=15
        )
        
        # 表头：统一使用网格的 pos() 方法定位，确保与内容对齐
        headers = ["商品名", "持有", "售价", "7日均价", "数量", "操作"]
        for col, header in enumerate(headers):
            header_x, header_y = sell_grid.pos(0, col)
            # 表头统一居中对齐（商品名列除外，使用左对齐）
            if col == 0:
                name_left = sell_grid.left + sell_grid.padding + (sell_grid.cell_w + sell_grid.padding) * 0 + 10
                arcade.draw_text(header, name_left, header_y, WIN98_TEXT_DARK, 14, anchor_x="left", anchor_y="center", bold=True)
            else:
                arcade.draw_text(header, header_x, header_y, WIN98_TEXT_DARK, 14, anchor_x="center", anchor_y="center", bold=True)
        
        inv_list = sorted(inv.items())
        for i, (pid, qty) in enumerate(inv_list):
            if i >= sell_grid.rows - 2:
                break
            prod = PRODUCTS.get(pid)
            if not prod:
                continue

            can_sell_here = can_sell_product_here(prod, city)
            sell_price = sell_unit_price(prod, city, self.state.daily_lambdas, quantity_sold=1)
            
            # 商品名：左对齐（距离单元格左边缘10像素）
            name_x, name_y = sell_grid.pos(i + 1, 0)
            name_left = sell_grid.left + sell_grid.padding + (sell_grid.cell_w + sell_grid.padding) * 0 + 10
            arcade.draw_text(product_display_name(prod), name_left, name_y, WIN98_TEXT_DARK, 12, anchor_x="left", anchor_y="center")
            
            # 持有数量：显示为 总数(损X)（若有损耗）
            hold_x, hold_y = sell_grid.pos(i + 1, 1)
            lost_total = self.state.loss_by_product.get(pid, 0)
            if lost_total > 0:
                hold_text = f"{qty} (损{lost_total})"
            else:
                hold_text = f"{qty}"
            arcade.draw_text(hold_text, hold_x, hold_y, WIN98_TEXT_DARK, 12, anchor_x="center", anchor_y="center")
            
            # 售价：居中对齐（若本地/本区禁止售卖，则用灰色+提示）
            price_x, price_y = sell_grid.pos(i + 1, 2)
            if can_sell_here:
                arcade.draw_text(f"{sell_price:.2f}", price_x, price_y, WIN98_TEXT_DARK, 12, anchor_x="center", anchor_y="center")
            else:
                arcade.draw_text("本地不可售", price_x, price_y, WIN98_BUTTON_DARK, 11, anchor_x="center", anchor_y="center")

            # 7 日均售卖价：从 GameState.price_history_sell_7d 读取
            avg_x, avg_y = sell_grid.pos(i + 1, 3)
            key = f"{city}|{pid}"
            hist_sell = self.state.price_history_sell_7d.get(key, [])
            if hist_sell:
                avg_val = sum(hist_sell) / len(hist_sell)
                arcade.draw_text(f"{avg_val:.2f}", avg_x, avg_y, WIN98_TEXT_DARK, 12, anchor_x="center", anchor_y="center")
            else:
                arcade.draw_text("--", avg_x, avg_y, WIN98_BUTTON_DARK, 12, anchor_x="center", anchor_y="center")
            
            # 数量显示：居中对齐
            qty_x, qty_y = sell_grid.pos(i + 1, 4)
            qty_text = str(self.market_qty) if i == self.market_index % max(1, len(inv_list)) else "1"
            arcade.draw_text(qty_text, qty_x, qty_y, WIN98_TEXT_DARK, 12, anchor_x="center", anchor_y="center")
            
            # 售卖按钮（若本地/本区禁止售卖，则不显示按钮）
            btn_x, btn_y = sell_grid.pos(i + 1, 5)
            if can_sell_here:
                # 主操作：售卖
                self._draw_win98_button(f"sell_{pid}", btn_x - 40, btn_y, 80, 30, "售卖", pressed=False)
                # 查看 7 日售卖价
                self._draw_win98_button(f"price_hist_sell_{pid}", btn_x + 70, btn_y, 60, 24, "7日", pressed=False)
        
        # 底部说明
        summary_x, summary_y = sell_grid.pos(sell_grid.rows - 1, 0)
        arcade.draw_text(
            "提示：每次完成采购或售卖操作都会消耗 1 天时间。",
            x1 + 20,
            summary_y,
            WIN98_BUTTON_DARK,
            12,
            anchor_y="center",
        )

    def _draw_market_order_dialog(self) -> None:
        """市场二级小弹窗：确认本次采购/售出的数量与金额。"""
        if not self.market_order_dialog:
            return

        mode = self.market_order_dialog.get("mode")
        pid = self.market_order_dialog.get("pid")

        # 文本输入优先，转换为数量；录制/演示模式下仅允许按钮选择档位
        if self.demo_recording:
            qty = int(self.market_order_dialog.get("qty", 1))
            qty = max(1, qty)
            self.market_order_dialog["qty"] = qty
        else:
            text = str(self.market_order_dialog.get("text", "")).strip()
            if not text:
                qty = int(self.market_order_dialog.get("qty", 1))
            else:
                try:
                    qty = int(text)
                except ValueError:
                    qty = 1
            qty = max(1, qty)
            self.market_order_dialog["qty"] = qty
            # 仅当用户输入了内容时才写回 text，空时保持 "" 以显示占位符
            if text:
                self.market_order_dialog["text"] = text

        if not pid or mode not in ("buy", "sell"):
            return

        p = self.state.player
        city = p.location
        prod = PRODUCTS.get(pid)
        if not prod:
            return

        # 计算单价与上限，并做数量裁剪（超过上限时自动裁到上限并提示）
        if mode == "buy":
            unit = purchase_price(prod, city, self.state.daily_lambdas)
            if unit is None:
                unit = prod.base_purchase_price
            cap = total_storage_capacity(p, city)
            from trade_game.inventory import cargo_used
            capacity_left = max(0, cap - cargo_used(p.cargo_lots))
            max_by_cash = int(p.cash / unit) if unit > 0 else capacity_left
            max_qty = max(1, min(capacity_left, max_by_cash))
            if qty > max_qty:
                qty = max_qty
                self.market_order_dialog["qty"] = qty
                self.market_order_dialog["text"] = str(qty)
            total = round(unit * qty, 2)
            balance_after = round(p.cash - total, 2)
            total_sale_price = 0.0  # 采购模式不用
            at_limit = max_qty > 0 and qty >= max_qty
            title = "确认采购"
            summary_label = "预计花费"
        else:
            have = sum(l.quantity for l in p.cargo_lots if l.product_id == pid)
            max_qty = max(0, have)
            if qty > max_qty:
                qty = max_qty
                self.market_order_dialog["qty"] = qty
                self.market_order_dialog["text"] = str(qty)
            effective_qty = max(0, min(qty, have))
            at_limit = max_qty > 0 and qty >= max_qty
            unit = sell_unit_price(prod, city, self.state.daily_lambdas, quantity_sold=max(1, effective_qty))
            revenue = round(unit * effective_qty, 2)
            # 将历史运输损耗成本按当前库存比例摊到本次售卖：
            # 若卖出全部库存，则把“当前持有 + 历史损耗”的成本全算进来。
            lost_total = self.state.loss_by_product.get(pid, 0)
            if have > 0 and lost_total > 0:
                loss_share = lost_total * (effective_qty / have)
            else:
                loss_share = 0.0
            cost_units = effective_qty + loss_share
            cost_basis = round(prod.base_purchase_price * cost_units, 2)
            total = round(revenue - cost_basis, 2)  # 预计利润 = 收入 - 成本（含历史损耗分摊）
            total_sale_price = revenue  # 售出总价
            balance_after = 0.0  # 售出模式不用
            title = "确认售出"
            summary_label = "预计利润"

        # 小弹窗尺寸：为新增档位按钮预留空间（面积约 4 倍 = 宽高各 *2）
        # 同时做屏幕裁剪，避免超出窗口导致绘制/点击区域错位。
        margin = 40
        box_width = min(int(460 * 2), int(SCREEN_WIDTH - margin))
        box_height = min(int(280 * 2), int(SCREEN_HEIGHT - margin))
        x1, y1, x2, y2 = self._draw_dialog_window(title, box_width, box_height)

        dialog_box = UIBox(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2, box_width, box_height)
        content = dialog_box.pad(20)  # 统一内边距
        hint_bar_height = 32
        main_content, hint_bar = content.split_vertical(content.height - hint_bar_height)

        # 主内容区布局
        from arcade import draw_lbwh_rectangle_filled

        if self.demo_recording:
            # 演示/录制模式：左侧信息区 + 右侧一列数量按钮（避免横向重叠）
            left_area, right_area = main_content.split_horizontal(0.72)

            # 左侧信息区
            left_grid = UIGrid(
                left=left_area.l,
                bottom=left_area.b,
                width=left_area.width,
                height=left_area.height,
                rows=7,
                cols=4,
                padding=14,
            )

            prod_x, prod_y = left_grid.pos(0, 0)
            arcade.draw_text(
                f"商品：{product_display_name(prod)}",
                prod_x,
                prod_y,
                WIN98_TEXT_DARK,
                13,
                anchor_x="left",
                anchor_y="center",
            )

            cash_x, cash_y = left_grid.pos(1, 0)
            if mode == "buy":
                cash_text = f"当前现金：{p.cash:,.0f} 元"
            else:
                have = sum(l.quantity for l in p.cargo_lots if l.product_id == pid)
                lost_total = self.state.loss_by_product.get(pid, 0)
                if lost_total > 0:
                    cash_text = f"当前持有：{have} 单位（累计损耗 {lost_total}）"
                else:
                    cash_text = f"当前持有：{have} 单位"
            arcade.draw_text(cash_text, cash_x, cash_y, WIN98_TEXT_DARK, 13, anchor_x="left", anchor_y="center")

            qty_x, qty_y = left_grid.pos(2, 0)
            arcade.draw_text(f"数量（当前 {qty}）：", qty_x, qty_y, WIN98_TEXT_DARK, 13, anchor_x="left", anchor_y="center")

            unit_x, unit_y = left_grid.pos(3, 0)
            arcade.draw_text(f"单价：{unit:.2f} 元", unit_x, unit_y, WIN98_TEXT_DARK, 13, anchor_x="left", anchor_y="center")

            row4_x, row4_y = left_grid.pos(4, 0)
            if mode == "buy":
                arcade.draw_text(f"{summary_label}：{total:,.2f} 元", row4_x, row4_y, WIN98_TEXT_DARK, 13, anchor_x="left", anchor_y="center")
            else:
                arcade.draw_text(f"售出总价：{total_sale_price:,.2f} 元", row4_x, row4_y, WIN98_TEXT_DARK, 13, anchor_x="left", anchor_y="center")

            row5_x, row5_y = left_grid.pos(5, 0)
            if mode == "buy":
                arcade.draw_text(f"购买后余额：{balance_after:,.2f} 元", row5_x, row5_y, WIN98_TEXT_DARK, 13, anchor_x="left", anchor_y="center")
            else:
                arcade.draw_text(f"{summary_label}：{total:,.2f} 元", row5_x, row5_y, WIN98_TEXT_DARK, 13, anchor_x="left", anchor_y="center")

            ok_x, ok_y = left_grid.pos(6, 1)
            cancel_x, cancel_y = left_grid.pos(6, 2)
            self._draw_win98_button("order_confirm", ok_x, ok_y, 140, 38, "确认", pressed=False)
            self._draw_win98_button("order_cancel", cancel_x, cancel_y, 140, 38, "取消", pressed=False)

            # 右侧按钮栏：单独一列
            right_grid = UIGrid(
                left=right_area.l,
                bottom=right_area.b,
                width=right_area.width,
                height=right_area.height,
                rows=10,
                cols=1,
                padding=12,
            )
            title_x, title_y = right_grid.pos(0, 0)
            arcade.draw_text("数量档位", title_x, title_y, WIN98_TEXT_DARK, 13, anchor_x="center", anchor_y="center", bold=True)

            btn_w = min(160, right_area.width * 0.92)
            btn_h = 30
            x, y = right_grid.pos(1, 0)
            self._draw_win98_button("order_qty_f1", x, y, btn_w, btn_h, "1/5", pressed=False)
            x, y = right_grid.pos(2, 0)
            self._draw_win98_button("order_qty_f2", x, y, btn_w, btn_h, "2/5", pressed=False)
            x, y = right_grid.pos(3, 0)
            self._draw_win98_button("order_qty_f3", x, y, btn_w, btn_h, "3/5", pressed=False)
            x, y = right_grid.pos(4, 0)
            self._draw_win98_button("order_qty_f4", x, y, btn_w, btn_h, "4/5", pressed=False)
            x, y = right_grid.pos(5, 0)
            self._draw_win98_button("order_qty_f5", x, y, btn_w, btn_h, "5/5", pressed=False)
            x, y = right_grid.pos(6, 0)
            self._draw_win98_button("order_qty_1", x, y, btn_w, btn_h, "1", pressed=False)
            x, y = right_grid.pos(7, 0)
            self._draw_win98_button("order_qty_2", x, y, btn_w, btn_h, "2", pressed=False)
            x, y = right_grid.pos(8, 0)
            self._draw_win98_button("order_qty_3", x, y, btn_w, btn_h, "3", pressed=False)
        else:
            # 普通模式：原有网格（带输入框与 ±1）
            main_grid = UIGrid(
                left=main_content.l,
                bottom=main_content.b,
                width=main_content.width,
                height=main_content.height,
                rows=7,
                cols=4,
                padding=12,
            )

            # 第0行：商品名
            prod_x, prod_y = main_grid.pos(0, 0)
            arcade.draw_text(
                f"商品：{product_display_name(prod)}",
                prod_x,
                prod_y,
                WIN98_TEXT_DARK,
                13,
                anchor_x="left",
                anchor_y="center",
            )

            # 第1行：当前现金/持有
            cash_x, cash_y = main_grid.pos(1, 0)
            if mode == "buy":
                cash_text = f"当前现金：{p.cash:,.0f} 元"
            else:
                have = sum(l.quantity for l in p.cargo_lots if l.product_id == pid)
                lost_total = self.state.loss_by_product.get(pid, 0)
                if lost_total > 0:
                    cash_text = f"当前持有：{have} 单位（累计损耗 {lost_total}）"
                else:
                    cash_text = f"当前持有：{have} 单位"
            arcade.draw_text(
                cash_text,
                cash_x,
                cash_y,
                WIN98_TEXT_DARK,
                13,
                anchor_x="left",
                anchor_y="center",
            )

            # 第2行：数量控制（标签 + -1按钮 + 输入框 + +1按钮）
            qty_label_x, qty_label_y = main_grid.pos(2, 0)
            arcade.draw_text("数量：", qty_label_x, qty_label_y, WIN98_TEXT_DARK, 13, anchor_x="left", anchor_y="center")

            minus_x, minus_y = main_grid.pos(2, 1)
            self._draw_win98_button("order_qty_minus", minus_x, minus_y, 50, 26, "-1", pressed=False)

            input_x, input_y = main_grid.pos(2, 2)
            input_width = min(main_grid.cell_w * 0.85, 90)
            input_height = 26
            input_left = input_x - input_width / 2
            input_bottom = input_y - input_height / 2
            self.button_regions["order_input"] = (
                input_left,
                input_bottom,
                input_left + input_width,
                input_bottom + input_height,
            )
            draw_lbwh_rectangle_filled(input_left, input_bottom, input_width, input_height, WIN98_BUTTON_FACE)
            self._draw_win98_3d_border(input_left, input_bottom, input_width, input_height, raised=not self.market_order_focused)
            raw_text = self.market_order_dialog.get("text", "")
            if not raw_text and not self.market_order_focused:
                display_text = "0"
                text_color = WIN98_BUTTON_DARK
            else:
                display_text = raw_text or "0"
                text_color = WIN98_TEXT_DARK
            arcade.draw_text(display_text, input_x, input_y - 1, text_color, 13, anchor_x="center", anchor_y="center")

            plus_x, plus_y = main_grid.pos(2, 3)
            self._draw_win98_button("order_qty_plus", plus_x, plus_y, 50, 26, "+1", pressed=False)

            # 单价行
            unit_x, unit_y = main_grid.pos(3, 0)
            arcade.draw_text(
                f"单价：{unit:.2f} 元",
                unit_x,
                unit_y,
                WIN98_TEXT_DARK,
                13,
                anchor_x="left",
                anchor_y="center",
            )

            # 花费/售出总价
            row4_x, row4_y = main_grid.pos(4, 0)
            if mode == "buy":
                arcade.draw_text(
                    f"{summary_label}：{total:,.2f} 元",
                    row4_x, row4_y, WIN98_TEXT_DARK, 13, anchor_x="left", anchor_y="center",
                )
            else:
                arcade.draw_text(
                    f"售出总价：{total_sale_price:,.2f} 元",
                    row4_x, row4_y, WIN98_TEXT_DARK, 13, anchor_x="left", anchor_y="center",
                )

            # 余额/利润
            row5_x, row5_y = main_grid.pos(5, 0)
            if mode == "buy":
                arcade.draw_text(
                    f"购买后余额：{balance_after:,.2f} 元",
                    row5_x, row5_y, WIN98_TEXT_DARK, 13, anchor_x="left", anchor_y="center",
                )
            else:
                arcade.draw_text(
                    f"{summary_label}：{total:,.2f} 元",
                    row5_x, row5_y, WIN98_TEXT_DARK, 13, anchor_x="left", anchor_y="center",
                )

            # 确认/取消按钮
            ok_x, ok_y = main_grid.pos(6, 1)
            cancel_x, cancel_y = main_grid.pos(6, 2)
            self._draw_win98_button("order_confirm", ok_x, ok_y, 120, 34, "确认", pressed=False)
            self._draw_win98_button("order_cancel", cancel_x, cancel_y, 120, 34, "取消", pressed=False)

        # 底栏：达到上限时居中显示「达到上限」（网格法预留的 hint_bar）
        if at_limit:
            hint_grid = UIGrid(left=hint_bar.l, bottom=hint_bar.b, width=hint_bar.width, height=hint_bar.height, rows=1, cols=1, padding=0)
            hint_x, hint_y = hint_grid.pos(0, 0)
            arcade.draw_text(
                "达到上限",
                hint_x,
                hint_y,
                (0, 120, 0),
                12,
                anchor_x="center",
                anchor_y="center",
            )
    
    def _draw_loans_dialog(self) -> None:
        """【借贷】弹窗 - 使用UIBox布局"""
        # 为 1/3~3/3 按钮档位增加空间（面积约 4 倍 = 宽高各 *2）
        # 同时做屏幕裁剪，避免超出窗口。
        margin = 40
        width = min(int(900 * 2), int(SCREEN_WIDTH - margin))
        height = min(int(580 * 2), int(SCREEN_HEIGHT - margin))
        
        # 1. 外框：绘制居中的大框
        x1, y1, x2, y2 = self._draw_dialog_window("银行借贷", width, height, help_key="loans")
        
        p = self.state.player
        loc = p.location
        
        # 无银行城市（海岛等）：与 CLI 一致，不提供借贷/还款
        if not CITIES[loc].has_bank:
            center_y = (y1 + y2) / 2
            arcade.draw_text(
                "当前城市无银行服务（CLI 规则：仅大陆城市有银行）。",
                SCREEN_WIDTH / 2, center_y, (255, 193, 7), 15, anchor_x="center", anchor_y="center", bold=True
            )
            return
        
        # 2. 切片：使用UIBox进行区域切分（网格法）；底部预留一栏用于「达到上限」提示
        dialog_box = UIBox(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2, width, height)
        content_box = dialog_box.pad(20)  # 统一内边距
        hint_bar_height = 32
        content_main, loans_hint_bar = content_box.split_vertical(content_box.height - hint_bar_height)
        
        # 垂直方向：概览 + 中间主区域 + 底部规则
        overview_height = 40
        rules_height = 80
        overview_box, below_overview = content_main.split_vertical(overview_height)
        center_height = max(0, below_overview.height - rules_height)
        main_box, rules_box = below_overview.split_vertical(center_height)
        
        # 中间主区域左右分栏
        left_box, right_box = main_box.split_horizontal(0.45)
        
        # 3. 网格化：所有元素放在网格单元格内，不贴边
        
        # 顶部概览（网格法，信息放在左侧）
        overview_grid = UIGrid(left=overview_box.l, bottom=overview_box.b, width=overview_box.width, height=overview_box.height, rows=1, cols=1, padding=0)
        # 净资产 = 总现金 - 总债务，可借金额 ≤ 净资产（显示实际值，可为负）
        total_debt_amount = sum(l.debt_total() for l in self.state.loans)
        net_assets = p.cash - total_debt_amount
        principal_total = total_outstanding_principal(self.state.loans)
        max_loan = max(0.0, net_assets - principal_total)
        _overview_x, overview_y = overview_grid.pos(0, 0)
        arcade.draw_text(
            f"净资产（现金-债务）：{net_assets:,.0f} 元  |  已借本金：{principal_total:,.0f} 元  |  当前可借额度：{max_loan:,.0f} 元",
            left_box.l + 10,
            overview_y,
            WIN98_TEXT_DARK,
            13,
            anchor_x="left",
            anchor_y="center",
        )
        
        # === 左右主区域：严格 6 行对齐 ===
        from arcade import draw_lbwh_rectangle_filled

        # 左侧：申请借贷
        left_grid = UIGrid(left=left_box.l, bottom=left_box.b, width=left_box.width, height=left_box.height, rows=6, cols=2, padding=10)
        right_grid = UIGrid(left=right_box.l, bottom=right_box.b, width=right_box.width, height=right_box.height, rows=6, cols=2, padding=10)

        # 第1行：左右标题
        left_title_y = left_grid.pos(0, 0)[1]
        right_title_y = right_grid.pos(0, 0)[1]
        arcade.draw_text("申请借贷", left_box.l + 10, left_title_y, WIN98_TEXT_DARK, 14, anchor_x="left", anchor_y="center", bold=True)
        arcade.draw_text("当前借贷", right_box.l + 10, right_title_y, WIN98_TEXT_DARK, 14, anchor_x="left", anchor_y="center", bold=True)

        # 第2行：借贷金额 / 还贷金额
        # 左：借贷金额
        label_left_x = left_box.l + 10
        amount_y = left_grid.pos(1, 0)[1]
        arcade.draw_text("借贷金额：", label_left_x, amount_y, WIN98_TEXT_DARK, 13, anchor_x="left", anchor_y="center")
        input_cx, input_cy = left_grid.pos(1, 1)
        input_w = min(200, left_grid.cell_w * 0.92)
        input_h = 26
        input_left = input_cx - input_w / 2
        input_bottom = input_cy - input_h / 2
        self.button_regions["loans_amount_input"] = (input_left, input_bottom, input_left + input_w, input_bottom + input_h)
        draw_lbwh_rectangle_filled(input_left, input_bottom, input_w, input_h, WIN98_BUTTON_FACE)
        self._draw_win98_3d_border(input_left, input_bottom, input_w, input_h, raised=not (self.loans_focused_input == "borrow"))
        raw = self.loans_amount_text.strip()
        display_text = raw or "0"
        text_color = WIN98_TEXT_DARK if raw else WIN98_BUTTON_DARK
        arcade.draw_text(display_text, input_cx, input_cy - 1, text_color, 13, anchor_x="center", anchor_y="center")

        # 右：还贷金额（达到可还上限时自动裁剪到上限）
        right_label_x = right_box.l + 10
        repay_label_y = right_grid.pos(1, 0)[1]
        arcade.draw_text("还贷金额：", right_label_x, repay_label_y, WIN98_TEXT_DARK, 13, anchor_x="left", anchor_y="center")
        max_repay = min(p.cash, total_debt_amount) if self.state.loans else 0.0
        repay_raw = (self.loans_repay_amount_text or "0").strip()
        try:
            current_repay = float(repay_raw.replace(",", "")) if repay_raw else self.loans_repay_amount
        except ValueError:
            current_repay = self.loans_repay_amount
        clamped_repay = max(0.0, min(current_repay, max_repay))
        if clamped_repay != current_repay:
            self.loans_repay_amount = clamped_repay
            self.loans_repay_amount_text = str(int(clamped_repay))
        repay_raw = (self.loans_repay_amount_text or "0").strip()
        repay_input_cx, repay_input_cy = right_grid.pos(1, 1)
        repay_input_w = min(200, right_grid.cell_w * 0.92)
        repay_input_h = 26
        repay_input_left = repay_input_cx - repay_input_w / 2
        repay_input_bottom = repay_input_cy - repay_input_h / 2
        self.button_regions["loans_repay_input"] = (repay_input_left, repay_input_bottom, repay_input_left + repay_input_w, repay_input_bottom + repay_input_h)
        draw_lbwh_rectangle_filled(repay_input_left, repay_input_bottom, repay_input_w, repay_input_h, WIN98_BUTTON_FACE)
        self._draw_win98_3d_border(repay_input_left, repay_input_bottom, repay_input_w, repay_input_h, raised=not (self.loans_focused_input == "repay"))
        arcade.draw_text(
            repay_raw or "0",
            repay_input_cx,
            repay_input_cy - 1,
            WIN98_TEXT_DARK,
            13,
            anchor_x="center",
            anchor_y="center",
        )

        # 第3行：金额调整
        # - 普通模式：-1000/+1000
        # - 演示/录制模式：1/3、2/3、3/3 按钮档位（对齐 CLI/训练）
        if self.demo_recording:
            can_borrow = max_loan > 0
            adjust_cx, adjust_cy = left_grid.pos(2, 1)
            cell_left = adjust_cx - left_grid.cell_w / 2
            cell_bottom = adjust_cy - left_grid.cell_h / 2
            frac_grid = UIGrid(left=cell_left, bottom=cell_bottom, width=left_grid.cell_w, height=left_grid.cell_h, rows=1, cols=3, padding=8)
            b1_x, b1_y = frac_grid.pos(0, 0)
            b2_x, b2_y = frac_grid.pos(0, 1)
            b3_x, b3_y = frac_grid.pos(0, 2)
            btn_w, btn_h = frac_grid.size(scale_w=0.95, scale_h=0.75)
            self._draw_win98_button("loans_borrow_f1", b1_x, b1_y, btn_w, btn_h, "1/3", enabled=can_borrow, pressed=False)
            self._draw_win98_button("loans_borrow_f2", b2_x, b2_y, btn_w, btn_h, "2/3", enabled=can_borrow, pressed=False)
            self._draw_win98_button("loans_borrow_f3", b3_x, b3_y, btn_w, btn_h, "3/3", enabled=can_borrow, pressed=False)

            r_adjust_cx, r_adjust_cy = right_grid.pos(2, 1)
            r_cell_left = r_adjust_cx - right_grid.cell_w / 2
            r_cell_bottom = r_adjust_cy - right_grid.cell_h / 2
            repay_frac_grid = UIGrid(left=r_cell_left, bottom=r_cell_bottom, width=right_grid.cell_w, height=right_grid.cell_h, rows=1, cols=3, padding=8)
            r1_x, r1_y = repay_frac_grid.pos(0, 0)
            r2_x, r2_y = repay_frac_grid.pos(0, 1)
            r3_x, r3_y = repay_frac_grid.pos(0, 2)
            r_btn_w, r_btn_h = repay_frac_grid.size(scale_w=0.95, scale_h=0.75)
            can_repay = max_repay > 0
            self._draw_win98_button("loans_repay_f1", r1_x, r1_y, r_btn_w, r_btn_h, "1/3", enabled=can_repay, pressed=False)
            self._draw_win98_button("loans_repay_f2", r2_x, r2_y, r_btn_w, r_btn_h, "2/3", enabled=can_repay, pressed=False)
            self._draw_win98_button("loans_repay_f3", r3_x, r3_y, r_btn_w, r_btn_h, "3/3", enabled=can_repay, pressed=False)
        else:
            adjust_cx, adjust_cy = left_grid.pos(2, 1)
            cell_left = adjust_cx - left_grid.cell_w / 2
            cell_bottom = adjust_cy - left_grid.cell_h / 2
            adjust_grid = UIGrid(left=cell_left, bottom=cell_bottom, width=left_grid.cell_w, height=left_grid.cell_h, rows=1, cols=2, padding=8)
            minus_x, minus_y = adjust_grid.pos(0, 0)
            plus_x, plus_y = adjust_grid.pos(0, 1)
            btn_w, btn_h = adjust_grid.size(scale_w=0.9, scale_h=0.75)
            can_borrow = max_loan > 0
            self._draw_win98_button("loans_minus", minus_x, minus_y, btn_w, btn_h, "-1000", enabled=can_borrow, pressed=False)
            self._draw_win98_button("loans_plus", plus_x, plus_y, btn_w, btn_h, "+1000", enabled=can_borrow, pressed=False)

            r_adjust_cx, r_adjust_cy = right_grid.pos(2, 1)
            r_cell_left = r_adjust_cx - right_grid.cell_w / 2
            r_cell_bottom = r_adjust_cy - right_grid.cell_h / 2
            repay_adjust_grid = UIGrid(left=r_cell_left, bottom=r_cell_bottom, width=right_grid.cell_w, height=right_grid.cell_h, rows=1, cols=2, padding=8)
            r_minus_x, r_minus_y = repay_adjust_grid.pos(0, 0)
            r_plus_x, r_plus_y = repay_adjust_grid.pos(0, 1)
            r_btn_w, r_btn_h = repay_adjust_grid.size(scale_w=0.9, scale_h=0.75)
            self._draw_win98_button("loans_repay_minus", r_minus_x, r_minus_y, r_btn_w, r_btn_h, "-1000", pressed=False)
            self._draw_win98_button("loans_repay_plus", r_plus_x, r_plus_y, r_btn_w, r_btn_h, "+1000", pressed=False)

        # 第4行：预计每日利息 / 当前借贷汇总
        raw = self.loans_amount_text.strip()
        try:
            current_amount = float(raw.replace(",", "")) if raw else self.loans_amount
        except ValueError:
            current_amount = self.loans_amount

        # 根据额度区间自动裁剪到 [100, max_loan]；可借额度为 0 时强制为 0
        if max_loan > 0:
            clamped = max(100.0, min(current_amount, max_loan))
        else:
            clamped = 0.0

        if clamped != current_amount:
            # 同步内部数值与输入框显示（用户输过大金额时自动调整为可借上限）
            self.loans_amount = clamped
            self.loans_amount_text = str(int(clamped))

        display_amount = clamped
        daily_interest = display_amount * LOAN_DAILY_INTEREST_RATE
        rate_pct = LOAN_DAILY_INTEREST_RATE * 100

        interest_y = left_grid.pos(3, 0)[1]
        arcade.draw_text("预计每日利息：", label_left_x, interest_y, WIN98_TEXT_DARK, 13, anchor_x="left", anchor_y="center")
        arcade.draw_text(
            f"{daily_interest:.2f} 元（按日利率 {rate_pct:.1f}%）",
            left_box.l + left_box.width * 0.35,
            interest_y,
            WIN98_TEXT_DARK,
            13,
            anchor_x="left",
            anchor_y="center",
        )

        summary_y = right_grid.pos(3, 0)[1]
        total_principal = total_outstanding_principal(self.state.loans)
        total_interest = sum(l.accrued_interest for l in self.state.loans)
        if self.state.loans:
            summary_text = f"当前借贷：总本金 {total_principal:,.0f} 元，已计利息 {total_interest:,.0f} 元。"
        else:
            summary_text = "当前无借贷"
        arcade.draw_text(
            summary_text,
            right_label_x,
            summary_y,
            WIN98_TEXT_DARK,
            12,
            anchor_x="left",
            anchor_y="center",
        )

        # 第5行：确认借贷 / 确认还贷按钮（可借额度为 0 时禁用借贷相关按钮）
        borrow_x, borrow_y = left_grid.pos(4, 1)
        self._draw_win98_button("loans_exec_borrow", borrow_x, borrow_y, 140, 34, "确认借贷", enabled=can_borrow, pressed=False)

        repay_btn_x, repay_btn_y = right_grid.pos(4, 1)
        has_loans = bool(self.state.loans)
        self._draw_win98_button(
            "loans_exec_repay",
            repay_btn_x,
            repay_btn_y,
            140,
            34,
            "确认还贷",
            enabled=has_loans,
            pressed=False,
        )

        # 第6行：右侧借贷详情（多条逐行往下排）
        if self.state.loans:
            detail_start_y = right_grid.pos(5, 0)[1]
            detail_y = detail_start_y
            max_rows = 3
            for i, loan in enumerate(self.state.loans[:max_rows]):
                line = f"#{i+1} 本金={loan.principal:.0f} 利息={loan.accrued_interest:.0f}"
                arcade.draw_text(
                    line,
                    right_label_x,
                    detail_y,
                    WIN98_TEXT_DARK,
                    11,
                    anchor_x="left",
                    anchor_y="center",
                    width=right_grid.cell_w * 2 - 10,
                )
                detail_y -= 18
        
        # 底部规则说明（网格法）
        rules_grid = UIGrid(left=rules_box.l, bottom=rules_box.b, width=rules_box.width, height=rules_box.height, rows=3, cols=1, padding=8)
        rules_title_x, rules_title_y = rules_grid.pos(0, 0)
        arcade.draw_text("规则说明：", rules_title_x, rules_title_y, WIN98_TEXT_DARK, 13, anchor_x="left", anchor_y="center", bold=True)
        rule1_x, rule1_y = rules_grid.pos(1, 0)
        rate_pct = LOAN_DAILY_INTEREST_RATE * 100
        arcade.draw_text(f"· 日利率 {rate_pct:.1f}%。", rule1_x, rule1_y, WIN98_BUTTON_DARK, 12, anchor_x="left", anchor_y="center")
        rule2_x, rule2_y = rules_grid.pos(2, 0)
        arcade.draw_text("· 无固定还款期限，可在任意银行一次性归还全部或部分借款。", rule2_x, rule2_y, WIN98_BUTTON_DARK, 12, anchor_x="left", anchor_y="center")
        
        # 底栏：达到可借上限时居中显示「达到上限」（网格法预留的 loans_hint_bar）
        at_limit = max_loan > 0 and display_amount >= max_loan
        if at_limit:
            hint_grid = UIGrid(left=loans_hint_bar.l, bottom=loans_hint_bar.b, width=loans_hint_bar.width, height=loans_hint_bar.height, rows=1, cols=1, padding=0)
            hint_x, hint_y = hint_grid.pos(0, 0)
            arcade.draw_text(
                "达到上限",
                hint_x,
                hint_y,
                (0, 120, 0),
                12,
                anchor_x="center",
                anchor_y="center",
            )
    
    def _draw_repair_dialog(self) -> None:
        """【车厂】弹窗 - 提供维修与买车两种服务（网格法，左右两模块）"""
        from arcade import draw_lbwh_rectangle_filled

        width = 920
        height = 560

        # 1. 外框：绘制居中的大框
        self._draw_dialog_window("车厂", width, height, help_key="repair")

        # 2. 切片：内容区左右分栏 + 底部规则（概览放入左栏顶部）
        dialog_box = UIBox(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2, width, height)
        content_box = dialog_box.pad(24)

        rules_h = 86
        main_h = max(0, content_box.height - rules_h)
        main_box, rules_box = content_box.split_vertical(main_h)

        left_box, right_box = main_box.split_horizontal(0.5)

        p = self.state.player
        truck_count = max(1, int(getattr(p, "truck_count", 1)))
        extra_trucks = max(0, truck_count - 1)
        total_labor = int(DAILY_LABOR_PER_TRUCK * extra_trucks)

        # 左栏顶部：概览一行（紧贴左），其下为维修服务
        overview_h = 32
        overview_box, repair_content_box = left_box.split_vertical(overview_h)
        pad_left = 12
        overview_y = overview_box.b + overview_box.height / 2
        arcade.draw_text(
            f"现金：{p.cash:,.0f} 元  |  车辆：{truck_count} 辆  |  货车总载重：{p.truck_total_capacity} 单位  |  人力成本：额外每辆 {DAILY_LABOR_PER_TRUCK:.0f} 元/天（首辆免），当前共 {total_labor:,.0f} 元/天",
            left_box.l + pad_left,
            overview_y,
            WIN98_TEXT_DARK,
            13,
            anchor_x="left",
            anchor_y="center",
        )

        # ===== 左侧：维修服务（标签列/内容列对齐） =====
        repair_grid = UIGrid(
            left=repair_content_box.l,
            bottom=repair_content_box.b,
            width=repair_content_box.width,
            height=repair_content_box.height,
            rows=8,
            cols=2,
            padding=12,
        )
        label_x = repair_grid.box.l + repair_grid.padding + 2
        value_x = repair_grid.box.l + repair_grid.padding + (repair_grid.cell_w + repair_grid.padding) + 2

        title_y = repair_grid.pos(0, 0)[1]
        arcade.draw_text("维修服务", label_x, title_y, WIN98_TEXT_DARK, 14, anchor_x="left", anchor_y="center", bold=True)

        # 计算维修百分比与费用
        repair_percent = max(0, int(round(100.0 - p.truck_durability)))
        repair_cost = int(TRUCK_REPAIR_COST_BASE * repair_percent * truck_count)
        damage_ratio, time_factor, loss_factor = _truck_damage_factors(p.truck_durability)

        y = repair_grid.pos(1, 0)[1]
        arcade.draw_text("车辆数量：", label_x, y, WIN98_TEXT_DARK, 13, anchor_x="left", anchor_y="center")
        arcade.draw_text(f"{truck_count} 辆", value_x, y, WIN98_TEXT_DARK, 13, anchor_x="left", anchor_y="center")

        y = repair_grid.pos(2, 0)[1]
        arcade.draw_text("当前耐久：", label_x, y, WIN98_TEXT_DARK, 13, anchor_x="left", anchor_y="center")
        arcade.draw_text(f"{p.truck_durability:.0f}%", value_x, y, WIN98_TEXT_DARK, 13, anchor_x="left", anchor_y="center")

        y = repair_grid.pos(3, 0)[1]
        arcade.draw_text("维修百分比：", label_x, y, WIN98_TEXT_DARK, 13, anchor_x="left", anchor_y="center")
        arcade.draw_text(f"{repair_percent}%", value_x, y, WIN98_TEXT_DARK, 13, anchor_x="left", anchor_y="center")

        y = repair_grid.pos(4, 0)[1]
        arcade.draw_text("维修费用：", label_x, y, WIN98_TEXT_DARK, 13, anchor_x="left", anchor_y="center")
        cost_color = (229, 57, 53) if (repair_percent > 0 and p.cash < repair_cost) else WIN98_TEXT_DARK
        arcade.draw_text(f"{repair_cost:,.0f} 元", value_x, y, cost_color, 13, anchor_x="left", anchor_y="center")
        # 费用公式
        y = repair_grid.pos(5, 0)[1]
        arcade.draw_text("费用公式：", label_x, y, WIN98_TEXT_DARK, 12, anchor_x="left", anchor_y="center")
        arcade.draw_text(f"{TRUCK_REPAIR_COST_BASE:.0f} × 维修百分比 × 车辆数", value_x, y, WIN98_TEXT_DARK, 12, anchor_x="left", anchor_y="center")
        # 当前损坏对运输的影响
        y = repair_grid.pos(6, 0)[1]
        time_pct = int(round((time_factor - 1.0) * 100))
        arcade.draw_text("当前损坏影响：", label_x, y, WIN98_TEXT_DARK, 12, anchor_x="left", anchor_y="center")
        arcade.draw_text(
            f"运输时间 +{time_pct}%，货损率 ×{loss_factor:.2f}",
            value_x,
            y,
            WIN98_TEXT_DARK,
            12,
            anchor_x="left",
            anchor_y="center",
        )

        # 维修按钮（内容列居中）
        btn_x, btn_y = repair_grid.pos(7, 1)
        can_repair = repair_percent > 0 and p.cash >= repair_cost
        self._draw_win98_button("factory_repair_exec", btn_x, btn_y, min(180, repair_grid.cell_w * 0.9), 36, "维修至 100%", enabled=can_repair, pressed=False)

        # ===== 右侧：购车服务（数量输入框 + 购买按钮） =====
        buy_grid = UIGrid(left=right_box.l, bottom=right_box.b, width=right_box.width, height=right_box.height, rows=7, cols=2, padding=12)
        label2_x = buy_grid.box.l + buy_grid.padding + 2
        value2_x = buy_grid.box.l + buy_grid.padding + (buy_grid.cell_w + buy_grid.padding) + 2

        title2_y = buy_grid.pos(0, 0)[1]
        arcade.draw_text("购车服务", label2_x, title2_y, WIN98_TEXT_DARK, 14, anchor_x="left", anchor_y="center", bold=True)

        car_price = TRUCK_PURCHASE_PRICE
        buy_qty = max(1, getattr(self, "factory_buy_qty", 1))

        y = buy_grid.pos(1, 0)[1]
        arcade.draw_text("单价：", label2_x, y, WIN98_TEXT_DARK, 13, anchor_x="left", anchor_y="center")
        arcade.draw_text(f"{car_price:,.0f} 元/辆", value2_x, y, WIN98_TEXT_DARK, 13, anchor_x="left", anchor_y="center")

        y = buy_grid.pos(2, 0)[1]
        arcade.draw_text("现有车辆：", label2_x, y, WIN98_TEXT_DARK, 13, anchor_x="left", anchor_y="center")
        arcade.draw_text(f"{truck_count} 辆", value2_x, y, WIN98_TEXT_DARK, 13, anchor_x="left", anchor_y="center")

        y = buy_grid.pos(3, 0)[1]
        arcade.draw_text("当前载重：", label2_x, y, WIN98_TEXT_DARK, 13, anchor_x="left", anchor_y="center")
        arcade.draw_text(f"{p.truck_total_capacity} 单位", value2_x, y, WIN98_TEXT_DARK, 13, anchor_x="left", anchor_y="center")

        y = buy_grid.pos(4, 0)[1]
        arcade.draw_text("购买后载重：", label2_x, y, WIN98_TEXT_DARK, 13, anchor_x="left", anchor_y="center")
        arcade.draw_text(f"{p.truck_total_capacity + TRUCK_CAPACITY_PER_VEHICLE * buy_qty} 单位", value2_x, y, WIN98_TEXT_DARK, 13, anchor_x="left", anchor_y="center")

        # 购买数量：标签 + 输入框
        y = buy_grid.pos(5, 0)[1]
        arcade.draw_text("购买数量：", label2_x, y, WIN98_TEXT_DARK, 13, anchor_x="left", anchor_y="center")
        input_cx, input_cy = buy_grid.pos(5, 1)
        input_w = min(90, buy_grid.cell_w * 0.5)
        input_h = 26
        input_left = value2_x
        input_bottom = input_cy - input_h / 2
        self.button_regions["factory_buy_input"] = (input_left, input_bottom, input_left + input_w, input_bottom + input_h)
        draw_lbwh_rectangle_filled(input_left, input_bottom, input_w, input_h, WIN98_BUTTON_FACE)
        self._draw_win98_3d_border(input_left, input_bottom, input_w, input_h, raised=not self.factory_buy_focused)
        raw_text = getattr(self, "factory_buy_text", "1")
        if not raw_text and not self.factory_buy_focused:
            display_text = "1"
            text_color = WIN98_BUTTON_DARK
        else:
            display_text = raw_text or "1"
            text_color = WIN98_TEXT_DARK
        arcade.draw_text(display_text, input_left + 6, input_cy - 1, text_color, 13, anchor_x="left", anchor_y="center")

        buy_btn_x, buy_btn_y = buy_grid.pos(6, 1)
        total_cost = car_price * buy_qty
        can_buy = p.cash >= total_cost
        self._draw_win98_button("factory_buy", buy_btn_x, buy_btn_y, min(180, buy_grid.cell_w * 0.9), 36, "购买车辆", enabled=can_buy, pressed=False)

        # 底部规则说明（挪到左半边显示）
        half_width = rules_box.width * 0.5
        rules_grid = UIGrid(
            left=rules_box.l,
            bottom=rules_box.b,
            width=half_width,
            height=rules_box.height,
            rows=4,
            cols=1,
            padding=8,
        )
        rx, ry = rules_grid.pos(0, 0)
        arcade.draw_text("规则说明：", rx, ry, WIN98_TEXT_DARK, 13, anchor_x="left", anchor_y="center", bold=True)
        rx, ry = rules_grid.pos(1, 0)
        arcade.draw_text(f"· 购车：每辆 {TRUCK_PURCHASE_PRICE:,.0f} 元；每购入 1 辆，货车总载重 +{TRUCK_CAPACITY_PER_VEHICLE}；每辆车每天 {DAILY_LABOR_PER_TRUCK:.0f} 元人力成本。", rx, ry, WIN98_BUTTON_DARK, 12, anchor_x="left", anchor_y="center")
        rx, ry = rules_grid.pos(2, 0)
        arcade.draw_text(f"· 维修：费用 = {TRUCK_REPAIR_COST_BASE:.0f} × 维修百分比 × 车辆数；维修后耐久恢复至 100%。", rx, ry, WIN98_BUTTON_DARK, 12, anchor_x="left", anchor_y="center")
        rx, ry = rules_grid.pos(3, 0)
        arcade.draw_text(f"· 维修：完成一次维修需要 {TRUCK_REPAIR_DAYS} 天时间。", rx, ry, WIN98_BUTTON_DARK, 12, anchor_x="left", anchor_y="center")

        # ===== 二级确认弹窗：确认购买车辆（支持数量） =====
        if self.dialog_data.get("factory_buy_pending"):
            # 仅保留二级弹窗按钮可点击
            self.button_regions.clear()

            confirm_qty = max(1, getattr(self, "factory_buy_qty", 1))
            total_cost = car_price * confirm_qty

            modal_w = 520
            modal_h = 240
            modal = UIBox(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2, modal_w, modal_h)
            draw_lbwh_rectangle_filled(modal.l, modal.b, modal.width, modal.height, WIN98_BUTTON_FACE)
            self._draw_win98_3d_border(modal.l, modal.b, modal.width, modal.height, raised=True)

            modal_content = modal.pad(18)
            modal_grid = UIGrid(left=modal_content.l, bottom=modal_content.b, width=modal_content.width, height=modal_content.height, rows=5, cols=4, padding=12)

            tx, ty = modal_grid.pos(0, 0, colspan=4)
            arcade.draw_text("确认购买车辆？", tx, ty, WIN98_TEXT_DARK, 14, anchor_x="center", anchor_y="center", bold=True)

            lx, ly = modal_grid.pos(1, 0, colspan=4)
            arcade.draw_text(f"购买 {confirm_qty} 辆，共花费 {total_cost:,.0f} 元，货车总载重 +{TRUCK_CAPACITY_PER_VEHICLE * confirm_qty}。", lx, ly, WIN98_TEXT_DARK, 13, anchor_x="center", anchor_y="center")

            lx, ly = modal_grid.pos(2, 0, colspan=4)
            after_cash = p.cash - total_cost
            arcade.draw_text(f"当前现金：{p.cash:,.0f} 元  →  购买后：{after_cash:,.0f} 元", lx, ly, WIN98_TEXT_DARK, 12, anchor_x="center", anchor_y="center")

            lx, ly = modal_grid.pos(3, 0, colspan=4)
            arcade.draw_text(f"当前载重：{p.truck_total_capacity}  →  购买后：{p.truck_total_capacity + TRUCK_CAPACITY_PER_VEHICLE * confirm_qty}", lx, ly, WIN98_TEXT_DARK, 12, anchor_x="center", anchor_y="center")

            ok_x, ok_y = modal_grid.pos(4, 1)
            cancel_x, cancel_y = modal_grid.pos(4, 2)
            self._draw_win98_button("factory_buy_confirm", ok_x, ok_y, 120, 34, "确认", pressed=False)
            self._draw_win98_button("factory_buy_cancel", cancel_x, cancel_y, 120, 34, "取消", pressed=False)
    
    def _draw_sail_dialog(self) -> None:
        """【出海】弹窗 - 与出行相同的一分为三布局：左目的地列表，中普通出海，右快速出海。"""
        width = 950
        height = 620
        x1, y1, x2, y2 = self._draw_dialog_window("出海", width, height, help_key="sail")
        dialog_box = UIBox(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2, width, height)
        content_box = dialog_box.pad(40)
        p = self.state.player
        loc = p.location

        if loc not in LAND_SEA_CITIES and loc not in SEA_CITIES:
            cx = content_box.l + content_box.width / 2
            cy = content_box.b + content_box.height / 2
            arcade.draw_text("请先前往海港城市（上海/福州/广州/深圳或海岛）使用出海。", cx, cy, (255, 193, 7), 14, anchor_x="center", anchor_y="center", bold=True)
            cancel_x = content_box.l + content_box.width / 2 - 45
            cancel_y = content_box.b + 40
            self._draw_win98_button("sail_cancel", cancel_x, cancel_y, 90, 28, "关闭", pressed=False)
            return

        # 与出行一致：左 45% 目的地，右 55% 再分为中（普通）右（快速）各半
        left_box, right_outer_box = content_box.split_horizontal(0.45)
        normal_box, fast_box = right_outer_box.split_horizontal(0.5)
        normal_grid = normal_box.make_grid(12, 1, gap=15)
        fast_grid = fast_box.make_grid(12, 1, gap=15)

        dests = self._sail_destinations()
        city_grid_rows = max(15, min(28, len(dests) + 2))
        city_grid = left_box.make_grid(city_grid_rows, 1, gap=8)
        arcade.draw_text("选择目的地", left_box.l + 10, left_box.t - 20, WIN98_TEXT_DARK, 14, anchor_y="top", bold=True)
        self.sail_row_regions = []
        for i, city in enumerate(dests):
            if i >= city_grid.rows - 1:
                break
            text_x, text_y = city_grid.pos(i + 1, 0)
            try:
                validate_mode_allowed(TransportMode.SEA, loc, city)
                km = route_km(TransportMode.SEA, loc, city)
            except RouteNotFound:
                info = "  耗时: --天  费用: --元"
            else:
                days = sample_travel_days(TransportMode.SEA, km, random.Random(hash((loc, city)) & 0xFFFFFFFF))
                is_taiwan = (loc in ("台北", "高雄")) ^ (city in ("台北", "高雄"))
                customs = TAIWAN_CUSTOMS if is_taiwan else 0.0
                units = current_cargo_units(p)
                total_cap = total_storage_capacity(p, loc)
                base = km * self.sea_cost_per_km + customs
                load_mult = 1.0 + (units / max(1, total_cap))
                cost = base * load_mult
                info = f"  耗时: {days}天  费用: {cost:.0f}元"
            self.sail_row_regions.append((city, (left_box.l, text_y - 14, left_box.l + left_box.width, text_y + 14)))
            arcade.draw_text(city + info, left_box.l + 10, text_y, WIN98_TEXT_DARK, 11, anchor_y="center")

        if self.sail_target and self.sail_target in dests:
            target = self.sail_target
            try:
                validate_mode_allowed(TransportMode.SEA, loc, target)
                km = route_km(TransportMode.SEA, loc, target)
            except RouteNotFound:
                arcade.draw_text("无法到达该目的地。", normal_box.l + 10, normal_grid.pos(2, 0)[1], (229, 57, 53), 13, anchor_y="center")
            else:
                days = sample_travel_days(TransportMode.SEA, km, random.Random(hash((loc, target)) & 0xFFFFFFFF))
                is_taiwan = (loc in ("台北", "高雄")) ^ (target in ("台北", "高雄"))
                customs = TAIWAN_CUSTOMS if is_taiwan else 0.0
                units = current_cargo_units(p)
                total_cap = total_storage_capacity(p, loc)
                base_cost = km * self.sea_cost_per_km + customs
                load_mult = 1.0 + (units / max(1, total_cap))
                cost_normal = base_cost * load_mult
                arcade.draw_text(f"目的地：{target}", normal_box.l + 10, normal_box.t - 40, WIN98_TEXT_DARK, 16, anchor_y="top", bold=True)
                info_items = [
                    ("出行方式", "海运"),
                    ("预计里程", f"{km} km"),
                    ("预计耗时", f"{days} 天"),
                    ("基础运输成本", f"{cost_normal:.0f} 元"),
                ]
                for i, (label, value) in enumerate(info_items):
                    text_x, text_y = normal_grid.pos(i + 2, 0)
                    arcade.draw_text(f"{label}：{value}", normal_box.l + 10, text_y, WIN98_TEXT_DARK, 13, anchor_y="center")
                loss_x, loss_y = normal_grid.pos(6, 0)
                arcade.draw_text("预计运输损耗：", normal_box.l + 10, loss_y, WIN98_TEXT_DARK, 11, anchor_y="center")
                loss_lines = expected_transport_loss_display(km, days).splitlines()
                cat_y = loss_y - 16
                for line in loss_lines:
                    arcade.draw_text(f"  {line}", normal_box.l + 20, cat_y, WIN98_TEXT_DARK, 11, anchor_y="center")
                    cat_y -= 14
                perishable_details = expected_perishable_loss_details(km, days)
                if perishable_details:
                    detail_y = cat_y - 18
                    for name, pct in perishable_details:
                        arcade.draw_text(f"  - {name}：约 {pct:.1f}%", normal_box.l + 20, detail_y, WIN98_TEXT_DARK, 11, anchor_y="center")
                        detail_y -= 16
                confirm_x, confirm_y = normal_grid.pos(10, 0)
                self._draw_win98_button("sail_confirm_normal", confirm_x, confirm_y, 130, 32, "普通出海", pressed=False)

                fast_days = max(FAST_TRAVEL_MIN_DAYS, int(days // FAST_TRAVEL_TIME_DIVISOR))
                fast_cost = cost_normal * FAST_TRAVEL_COST_MULTIPLIER
                f_items = [
                    ("快速出行", "海运"),
                    ("预计里程", f"{km} km"),
                    ("预计耗时", f"{fast_days} 天"),
                    ("快速运输成本", f"{fast_cost:.0f} 元"),
                ]
                for i, (label, value) in enumerate(f_items):
                    fx, fy = fast_grid.pos(i + 2, 0)
                    arcade.draw_text(f"{label}：{value}", fast_box.l + 10, fy, WIN98_TEXT_DARK, 13, anchor_y="center")
                f_loss_x, f_loss_y = fast_grid.pos(6, 0)
                arcade.draw_text("预计运输损耗：", fast_box.l + 10, f_loss_y, WIN98_TEXT_DARK, 11, anchor_y="center")
                f_loss_lines = expected_transport_loss_display(km, fast_days).splitlines()
                f_cat_y = f_loss_y - 16
                for line in f_loss_lines:
                    arcade.draw_text(f"  {line}", fast_box.l + 20, f_cat_y, WIN98_TEXT_DARK, 11, anchor_y="center")
                    f_cat_y -= 14
                f_perishable = expected_perishable_loss_details(km, fast_days)
                if f_perishable:
                    f_detail_y = f_cat_y - 18
                    for name, pct in f_perishable:
                        arcade.draw_text(f"  - {name}：约 {pct:.1f}%", fast_box.l + 20, f_detail_y, WIN98_TEXT_DARK, 11, anchor_y="center")
                        f_detail_y -= 16
                f_confirm_x, f_confirm_y = fast_grid.pos(10, 0)
                self._draw_win98_button("sail_confirm_fast", f_confirm_x, f_confirm_y, 130, 32, "快速出海", pressed=False)
        else:
            arcade.draw_text("请在左侧选择目的地。", normal_box.l + 10, normal_grid.pos(2, 0)[1], WIN98_TEXT_DARK, 13, anchor_y="center")
        cancel_x, cancel_y = normal_grid.pos(11, 0)
        self._draw_win98_button("sail_cancel", cancel_x, cancel_y, 90, 28, "取消", pressed=False)

    def _draw_save_dialog(self) -> None:
        """【存档】弹窗 - 使用UIBox布局"""
        width = 850
        height = 560
        
        # 1. 外框：绘制居中的大框
        x1, y1, x2, y2 = self._draw_dialog_window("保存游戏", width, height, help_key="save")
        
        # 2. 切片：使用UIBox进行区域切分
        dialog_box = UIBox(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2, width, height)
        
        # 内容区域
        content_box = dialog_box.pad(40)
        
        # 左右分栏
        left_box, right_box = content_box.split_horizontal(0.45)
        
        # === 左侧：存档槽列表 ===
        left_grid = left_box.make_grid(8, 1, gap=15)
        
        # 左侧标题
        arcade.draw_text("存档槽位", left_box.l + 10, left_box.t - 15, WIN98_TEXT_DARK, 14, anchor_y="top", bold=True)
        
        # 存档槽列表
        for i in range(5):
            slot_name = f"存档 {i+1}"
            
            # 尝试读取存档信息
            try:
                slot_state = load_game(f"slot_{i+1}")
                gm = getattr(slot_state, "game_mode", "free")
                mode_label = "挑战" if gm == "challenge" else ("演示" if gm == "demo" else "自由")
                slot_info = f"{mode_label} 第{slot_state.player.day}天 | 现金：{slot_state.player.cash:,.0f} 元"
            except FileNotFoundError:
                slot_info = "空槽位"
            
            # 槽位按钮
            btn_pressed = self.save_slot_selected == i
            btn_x, btn_y = left_grid.pos(i + 1, 0)
            self._draw_win98_button(
                f"save_slot_{i}", btn_x, btn_y, 290, 55, f"{slot_name}\n{slot_info}",
                pressed=btn_pressed
            )
        
        # === 右侧：操作区 ===
        right_grid = right_box.make_grid(8, 1, gap=15)
        
        # 右侧标题
        arcade.draw_text("保存操作", right_box.l + 10, right_box.t - 15, WIN98_TEXT_DARK, 14, anchor_y="top", bold=True)
        
        if self.save_slot_selected is not None:
            slot_name = f"slot_{self.save_slot_selected + 1}"
            try:
                load_game(slot_name)
                has_save = True
                confirm_text = "是否覆盖当前存档？\n此操作无法撤销。"
            except FileNotFoundError:
                has_save = False
                confirm_text = "将创建新存档。"
            
            # 确认文本
            text_x, text_y = right_grid.pos(1, 0)
            arcade.draw_text(confirm_text, right_box.l + 10, text_y + 30, WIN98_TEXT_DARK, 13, anchor_y="top", width=260)
            
            # 保存按钮
            confirm_x, confirm_y = right_grid.pos(3, 0)
            self._draw_win98_button("save_confirm", confirm_x, confirm_y, 130, 36, "确认保存", pressed=False)

            # 删除按钮（仅当该槽位已有存档时可用）
            delete_x, delete_y = right_grid.pos(4, 0)
            self._draw_win98_button("save_delete", delete_x, delete_y, 130, 32, "删除该存档", enabled=has_save, pressed=False)
        
        # 取消按钮
        cancel_x, cancel_y = right_grid.pos(7, 0)
        self._draw_win98_button("save_cancel", cancel_x, cancel_y, 90, 28, "取消", pressed=False)
    
    def _draw_load_dialog(self) -> None:
        """【读档】弹窗 - 使用UIBox布局"""
        width = 850
        height = 560
        
        # 1. 外框：绘制居中的大框
        x1, y1, x2, y2 = self._draw_dialog_window("读取游戏", width, height, help_key="load")
        
        # 2. 切片：使用UIBox进行区域切分
        dialog_box = UIBox(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2, width, height)
        
        # 内容区域
        content_box = dialog_box.pad(40)
        
        # 左右分栏
        left_box, right_box = content_box.split_horizontal(0.45)
        
        # === 左侧：存档槽列表 ===
        left_grid = left_box.make_grid(8, 1, gap=15)
        
        # 左侧标题
        arcade.draw_text("存档槽位", left_box.l + 10, left_box.t - 15, WIN98_TEXT_DARK, 14, anchor_y="top", bold=True)
        
        # 存档槽列表
        for i in range(5):
            slot_name = f"存档 {i+1}"
            
            try:
                slot_state = load_game(f"slot_{i+1}")
                gm = getattr(slot_state, "game_mode", "free")
                mode_label = "挑战" if gm == "challenge" else ("演示" if gm == "demo" else "自由")
                slot_info = f"{mode_label} 第{slot_state.player.day}天 | 现金：{slot_state.player.cash:,.0f} 元"
                has_save = True
            except FileNotFoundError:
                slot_info = "空槽位"
                has_save = False
            
            btn_pressed = self.load_slot_selected == i
            btn_x, btn_y = left_grid.pos(i + 1, 0)
            self._draw_win98_button(
                f"load_slot_{i}", btn_x, btn_y, 290, 55, f"{slot_name}\n{slot_info}",
                pressed=btn_pressed, enabled=has_save
            )
        
        # === 右侧：操作区 ===
        right_grid = right_box.make_grid(8, 1, gap=15)
        
        # 右侧标题
        arcade.draw_text("读取操作", right_box.l + 10, right_box.t - 15, WIN98_TEXT_DARK, 14, anchor_y="top", bold=True)
        
        if self.load_slot_selected is not None:
            try:
                load_game(f"slot_{self.load_slot_selected + 1}")
                confirm_text = "确认读取该存档？\n当前进度将丢失。"
                can_load = True
            except FileNotFoundError:
                confirm_text = "该槽位暂无存档。"
                can_load = False
            
            # 确认文本
            text_x, text_y = right_grid.pos(1, 0)
            arcade.draw_text(confirm_text, right_box.l + 10, text_y + 30, WIN98_TEXT_DARK, 13, anchor_y="top", width=260)
            
            # 确认按钮
            confirm_x, confirm_y = right_grid.pos(3, 0)
            self._draw_win98_button("load_confirm", confirm_x, confirm_y, 130, 36, "确认读取", enabled=can_load, pressed=False)
        
        # 取消按钮
        cancel_x, cancel_y = right_grid.pos(7, 0)
        self._draw_win98_button("load_cancel", cancel_x, cancel_y, 90, 28, "取消", pressed=False)

    # --- Arcade 回调 ---

    def _draw_mode_select(self) -> None:
        """模式选择界面：挑战模式 / 自由模式 / 读档"""
        from arcade import draw_lbwh_rectangle_filled

        # 浅色主题背景
        draw_lbwh_rectangle_filled(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT, WIN98_BUTTON_FACE)
        cx, cy = SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2
        
        # 标题和提示文字（深色）
        arcade.draw_text("风物千程", cx, cy + 120, WIN98_TEXT_DARK, 36, anchor_x="center", anchor_y="center", bold=True)
        arcade.draw_text("请选择游戏模式", cx, cy + 70, WIN98_TEXT_DARK, 18, anchor_x="center", anchor_y="center")

        # 按钮居中，宽度足够框住文字
        btn_w, btn_h = 450, 60
        self._draw_win98_button("mode_challenge", cx, cy + 10, btn_w, btn_h, f"挑战模式\n{CHALLENGE_DAYS}天时间上限，根据总资产评分", pressed=False)
        self._draw_win98_button("mode_free", cx, cy - 60, btn_w, btn_h, "自由模式\n无时间限制，自由经营", pressed=False)
        self._draw_win98_button("mode_load", cx, cy - 130, 200, 40, "读档", pressed=False)

    def _draw_start_cash_dialog(self) -> None:
        """自由模式：设置初始金额的小弹窗"""
        from arcade import draw_lbwh_rectangle_filled

        width = 520
        height = 260
        self._draw_dialog_window("自由模式 - 初始金额", width, height)

        dialog_box = UIBox(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2, width, height)
        content = dialog_box.pad(26)

        arcade.draw_text(
            "请输入自由模式初始金额（元）：",
            content.l,
            content.t - 10,
            WIN98_TEXT_DARK,
            14,
            anchor_y="top",
        )

        grid = UIGrid(left=content.l, bottom=content.b + 70, width=content.width, height=110, rows=2, cols=5, padding=10)

        # 输入框
        input_cx, input_cy = grid.pos(0, 2)
        input_w, input_h = 180, 32
        input_left = input_cx - input_w / 2
        input_bottom = input_cy - input_h / 2
        self.button_regions["start_cash_input"] = (input_left, input_bottom, input_left + input_w, input_bottom + input_h)
        draw_lbwh_rectangle_filled(input_left, input_bottom, input_w, input_h, WIN98_BUTTON_HIGHLIGHT)
        self._draw_win98_3d_border(input_left, input_bottom, input_w, input_h, raised=not self.start_cash_focused)

        txt = (self.start_cash_text or "").strip()
        show = txt if txt else "0"
        arcade.draw_text(show, input_cx, input_cy - 1, WIN98_TEXT_DARK, 14, anchor_x="center", anchor_y="center")

        # 按钮：-10000 / -1000 / +1000 / +10000
        b1x, b1y = grid.pos(0, 0)
        b2x, b2y = grid.pos(0, 1)
        b4x, b4y = grid.pos(0, 3)
        b5x, b5y = grid.pos(0, 4)
        self._draw_win98_button("start_cash_minus_10000", b1x, b1y, 90, 30, "-10000", pressed=False)
        self._draw_win98_button("start_cash_minus_1000", b2x, b2y, 90, 30, "-1000", pressed=False)
        self._draw_win98_button("start_cash_plus_1000", b4x, b4y, 90, 30, "+1000", pressed=False)
        self._draw_win98_button("start_cash_plus_10000", b5x, b5y, 90, 30, "+10000", pressed=False)

        # 底部按钮：确认/取消
        btn_grid = UIGrid(left=content.l, bottom=content.b, width=content.width, height=60, rows=1, cols=3, padding=10)
        ok_x, ok_y = btn_grid.pos(0, 1)
        cancel_x, cancel_y = btn_grid.pos(0, 2)
        self._draw_win98_button("start_cash_confirm", ok_x, ok_y, 140, 34, "确认开始", pressed=False)
        self._draw_win98_button("start_cash_cancel", cancel_x, cancel_y, 100, 34, "取消", pressed=False)

    def _draw_challenge_end(self) -> None:
        """挑战模式结算界面"""
        from arcade import draw_lbwh_rectangle_filled

        # 浅色主题背景
        draw_lbwh_rectangle_filled(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT, WIN98_BUTTON_FACE)
        d = self.challenge_end_data or {}
        days = d.get("days", 0)
        assets = d.get("total_assets", 0)
        rating = d.get("rating", "拉完了")
        bankrupt = d.get("bankrupt", False)

        cx, cy = SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2
        # 文字改为深色
        arcade.draw_text("挑战结束", cx, cy + 100, WIN98_TEXT_DARK, 28, anchor_x="center", anchor_y="center", bold=True)
        if bankrupt:
            arcade.draw_text("提前破产", cx, cy + 50, WIN98_TEXT_DARK, 18, anchor_x="center", anchor_y="center")
        arcade.draw_text(f"结算天数：{days} 天", cx, cy + 10, WIN98_TEXT_DARK, 16, anchor_x="center", anchor_y="center")
        arcade.draw_text(f"结算金额：{assets:,.0f} 元", cx, cy - 30, WIN98_TEXT_DARK, 16, anchor_x="center", anchor_y="center")
        arcade.draw_text(f"评分：{rating}", cx, cy - 70, (200, 0, 0), 22, anchor_x="center", anchor_y="center", bold=True)
        self._draw_win98_button("challenge_restart", cx, cy - 140, 160, 40, "重新开始", pressed=False)

    def on_draw(self) -> None:  # type: ignore[override]
        """主界面绘制：上信息框+下按钮区"""
        from arcade import draw_lbwh_rectangle_filled
        
        self.clear()
        self.button_regions.clear()
        self.city_row_regions.clear()
        
        # --- 模式选择界面 ---
        if self.current_screen == "mode_select":
            self._draw_mode_select()
            if self.active_dialog == "load":
                self._draw_load_dialog()
            elif self.active_dialog == "start_cash":
                # 仅保留弹窗按钮可点击，屏蔽底层模式选择按钮（避免区域重叠误触发）
                self.button_regions.clear()
                self._draw_start_cash_dialog()
            if self.help_popup_text:
                self._draw_help_popup()
            return

        # --- 挑战结算界面 ---
        if self.current_screen == "challenge_end":
            self._draw_challenge_end()
            return

        # --- 1. 绘制背景 ---
        draw_lbwh_rectangle_filled(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT, WIN98_BG_DARK)
        
        # --- 2. 区域划分 ---
        INFO_HEIGHT = 480
        BUTTON_HEIGHT = SCREEN_HEIGHT - INFO_HEIGHT - 60
        LOG_HEIGHT = 60
        
        # --- 3. 上部分：大信息框 ---
        info_box = UIBox(SCREEN_WIDTH / 2, SCREEN_HEIGHT - INFO_HEIGHT / 2, SCREEN_WIDTH - 50, INFO_HEIGHT)
        
        # 绘制信息框背景和边框
        draw_lbwh_rectangle_filled(info_box.l, info_box.b, info_box.width, info_box.height, WIN98_BUTTON_FACE)
        self._draw_win98_3d_border(info_box.l, info_box.b, info_box.width, info_box.height, raised=True)
        
        # 信息框内边距
        info_content = info_box.pad(35)
        
        # 信息框分为左右两列
        left_col, right_col = info_content.split_horizontal(0.5)
        
        # --- 左侧列：基本信息和货物 ---
        left_grid = left_col.make_grid(15, 1, gap=12)
        p = self.state.player
        
        # 左侧标题
        arcade.draw_text("【基本信息】", left_col.l, left_col.t - 10, WIN98_TEXT_DARK, 14, anchor_y="top", bold=True)
        
        # 时间和地点
        gm = getattr(self.state, "game_mode", "free")
        if gm == "challenge":
            day_text = f"第{p.day}/{CHALLENGE_DAYS}天"
        elif gm == "demo":
            day_text = f"第{p.day}/{DEMO_MODE_DAYS}天"
        else:
            day_text = f"Day {p.day}"
        arcade.draw_text(f"{day_text}  |  地点：{p.location}", left_col.l + 15, left_col.t - 40, WIN98_TEXT_DARK, 13, anchor_y="top")
        
        # 资金情况
        arcade.draw_text(f"现金：{p.cash:,.0f} 元", left_col.l + 15, left_col.t - 70, WIN98_CASH_HIGHLIGHT, 14, anchor_y="top", bold=True)
        
        # 借贷情况
        total_debt = sum(l.debt_total() for l in self.state.loans)
        arcade.draw_text(f"当前借贷：{total_debt:,.0f} 元", left_col.l + 15, left_col.t - 100, WIN98_TEXT_DARK, 12, anchor_y="top")
        
        # 车辆状态
        arcade.draw_text("【车辆状态】", left_col.l, left_col.t - 130, WIN98_TEXT_DARK, 14, anchor_y="top", bold=True)
        arcade.draw_text(f"货车耐久：{p.truck_durability:.0f}%", left_col.l + 15, left_col.t - 160, WIN98_TEXT_DARK, 12, anchor_y="top")
        truck_count = max(1, int(getattr(p, "truck_count", 1)))
        extra_trucks = max(0, truck_count - 1)
        daily_labor_cost = int(DAILY_LABOR_PER_TRUCK * extra_trucks)
        arcade.draw_text(
            f"车辆每日人力成本：{daily_labor_cost:,.0f} 元/天（额外 {extra_trucks} 辆；首辆车免）",
            left_col.l + 15,
            left_col.t - 185,
            WIN98_TEXT_DARK,
            12,
            anchor_y="top",
        )
        
        
        # 货物列表
        arcade.draw_text("【当前货物】", left_col.l, left_col.t - 240, WIN98_TEXT_DARK, 14, anchor_y="top", bold=True)
        
        # 货物表头：商品名与【当前货物】左对齐；数量/采购价/当前售价按列排布
        cargo_header_y = left_col.t - 270
        arcade.draw_text("商品名", left_col.l, cargo_header_y, WIN98_TEXT_DARK, 12, anchor_y="top", bold=True)
        arcade.draw_text("数量", left_col.l + 220, cargo_header_y, WIN98_TEXT_DARK, 12, anchor_y="top", bold=True)
        arcade.draw_text("采购价", left_col.l + 310, cargo_header_y, WIN98_TEXT_DARK, 12, anchor_y="top", bold=True)
        arcade.draw_text("当前售价", left_col.l + 400, cargo_header_y, WIN98_TEXT_DARK, 12, anchor_y="top", bold=True)
        
        # 货物内容
        cargo_y = cargo_header_y - 25
        inv: dict[str, int] = {}
        for lot in p.cargo_lots:
            inv[lot.product_id] = inv.get(lot.product_id, 0) + lot.quantity
        
        for pid, qty in inv.items():
            if cargo_y < left_col.b + 20:
                break
            prod = PRODUCTS.get(pid)
            if prod:
                buy_price = purchase_price(prod, p.location, self.state.daily_lambdas)
                sell_price = sell_unit_price(prod, p.location, self.state.daily_lambdas, quantity_sold=1)
                arcade.draw_text(product_display_name(prod), left_col.l, cargo_y, WIN98_TEXT_DARK, 11, anchor_y="top")
                arcade.draw_text(f"{qty}", left_col.l + 220, cargo_y, WIN98_TEXT_DARK, 11, anchor_y="top")
                if buy_price is not None:
                    arcade.draw_text(f"{buy_price:.2f}", left_col.l + 310, cargo_y, WIN98_TEXT_DARK, 11, anchor_y="top")
                else:
                    arcade.draw_text("--", left_col.l + 310, cargo_y, WIN98_BUTTON_DARK, 11, anchor_y="top")
                arcade.draw_text(f"{sell_price:.2f}", left_col.l + 400, cargo_y, WIN98_TEXT_DARK, 11, anchor_y="top")
                cargo_y -= 20
        
        # --- 右侧列：资产概览 + 手记 ---
        assets_box = right_col.top_slice(160)
        notes_box = right_col.bottom_slice(right_col.height - 160)
        
        # --- 资产概览 ---
        arcade.draw_text("【资产概览】", assets_box.l, assets_box.t - 10, WIN98_TEXT_DARK, 14, anchor_y="top", bold=True)
        
        # 预估总资产（现金 + 货物价值）
        assets = estimated_assets(p.cash, p.cargo_lots)
        arcade.draw_text(f"预估总资产：{assets:,.0f} 元", assets_box.l + 15, assets_box.t - 40, WIN98_TEXT_DARK, 13, anchor_y="top")
        
        # 净资产 = 总现金 - 总债务（显示实际值，可为负）
        net_assets = p.cash - total_debt
        arcade.draw_text(f"净资产：{net_assets:,.0f} 元", assets_box.l + 15, assets_box.t - 70, WIN98_TEXT_DARK, 13, anchor_y="top")
        
        # 载重信息
        cargo_used_val = cargo_used(p.cargo_lots)
        cargo_max = total_storage_capacity(p, p.location)
        cargo_percent = int(cargo_used_val / cargo_max * 100) if cargo_max > 0 else 0
        arcade.draw_text(f"载重：{cargo_used_val} / {cargo_max}  ({cargo_percent}%)", assets_box.l + 15, assets_box.t - 100, WIN98_TEXT_DARK, 12, anchor_y="top")
        
        # --- 文本框：右侧【机会】（价格信息单独用弹窗，不占用此区域） ---
        draw_lbwh_rectangle_filled(notes_box.l, notes_box.b, notes_box.width, notes_box.height, WIN98_BG_LIGHT)
        self._draw_win98_3d_border(notes_box.l, notes_box.b, notes_box.width, notes_box.height, raised=False)
        
        notes_content = notes_box.pad(20)
        line_height = 20

        # 每帧自动刷新价格机会提示（基于最近 7 日趋势）
        self._update_price_notes_only()

        # 顶部右侧：价格信息按钮
        btn_px = notes_content.l + notes_content.width - 90
        btn_py = notes_content.t - 16
        self._draw_win98_button("btn_price_info", btn_px, btn_py, 80, 24, "价格信息", pressed=False)

        # 文本内容：仅显示【机会】（若无机会则显示占位提示）
        lines: List[str] = []
        if self.price_note_lines:
            lines.extend(self.price_note_lines)
        else:
            lines.append("【机会】")
            lines.append("当前暂无明显的连续涨跌机会。")

        # 逐行绘制，若超过框底部则停止，避免溢出
        max_y = notes_content.t - 30
        y = max_y
        for text in lines:
            if y < notes_content.b + 10:
                break
            arcade.draw_text(
                text,
                notes_content.l,
                y,
                WIN98_TEXT_LIGHT,
                12,
                anchor_y="top",
            )
            y -= line_height
        
        # --- 4. 下部分：按钮区（仅在无弹窗时显示）---
        if not self.active_dialog:
            button_box = UIBox(SCREEN_WIDTH / 2, BUTTON_HEIGHT / 2 + LOG_HEIGHT, SCREEN_WIDTH - 50, BUTTON_HEIGHT)
            button_grid = UIGrid(
                left=button_box.l,
                bottom=button_box.b,
                width=button_box.width,
                height=button_box.height,
                rows=2, cols=4,
                padding=20
            )
            btn_w, btn_h = button_grid.size(scale_w=0.85, scale_h=0.7)
            
            # 录制模式下对齐 CLI：借贷仅在有银行城市可用，出海仅在港口城市可用
            can_loan = self._cli_can_loan() if self.demo_recording else True
            can_sail = self._cli_can_sail() if self.demo_recording else True
            
            # 第一行按钮
            self._draw_win98_button("btn_travel", *button_grid.pos(0, 0), btn_w, btn_h, "出行 (A)", pressed=False)
            self._draw_win98_button("btn_market", *button_grid.pos(0, 1), btn_w, btn_h, "市场 (M)", pressed=False)
            self._draw_win98_button("btn_loans", *button_grid.pos(0, 2), btn_w, btn_h, "借贷 (J)", enabled=can_loan, pressed=False)
            self._draw_win98_button("btn_repair_truck", *button_grid.pos(0, 3), btn_w, btn_h, "车厂 (R)", pressed=False)
            
            # 第二行按钮
            self._draw_win98_button("btn_sail", *button_grid.pos(1, 0), btn_w, btn_h, "出海 (T)", enabled=can_sail, pressed=False)
            self._draw_win98_button("btn_next", *button_grid.pos(1, 1), btn_w, btn_h, "下一天 (N)")
            self._draw_win98_button("btn_save", *button_grid.pos(1, 2), btn_w, btn_h, "存档 (F5)", pressed=False)
            self._draw_win98_button("btn_load", *button_grid.pos(1, 3), btn_w, btn_h, "读档 (F9)", pressed=False)

        # --- 5. 底部日志栏 ---
        log_box = UIBox(SCREEN_WIDTH / 2, LOG_HEIGHT / 2, SCREEN_WIDTH, LOG_HEIGHT)
        draw_lbwh_rectangle_filled(log_box.l, log_box.b, log_box.width, log_box.height, WIN98_BG_LIGHT)
        self._draw_win98_3d_border(log_box.l, log_box.b, log_box.width, log_box.height, raised=False)
        
        if self.log:
            # 减小内边距与顶部偏移，使首行几乎贴近小框上边线
            log_content = log_box.pad(10)
            latest_msg = self.log[-1] if len(self.log) >= 1 else ""
            # 主行：通常是“到达 … 用时 … 成本 …”
            y_main = log_content.t - 4
            arcade.draw_text(latest_msg[:180], log_content.l, y_main, WIN98_TEXT_LIGHT, 12, anchor_y="top")

            # 次行：若上一条有“运输损耗”信息，则简要显示在下一行
            if len(self.log) >= 2:
                prev = self.log[-2]
                if prev.startswith("运输损耗："):
                    y_sub = y_main - 18
                    arcade.draw_text(prev[:180], log_content.l, y_sub, WIN98_TEXT_LIGHT, 11, anchor_y="top")

        # --- 7. 绘制弹窗 (如果有) ---
        if self.active_dialog == "travel":
            self._draw_travel_dialog()
        elif self.active_dialog == "market":
            self._draw_market_dialog()
        elif self.active_dialog == "loans":
            self._draw_loans_dialog()
        elif self.active_dialog == "repair":
            self._draw_repair_dialog()
        elif self.active_dialog == "sail":
            self._draw_sail_dialog()
        elif self.active_dialog == "save":
            self._draw_save_dialog()
        elif self.active_dialog == "load":
            self._draw_load_dialog()
        elif self.active_dialog == "price_info":
            self._draw_price_info_dialog()

        # --- 8. 帮助弹窗（覆盖在最上层）---
        if self.help_popup_text:
            self._draw_help_popup()

    def _draw_help_popup(self) -> None:
        """绘制帮助说明小窗（覆盖在当前弹窗上方）"""
        from arcade import draw_lbwh_rectangle_filled

        text = self.help_popup_text or ""
        lines = text.split("\n")
        line_height = 20
        box_w = min(440, SCREEN_WIDTH - 80)
        box_h = min(len(lines) * line_height + 80, 420)

        cx, cy = SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2
        x1 = cx - box_w / 2
        y1 = cy - box_h / 2
        x2 = cx + box_w / 2
        y2 = cy + box_h / 2

        draw_lbwh_rectangle_filled(x1, y1, box_w, box_h, WIN98_BUTTON_FACE)
        self._draw_win98_3d_border(x1, y1, box_w, box_h, raised=True)

        title_bar_h = 28
        title_y = y2 - title_bar_h
        draw_lbwh_rectangle_filled(x1, title_y, box_w, title_bar_h, WIN98_TITLE_BAR)
        arcade.draw_text("帮助说明", x1 + 12, title_y + 8, WIN98_BUTTON_HIGHLIGHT, 13, anchor_y="bottom")

        close_btn_w, close_btn_h = 70, 26
        close_y = y1 + 14
        close_x = x2 - close_btn_w - 14
        self._draw_win98_button("help_popup_close", close_x, close_y, close_btn_w, close_btn_h, "关闭", pressed=False)

        content_top = title_y - 20
        for i, line in enumerate(lines):
            arcade.draw_text(
                line, x1 + 16, content_top - i * line_height,
                WIN98_TEXT_DARK, 12, anchor_y="top", width=box_w - 32
            )

    def on_mouse_motion(self, x: float, y: float, dx: float, dy: float) -> None:  # type: ignore[override]
        """鼠标移动，更新hover状态"""
        # 若正在拖动价格信息滚动条，则根据鼠标位置更新滚动
        if self.active_dialog == "price_info" and self.price_info_dragging and self.price_info_scrollbar_meta:
            _track_x, track_y, track_h, handle_h, max_visible, total_lines = self.price_info_scrollbar_meta
            max_scroll = max(0, total_lines - max_visible)
            if max_scroll > 0:
                rel = (y - track_y - handle_h / 2) / max(1.0, (track_h - handle_h))
                rel = max(0.0, min(1.0, rel))
                self.price_info_scroll = int(rel * max_scroll)
            return

        self.hover_button = None
        self.hover_city = None
        
        # 检查按钮hover
        for name, (x1, y1, x2, y2) in self.button_regions.items():
            if x1 <= x <= x2 and y1 <= y <= y2:
                self.hover_button = name
                return

    def on_mouse_release(self, x: float, y: float, button: int, modifiers: int) -> None:  # type: ignore[override]
        """鼠标松开时，结束价格信息滚动条拖动。"""
        if button == arcade.MOUSE_BUTTON_LEFT:
            self.price_info_dragging = False
        
        # 检查城市行hover（允许同城市多处出现）
        for name, (x1, y1, x2, y2) in self.city_row_regions:
            if x1 <= x <= x2 and y1 <= y <= y2:
                self.hover_city = name
                return
    
    def on_mouse_press(  # type: ignore[override]
        self, x: float, y: float, button: int, modifiers: int
    ) -> None:
        if button != arcade.MOUSE_BUTTON_LEFT:
            return

        # 如果有活动弹窗，只处理弹窗内的交互，屏蔽主界面
        if self.active_dialog:
            # 优先命中弹窗按钮区域
            for name, (x1, y1, x2, y2) in self.button_regions.items():
                if x1 <= x <= x2 and y1 <= y <= y2:
                    self._handle_button(name)
                    return

            # 价格信息弹窗：滚动条拖动
            if self.active_dialog == "price_info" and self.price_info_scrollbar_meta:
                track_x, track_y, track_h, handle_h, max_visible, total_lines = self.price_info_scrollbar_meta
                scrollbar_w = 10
                if track_x <= x <= track_x + scrollbar_w and track_y <= y <= track_y + track_h:
                    self.price_info_dragging = True
                    # 根据当前鼠标 y 位置设置 scroll
                    max_scroll = max(0, total_lines - max_visible)
                    if max_scroll > 0:
                        # 把 y 映射到 0~1 再映射到滚动行数
                        rel = (y - track_y - handle_h / 2) / max(1.0, (track_h - handle_h))
                        rel = max(0.0, min(1.0, rel))
                        self.price_info_scroll = int(rel * max_scroll)
                    return

        # 全局快捷键（无弹窗时）

            # 出行弹窗特殊处理
            if self.active_dialog == "travel":
                for name, (x1, y1, x2, y2) in self.city_row_regions:
                    if x1 <= x <= x2 and y1 <= y <= y2:
                        if name not in self._reachable_cities():
                            self._log("该城市在当前模式下不可达。")
                            return
                        self.travel_target = name
                        return
            # 出海弹窗：点击目的地行
            if self.active_dialog == "sail":
                for name, (x1, y1, x2, y2) in self.sail_row_regions:
                    if x1 <= x <= x2 and y1 <= y <= y2:
                        self.sail_target = name
                        return
            return

        # 没有活动弹窗时才处理主界面交互
        for name, (x1, y1, x2, y2) in self.button_regions.items():
            if x1 <= x <= x2 and y1 <= y <= y2:
                self._handle_button(name)
                return

    def on_key_press(self, key: int, modifiers: int) -> None:  # type: ignore[override]
        # 挑战结算：仅按钮有效，忽略键盘
        if self.current_screen == "challenge_end":
            return
        # 模式选择且无弹窗：仅 F9 打开读档
        if self.current_screen == "mode_select" and not self.active_dialog:
            if key == arcade.key.F9:
                self.load_slot_selected = None
                self.active_dialog = "load"
            return
        # 弹窗优先处理
        if self.active_dialog:
            if key == arcade.key.ESCAPE:
                # 优先关闭帮助弹窗
                if self.help_popup_text:
                    self.help_popup_text = None
                    return
                # 自由模式初始金额弹窗
                if self.active_dialog == "start_cash":
                    self.start_cash_focused = False
                    self.active_dialog = None
                    return
                # 若有二级弹窗，优先关闭二级弹窗
                if self.active_dialog == "market" and self.market_order_dialog is not None:
                    self.market_order_dialog = None
                    self.market_order_focused = False
                else:
                    if self.active_dialog == "loans":
                        self.loans_focused_input = None
                        self.loans_amount_hint = ""
                    self.active_dialog = None
                return

            # 自由模式初始金额弹窗键盘输入
            if self.active_dialog == "start_cash":
                if not self.start_cash_focused:
                    # 未聚焦时仅允许 Enter 直接确认
                    if key in (arcade.key.ENTER, arcade.key.SPACE):
                        self._handle_button("start_cash_confirm")
                    return

                if key in (
                    arcade.key.NUM_0, arcade.key.NUM_1, arcade.key.NUM_2, arcade.key.NUM_3, arcade.key.NUM_4,
                    arcade.key.NUM_5, arcade.key.NUM_6, arcade.key.NUM_7, arcade.key.NUM_8, arcade.key.NUM_9,
                    arcade.key.KEY_0, arcade.key.KEY_1, arcade.key.KEY_2, arcade.key.KEY_3, arcade.key.KEY_4,
                    arcade.key.KEY_5, arcade.key.KEY_6, arcade.key.KEY_7, arcade.key.KEY_8, arcade.key.KEY_9,
                ):
                    mapping = {
                        arcade.key.NUM_0: "0", arcade.key.KEY_0: "0",
                        arcade.key.NUM_1: "1", arcade.key.KEY_1: "1",
                        arcade.key.NUM_2: "2", arcade.key.KEY_2: "2",
                        arcade.key.NUM_3: "3", arcade.key.KEY_3: "3",
                        arcade.key.NUM_4: "4", arcade.key.KEY_4: "4",
                        arcade.key.NUM_5: "5", arcade.key.KEY_5: "5",
                        arcade.key.NUM_6: "6", arcade.key.KEY_6: "6",
                        arcade.key.NUM_7: "7", arcade.key.KEY_7: "7",
                        arcade.key.NUM_8: "8", arcade.key.KEY_8: "8",
                        arcade.key.NUM_9: "9", arcade.key.KEY_9: "9",
                    }
                    digit = mapping.get(key)
                    if digit:
                        text = (self.start_cash_text or "0").strip()
                        new_text = (text + digit).lstrip("0")
                        self.start_cash_text = (new_text or "0")[:12]
                    return

                if key in (arcade.key.BACKSPACE, arcade.key.DELETE):
                    t = (self.start_cash_text or "0").strip()
                    self.start_cash_text = (t[:-1] if len(t) > 0 else "0") or "0"
                    return

                if key in (arcade.key.ENTER, arcade.key.SPACE):
                    self._handle_button("start_cash_confirm")
                    return

                return
            
            # 市场弹窗快捷键
            if self.active_dialog == "market":
                # 若二级数量确认弹窗开启，则键盘只作用于该弹窗
                if self.market_order_dialog is not None:
                    if self.demo_recording:
                        # 演示/录制模式下仅允许按钮档位，避免键盘绕过
                        if key in (arcade.key.ENTER, arcade.key.SPACE):
                            self._handle_button("order_confirm")
                        return
                    text = str(self.market_order_dialog.get("text", "")).strip()
                    # 只有在输入框获得焦点时，才处理数字输入/删除
                    if self.market_order_focused and key in (
                        arcade.key.NUM_0, arcade.key.NUM_1, arcade.key.NUM_2,
                               arcade.key.NUM_3, arcade.key.NUM_4, arcade.key.NUM_5,
                               arcade.key.NUM_6, arcade.key.NUM_7, arcade.key.NUM_8,
                               arcade.key.NUM_9,
                               arcade.key.KEY_0, arcade.key.KEY_1, arcade.key.KEY_2,
                               arcade.key.KEY_3, arcade.key.KEY_4, arcade.key.KEY_5,
                               arcade.key.KEY_6, arcade.key.KEY_7, arcade.key.KEY_8,
                               arcade.key.KEY_9):
                        digit = chr(key & 0xFF) if 48 <= (key & 0xFF) <= 57 else None
                        if digit is None:
                            # 退而求其次：用最后一位数字推断
                            mapping = {
                                arcade.key.NUM_0: "0", arcade.key.KEY_0: "0",
                                arcade.key.NUM_1: "1", arcade.key.KEY_1: "1",
                                arcade.key.NUM_2: "2", arcade.key.KEY_2: "2",
                                arcade.key.NUM_3: "3", arcade.key.KEY_3: "3",
                                arcade.key.NUM_4: "4", arcade.key.KEY_4: "4",
                                arcade.key.NUM_5: "5", arcade.key.KEY_5: "5",
                                arcade.key.NUM_6: "6", arcade.key.KEY_6: "6",
                                arcade.key.NUM_7: "7", arcade.key.KEY_7: "7",
                                arcade.key.NUM_8: "8", arcade.key.KEY_8: "8",
                                arcade.key.NUM_9: "9", arcade.key.KEY_9: "9",
                            }
                            digit = mapping.get(key)
                        if digit:
                            new_text = (text + digit).lstrip("0")
                            if not new_text:
                                new_text = "0"
                            # 限制长度，防止溢出
                            new_text = new_text[:6]
                            self.market_order_dialog["text"] = new_text
                    elif self.market_order_focused and key in (arcade.key.BACKSPACE, arcade.key.DELETE):
                        new_text = text[:-1]
                        self.market_order_dialog["text"] = new_text
                    elif key == arcade.key.LEFT:
                        # 等价于 -1
                        qty = int(self.market_order_dialog.get("qty", 1))
                        qty = max(1, qty - 1)
                        self.market_order_dialog["qty"] = qty
                        self.market_order_dialog["text"] = str(qty)
                    elif key == arcade.key.RIGHT:
                        qty = int(self.market_order_dialog.get("qty", 1))
                        qty = qty + 1
                        self.market_order_dialog["qty"] = qty
                        self.market_order_dialog["text"] = str(qty)
                    elif key in (arcade.key.ENTER, arcade.key.SPACE):
                        # 等价于点击确认按钮
                        self._handle_button("order_confirm")
                    return

                products = sorted(PRODUCTS.keys())
                if key == arcade.key.M:
                    self.active_dialog = None
                    return
                if not products:
                    return
                if key == arcade.key.UP:
                    self.market_index = (self.market_index - 1) % len(products)
                elif key == arcade.key.DOWN:
                    self.market_index = (self.market_index + 1) % len(products)
                elif key == arcade.key.LEFT:
                    self.market_qty = max(1, self.market_qty - 1)
                elif key == arcade.key.RIGHT:
                    self.market_qty = min(999, self.market_qty + 1)
                elif key == arcade.key.TAB:
                    self.market_tab = "sell" if self.market_tab == "buy" else "buy"
                return

            # 车厂弹窗：购买数量输入框
            if self.active_dialog == "repair" and not self.dialog_data.get("factory_buy_pending"):
                if self.factory_buy_focused and key in (
                    arcade.key.NUM_0, arcade.key.NUM_1, arcade.key.NUM_2, arcade.key.NUM_3, arcade.key.NUM_4,
                    arcade.key.NUM_5, arcade.key.NUM_6, arcade.key.NUM_7, arcade.key.NUM_8, arcade.key.NUM_9,
                    arcade.key.KEY_0, arcade.key.KEY_1, arcade.key.KEY_2, arcade.key.KEY_3, arcade.key.KEY_4,
                    arcade.key.KEY_5, arcade.key.KEY_6, arcade.key.KEY_7, arcade.key.KEY_8, arcade.key.KEY_9,
                ):
                    mapping = {
                        arcade.key.NUM_0: "0", arcade.key.KEY_0: "0", arcade.key.NUM_1: "1", arcade.key.KEY_1: "1",
                        arcade.key.NUM_2: "2", arcade.key.KEY_2: "2", arcade.key.NUM_3: "3", arcade.key.KEY_3: "3",
                        arcade.key.NUM_4: "4", arcade.key.KEY_4: "4", arcade.key.NUM_5: "5", arcade.key.KEY_5: "5",
                        arcade.key.NUM_6: "6", arcade.key.KEY_6: "6", arcade.key.NUM_7: "7", arcade.key.KEY_7: "7",
                        arcade.key.NUM_8: "8", arcade.key.KEY_8: "8", arcade.key.NUM_9: "9", arcade.key.KEY_9: "9",
                    }
                    digit = mapping.get(key)
                    if digit:
                        text = (self.factory_buy_text or "").strip().lstrip("0") or "0"
                        new_text = (text + digit).lstrip("0") or "0"
                        self.factory_buy_text = new_text[:5]
                elif self.factory_buy_focused and key in (arcade.key.BACKSPACE, arcade.key.DELETE):
                    text = (self.factory_buy_text or "1").strip()
                    new_text = text[:-1]
                    self.factory_buy_text = (new_text or "1") if new_text else "1"
                return
            
            # 借贷弹窗快捷键
            if self.active_dialog == "loans":
                if key == arcade.key.J:
                    self.active_dialog = None
                    self.loans_focused_input = None
                    self.loans_amount_hint = ""
                    return
                # 键盘输入根据当前焦点写入对应输入框
                if self.loans_focused_input == "borrow":
                    text = self.loans_amount_text
                    if key in (
                        arcade.key.NUM_0, arcade.key.NUM_1, arcade.key.NUM_2, arcade.key.NUM_3, arcade.key.NUM_4,
                        arcade.key.NUM_5, arcade.key.NUM_6, arcade.key.NUM_7, arcade.key.NUM_8, arcade.key.NUM_9,
                        arcade.key.KEY_0, arcade.key.KEY_1, arcade.key.KEY_2, arcade.key.KEY_3, arcade.key.KEY_4,
                        arcade.key.KEY_5, arcade.key.KEY_6, arcade.key.KEY_7, arcade.key.KEY_8, arcade.key.KEY_9,
                    ):
                        mapping = {
                            arcade.key.NUM_0: "0", arcade.key.KEY_0: "0", arcade.key.NUM_1: "1", arcade.key.KEY_1: "1",
                            arcade.key.NUM_2: "2", arcade.key.KEY_2: "2", arcade.key.NUM_3: "3", arcade.key.KEY_3: "3",
                            arcade.key.NUM_4: "4", arcade.key.KEY_4: "4", arcade.key.NUM_5: "5", arcade.key.KEY_5: "5",
                            arcade.key.NUM_6: "6", arcade.key.KEY_6: "6", arcade.key.NUM_7: "7", arcade.key.KEY_7: "7",
                            arcade.key.NUM_8: "8", arcade.key.KEY_8: "8", arcade.key.NUM_9: "9", arcade.key.KEY_9: "9",
                        }
                        digit = mapping.get(key)
                        if digit:
                            new_text = (text + digit).lstrip("0")
                            if not new_text:
                                new_text = "0"
                            self.loans_amount_text = new_text[:10]
                    elif key in (arcade.key.BACKSPACE, arcade.key.DELETE):
                        self.loans_amount_text = text[:-1]
                    elif key in (arcade.key.ENTER, arcade.key.SPACE):
                        self._handle_button("loans_exec_borrow")
                    return
                if self.loans_focused_input == "repay":
                    text = (self.loans_repay_amount_text or "0").strip()
                    if key in (
                        arcade.key.NUM_0, arcade.key.NUM_1, arcade.key.NUM_2, arcade.key.NUM_3, arcade.key.NUM_4,
                        arcade.key.NUM_5, arcade.key.NUM_6, arcade.key.NUM_7, arcade.key.NUM_8, arcade.key.NUM_9,
                        arcade.key.KEY_0, arcade.key.KEY_1, arcade.key.KEY_2, arcade.key.KEY_3, arcade.key.KEY_4,
                        arcade.key.KEY_5, arcade.key.KEY_6, arcade.key.KEY_7, arcade.key.KEY_8, arcade.key.KEY_9,
                    ):
                        mapping = {
                            arcade.key.NUM_0: "0", arcade.key.KEY_0: "0", arcade.key.NUM_1: "1", arcade.key.KEY_1: "1",
                            arcade.key.NUM_2: "2", arcade.key.KEY_2: "2", arcade.key.NUM_3: "3", arcade.key.KEY_3: "3",
                            arcade.key.NUM_4: "4", arcade.key.KEY_4: "4", arcade.key.NUM_5: "5", arcade.key.KEY_5: "5",
                            arcade.key.NUM_6: "6", arcade.key.KEY_6: "6", arcade.key.NUM_7: "7", arcade.key.KEY_7: "7",
                            arcade.key.NUM_8: "8", arcade.key.KEY_8: "8", arcade.key.NUM_9: "9", arcade.key.KEY_9: "9",
                        }
                        digit = mapping.get(key)
                        if digit:
                            new_text = (text + digit).lstrip("0")
                            if not new_text:
                                new_text = "0"
                            self.loans_repay_amount_text = new_text[:10]
                            try:
                                self.loans_repay_amount = float(new_text.replace(",", ""))
                            except ValueError:
                                pass
                    elif key in (arcade.key.BACKSPACE, arcade.key.DELETE):
                        self.loans_repay_amount_text = text[:-1]
                        try:
                            self.loans_repay_amount = float(self.loans_repay_amount_text.replace(",", "")) if self.loans_repay_amount_text.strip() else 0.0
                        except ValueError:
                            self.loans_repay_amount = 0.0
                    elif key in (arcade.key.ENTER, arcade.key.SPACE):
                        self._handle_button("loans_exec_repay")
                    return
                # 无焦点或焦点不在输入框时：方向键不修改任一输入框
                if key == arcade.key.LEFT:
                    if self.loans_focused_input == "borrow":
                        self.loans_amount = max(0.0, self.loans_amount - 1000.0)
                        self.loans_amount_text = str(int(self.loans_amount))
                    elif self.loans_focused_input == "repay":
                        self.loans_repay_amount = max(0.0, self.loans_repay_amount - 1000.0)
                        self.loans_repay_amount_text = str(int(self.loans_repay_amount))
                elif key == arcade.key.RIGHT:
                    if self.loans_focused_input == "borrow":
                        self.loans_amount += 1000.0
                        self.loans_amount_text = str(int(self.loans_amount))
                    elif self.loans_focused_input == "repay":
                        self.loans_repay_amount += 1000.0
                        self.loans_repay_amount_text = str(int(self.loans_repay_amount))
                elif key == arcade.key.UP:
                    if self.loans_focused_input == "borrow":
                        self.loans_amount += 100.0
                        self.loans_amount_text = str(int(self.loans_amount))
                    elif self.loans_focused_input == "repay":
                        self.loans_repay_amount += 100.0
                        self.loans_repay_amount_text = str(int(self.loans_repay_amount))
                elif key == arcade.key.DOWN:
                    if self.loans_focused_input == "borrow":
                        self.loans_amount = max(0.0, self.loans_amount - 100.0)
                        self.loans_amount_text = str(int(self.loans_amount))
                    elif self.loans_focused_input == "repay":
                        self.loans_repay_amount = max(0.0, self.loans_repay_amount - 100.0)
                        self.loans_repay_amount_text = str(int(self.loans_repay_amount))
                elif key == arcade.key.TAB:
                    self.loans_mode = "daily" if self.loans_mode == "lump" else "lump"
                return
            return

        # 全局快捷键（无弹窗时）
        if key == arcade.key.ESCAPE:
            arcade.close_window()
        elif key == arcade.key.A:
            self.active_dialog = "travel"
        elif key == arcade.key.M:
            self.active_dialog = "market"
        elif key == arcade.key.J:
            if self.demo_recording and not self._cli_can_loan():
                self._log("当前城市无银行，无法借贷（CLI 规则）。")
            else:
                self.active_dialog = "loans"
        elif key == arcade.key.R:
            self.active_dialog = "repair"
        elif key == arcade.key.T:
            if self.demo_recording and not self._cli_can_sail():
                self._log("当前城市无法出海（CLI 规则：仅港口城市可出海）。")
            else:
                self.active_dialog = "sail"
        elif key == arcade.key.F8:
            # 录制人类玩家轨迹（用于 PPO warmstart）
            if not self.demo_recording:
                self.demo_recorder = HumanDemoRecorder()
                self.demo_recording = True
                self._log("已开始录制玩家轨迹（F8 停止并保存）。")
            else:
                self.demo_recording = False
                if self.demo_recorder is None:
                    self._log("录制已停止（无数据）。")
                else:
                    out = default_demo_path()
                    self.demo_recorder.save_npz(out)
                    self._log(f"录制已保存：{out}（steps={self.demo_recorder.size} dropped={self.demo_recorder.dropped}）。")
                self.demo_recorder = None
        elif key == arcade.key.N:
            # 下一天
            if self.demo_recording and self.demo_recorder is not None:
                snap = copy.deepcopy(self.state)
                self.demo_recorder.record(snap, self.rng, api_actions.ActionNextDay())
            result = self._do_advance_day()
            if result is not None:
                msgs, _ = result
                for m in msgs:
                    self._log(m)
                self._log(f"新的一天开始了，商品价格已更新。")
        elif key == arcade.key.F5:
            self.active_dialog = "save"
        elif key == arcade.key.F9:
            self.active_dialog = "load"

    def _handle_button(self, name: str) -> None:
        """根据按钮 id 分派操作。"""
        # 模式选择界面
        if self.current_screen == "mode_select":
            if name == "mode_challenge":
                self.state.game_mode = "challenge"
                self.current_screen = "playing"
                return
            if name == "mode_free":
                # 自由模式：先弹出初始金额设置
                self.start_cash_text = str(int(INITIAL_CASH))
                self.start_cash_focused = True
                self.active_dialog = "start_cash"
                return
            if name == "mode_load":
                self.load_slot_selected = None
                self.active_dialog = "load"
                return
            if name.startswith("start_cash_"):
                if name == "start_cash_input":
                    self.start_cash_focused = True
                    return
                if name == "start_cash_cancel":
                    self.start_cash_focused = False
                    self.active_dialog = None
                    return
                if name in ("start_cash_minus_1000", "start_cash_plus_1000", "start_cash_minus_10000", "start_cash_plus_10000"):
                    try:
                        cur = int(float((self.start_cash_text or "0").replace(",", "").strip() or "0"))
                    except Exception:
                        cur = 0
                    delta = 0
                    if name == "start_cash_minus_1000":
                        delta = -1000
                    elif name == "start_cash_plus_1000":
                        delta = 1000
                    elif name == "start_cash_minus_10000":
                        delta = -10000
                    elif name == "start_cash_plus_10000":
                        delta = 10000
                    cur = max(0, cur + delta)
                    self.start_cash_text = str(cur)
                    return
                if name == "start_cash_confirm":
                    try:
                        val = int(float((self.start_cash_text or "0").replace(",", "").strip() or "0"))
                    except Exception:
                        val = int(INITIAL_CASH)
                    val = max(0, min(val, 1_000_000_000))
                    # 重新开一局自由模式
                    self.state = GameState()
                    self.state.game_mode = "free"
                    self.state.player.cash = float(val)
                    self.state.daily_lambdas = refresh_daily_lambdas(self.rng, None)
                    self.log = []
                    self.active_dialog = None
                    self.start_cash_focused = False
                    self.current_screen = "playing"
                    return

        # 挑战结算界面
        if self.current_screen == "challenge_end" and name == "challenge_restart":
            self.challenge_end_data = None
            self.state = GameState()
            self.state.daily_lambdas = refresh_daily_lambdas(self.rng, None)
            self.current_screen = "mode_select"
            return

        # 全局主界面按钮
        if name == "btn_travel":
            self.active_dialog = "travel"
            self.travel_target = None
            return
        if name == "btn_market":
            self.active_dialog = "market"
            return
        if name == "btn_loans":
            self.active_dialog = "loans"
            self.loans_amount_text = "0"
            self.loans_repay_amount_text = "0"
            self.loans_amount_hint = ""
            self.loans_focused_input = None
            return
        if name == "btn_repair_truck":
            self.active_dialog = "repair"
            self.dialog_data.pop("factory_buy_pending", None)
            self.factory_buy_text = "1"
            self.factory_buy_qty = 1
            self.factory_buy_focused = False
            return
        if name == "btn_sail":
            p = self.state.player
            if p.location not in LAND_SEA_CITIES and p.location not in SEA_CITIES:
                self._log("请先前往海港城市（上海/福州/广州/深圳或海岛）再使用出海。")
                return
            self.active_dialog = "sail"
            self.sail_target = None
            return
        if name == "btn_next":
            self.on_key_press(arcade.key.N, 0)
            return
        if name == "btn_save":
            self.active_dialog = "save"
            return
        if name == "btn_load":
            self.active_dialog = "load"
            return
        if name == "btn_price_info":
            self._update_price_info_text()
            self.price_info_scroll = 0
            self.active_dialog = "price_info"
            return
        
        # 帮助弹窗关闭
        if name == "help_popup_close":
            self.help_popup_text = None
            return

        # 弹窗帮助按钮
        if name == "dialog_help" and self.active_dialog:
            self.help_popup_text = DIALOG_HELP_TEXTS.get(self.active_dialog, "")
            return

        # 弹窗关闭按钮
        if name == "dialog_close":
            self.help_popup_text = None
            if self.active_dialog == "loans":
                self.loans_focused_input = None
                self.loans_amount_hint = ""
            if self.active_dialog == "repair":
                self.dialog_data.pop("factory_buy_pending", None)
                self.factory_buy_focused = False
            self.active_dialog = None
            return
        
        # 出行弹窗按钮
        if name.startswith("travel_"):
            if name == "travel_cancel":
                self.active_dialog = None
                self.travel_target = None
            elif name in ("travel_confirm_normal", "travel_confirm_fast"):
                if not self.travel_target:
                    self._log("请先在左侧城市列表中选择目的地。")
                else:
                    fast = name == "travel_confirm_fast"
                    self._travel(self.travel_target, fast=fast)
                    self.active_dialog = None
            return
        
        # 市场弹窗按钮
        if name == "market_tab_buy":
            self.market_tab = "buy"
            return
        if name == "market_tab_sell":
            self.market_tab = "sell"
            return
        if name.startswith("price_hist_"):
            p = self.state.player
            city = p.location
            lines: List[str] = []
            if name.startswith("price_hist_buy_"):
                pid = name.replace("price_hist_buy_", "")
                key = f"{city}|{pid}"
                hist = self.state.price_history_buy_7d.get(key, [])
                if not hist:
                    lines.append("暂无该商品的 7 日采购价记录。")
                else:
                    series = "，".join(f"{v:.2f}" for v in hist)
                    lines.append(f"{city}【{PRODUCTS[pid].name}】最近 7 日采购价：")
                    lines.append(series + " 元")
            elif name.startswith("price_hist_sell_"):
                pid = name.replace("price_hist_sell_", "")
                key = f"{city}|{pid}"
                hist = self.state.price_history_sell_7d.get(key, [])
                if not hist:
                    lines.append("暂无该商品的 7 日售卖价记录。")
                else:
                    series = "，".join(f"{v:.2f}" for v in hist)
                    lines.append(f"{city}【{PRODUCTS[pid].name}】最近 7 日售卖价：")
                    lines.append(series + " 元")

            # 点击行内【7日】时，直接将该商品的明细写入价格信息弹窗内容，并弹出
            if lines:
                self.price_info_lines = lines
                self.price_info_scroll = 0
                self.active_dialog = "price_info"
            return
        if name.startswith("buy_"):
            pid = name.replace("buy_", "")
            # 打开数量确认弹窗（采购）
            qty = max(1, self.market_qty)
            # 初始为 0，等待用户输入
            self.market_order_dialog = {"mode": "buy", "pid": pid, "qty": qty, "text": "0"}
            self.market_order_focused = False
            return
        if name.startswith("sell_"):
            pid = name.replace("sell_", "")
            # 打开数量确认弹窗（售出）
            qty = max(1, self.market_qty)
            self.market_order_dialog = {"mode": "sell", "pid": pid, "qty": qty, "text": "0"}
            self.market_order_focused = False
            return

        if name == "order_input":
            # 输入框获得焦点
            if self.market_order_dialog is not None:
                if self.demo_recording:
                    self._log("演示/录制模式下请使用数量档位按钮。")
                    self.market_order_focused = False
                    return
                self.market_order_focused = True
            return

        # 市场二级小弹窗按钮
        if name.startswith("order_"):
            if not self.market_order_dialog:
                return
            # 这里需要在函数作用域内统一引入，否则会导致某些分支引用未赋值
            # （Python 会把 import 绑定视为局部变量声明）。
            from trade_game.inventory import cargo_used
            mode = self.market_order_dialog.get("mode")
            pid = self.market_order_dialog.get("pid")
            qty = int(self.market_order_dialog.get("qty", 1))
            qty = max(1, qty)

            if name == "order_qty_minus":
                self.market_order_dialog["qty"] = max(1, qty - 1)
                self.market_order_dialog["text"] = str(max(1, qty - 1))
            elif name == "order_qty_plus":
                self.market_order_dialog["qty"] = qty + 1
                self.market_order_dialog["text"] = str(qty + 1)
            elif self.demo_recording and name in (
                "order_qty_f1", "order_qty_f2", "order_qty_f3", "order_qty_f4", "order_qty_f5",
                "order_qty_1", "order_qty_2", "order_qty_3",
            ):
                # 演示/录制模式：数量仅允许「当前上限 * (1/5..5/5)」或固定 1/2/3
                if not pid or mode not in ("buy", "sell"):
                    return
                p = self.state.player
                city = p.location
                prod = PRODUCTS.get(pid)
                if not prod:
                    return
                if mode == "buy":
                    unit = purchase_price(prod, city, self.state.daily_lambdas)
                    if unit is None:
                        unit = prod.base_purchase_price
                    cap = total_storage_capacity(p, city)
                    capacity_left = max(0, cap - cargo_used(p.cargo_lots))
                    max_by_cash = int(p.cash / unit) if unit > 0 else capacity_left
                    base_max = max(1, min(capacity_left, max_by_cash))
                else:
                    have = sum(l.quantity for l in p.cargo_lots if l.product_id == pid)
                    base_max = max(1, int(have))

                if name.startswith("order_qty_f"):
                    k = int(name.replace("order_qty_f", ""))
                    k = max(1, min(5, k))
                    if k >= 5:
                        new_qty = base_max
                    else:
                        new_qty = max(1, (base_max * k) // 5)
                else:
                    fixed = int(name.replace("order_qty_", ""))
                    new_qty = max(1, min(base_max, fixed))

                self.market_order_dialog["qty"] = int(new_qty)
                # 保持 text 为空，避免键盘输入；仍可在非录制模式使用输入框
                self.market_order_dialog["text"] = ""
            elif name == "order_cancel":
                self.market_order_dialog = None
                self.market_order_focused = False
            elif name == "order_confirm":
                # 演示/录制模式：数量由「上限的 1/5~5/5 + 固定 1/2/3」按钮产生，
                # 确认阶段仅做“不得超过当前上限”的裁剪。
                if self.demo_recording:
                    p = self.state.player
                    if mode == "buy":
                        rem = int(total_storage_capacity(p, p.location) - cargo_used(p.cargo_lots))
                        if rem <= 0:
                            self._log("容量已满，无法采购。")
                            return
                        qty = max(1, min(int(qty), rem))
                    else:
                        have = sum(l.quantity for l in p.cargo_lots if l.product_id == pid)
                        if have <= 0:
                            self._log("你没有该商品，无法售出。")
                            return
                        qty = max(1, min(int(qty), int(have)))
                if mode == "buy":
                    snap = copy.deepcopy(self.state) if (self.demo_recording and self.demo_recorder is not None) else None
                    ok, msg = self._buy_in_ui(pid, qty)
                    self._log(msg)
                    if ok and snap is not None:
                        self.demo_recorder.record(snap, self.rng, api_actions.ActionBuy(product_id=str(pid), quantity=int(qty)))
                elif mode == "sell":
                    snap = copy.deepcopy(self.state) if (self.demo_recording and self.demo_recorder is not None) else None
                    ok, msg = self._sell_in_ui(pid, qty)
                    self._log(msg)
                    if ok and snap is not None:
                        self.demo_recorder.record(snap, self.rng, api_actions.ActionSell(product_id=str(pid), quantity=int(qty)))
                # 无论成功与否，都关闭小弹窗
                self.market_order_dialog = None
                self.market_order_focused = False
                # 成功的买入/卖出操作会消耗 1 天时间
                if ok:
                    result = self._do_advance_day()
                    if result is not None:
                        msgs, _ = result
                        for m in msgs:
                            self._log(m)
            return
        
        # 借贷弹窗按钮（两个输入框完全独立：借贷 / 还贷）
        if name == "loans_amount_input":
            if self.demo_recording:
                self._log("演示/录制模式下请使用 1/3~3/3 档位按钮选择金额。")
                self.loans_focused_input = None
                return
            self.loans_focused_input = "borrow"
            return
        if name == "loans_repay_input":
            if self.demo_recording:
                self._log("演示/录制模式下请使用 1/3~3/3 档位按钮选择金额。")
                self.loans_focused_input = None
                return
            self.loans_focused_input = "repay"
            return
        if name.startswith("loans_"):
            if self.demo_recording and name in ("loans_borrow_f1", "loans_borrow_f2", "loans_borrow_f3"):
                p = self.state.player
                total_debt = sum(l.debt_total() for l in self.state.loans)
                net_assets = p.cash - total_debt
                principal_total = total_outstanding_principal(self.state.loans)
                max_loan = max(0.0, net_assets - principal_total)
                if max_loan <= 0:
                    self._log("当前无可借额度（净资产不足）。")
                    return
                k = int(name[-1])
                frac = (1.0 / 3.0) * k
                amount = max(100.0, round(frac * max_loan, 2))
                self.loans_amount = amount
                self.loans_amount_text = str(int(amount))
                self.loans_amount_hint = ""
                return
            if self.demo_recording and name in ("loans_repay_f1", "loans_repay_f2", "loans_repay_f3"):
                total_debt = sum(l.debt_total() for l in self.state.loans)
                cash = self.state.player.cash
                max_repay = min(cash, total_debt)
                if max_repay <= 0:
                    self._log("当前无可还金额。")
                    return
                k = int(name[-1])
                frac = (1.0 / 3.0) * k
                amount = round(frac * max_repay, 2)
                # 3/3 按钮显示为“全还”，让后续执行逻辑按 all 处理
                if k >= 3:
                    amount = max_repay
                self.loans_repay_amount = float(amount)
                self.loans_repay_amount_text = str(int(amount))
                return
            if name == "loans_minus":
                self.loans_amount = max(100.0, self.loans_amount - 1000.0)
                self.loans_amount_text = str(int(self.loans_amount))
                self.loans_amount_hint = ""
            elif name == "loans_plus":
                self.loans_amount += 1000.0
                self.loans_amount_text = str(int(self.loans_amount))
                self.loans_amount_hint = ""
            elif name == "loans_exec_borrow":
                p = self.state.player
                total_debt = sum(l.debt_total() for l in self.state.loans)
                net_assets = p.cash - total_debt
                principal_total = total_outstanding_principal(self.state.loans)
                max_loan = max(0.0, net_assets - principal_total)
                if max_loan <= 0:
                    self._log("当前无可借额度（净资产不足）。")
                    return
                if self.demo_recording:
                    # 录制模式下对齐 train_config：仅允许 1/3、2/3、3/3 额度
                    raw = self.loans_amount_text.strip()
                    try:
                        entered = float(raw.replace(",", "")) if raw else 0.0
                    except ValueError:
                        entered = 0.0
                    ratio = entered / max_loan if max_loan > 0 else 0.0
                    best_i = 0
                    best_err = abs(ratio - AMOUNT_FRACTIONS[0])
                    for i, frac in enumerate(AMOUNT_FRACTIONS):
                        if abs(ratio - frac) < best_err:
                            best_err = abs(ratio - frac)
                            best_i = i
                    amount = max(100.0, round(AMOUNT_FRACTIONS[best_i] * max_loan, 2))
                else:
                    raw = self.loans_amount_text.strip()
                    try:
                        amount = float(raw.replace(",", "")) if raw else self.loans_amount
                    except ValueError:
                        amount = self.loans_amount
                    amount = max(100.0, min(amount, max_loan))
                self.loans_amount = amount
                self._do_borrow_ui()
                self.active_dialog = None
                self.loans_amount_hint = ""
                self.loans_amount = 0.0
                self.loans_amount_text = "0"
                self.loans_focused_input = None
            elif name == "loans_repay_minus":
                self.loans_repay_amount = max(0.0, self.loans_repay_amount - 1000.0)
                self.loans_repay_amount_text = str(int(self.loans_repay_amount))
            elif name == "loans_repay_plus":
                self.loans_repay_amount += 1000.0
                self.loans_repay_amount_text = str(int(self.loans_repay_amount))
            elif name == "loans_exec_repay":
                total_debt = sum(l.debt_total() for l in self.state.loans)
                cash = self.state.player.cash
                max_repay = min(cash, total_debt)
                if self.demo_recording and max_repay > 0:
                    # 录制模式下对齐 train_config：仅允许 1/3、2/3、3/3（全部）
                    raw = (self.loans_repay_amount_text or "0").strip()
                    try:
                        entered = float(raw.replace(",", "")) if raw else 0.0
                    except ValueError:
                        entered = 0.0
                    ratio = entered / max_repay
                    if ratio >= 0.99:
                        amount = "all"
                    else:
                        best_i = 0
                        best_err = abs(ratio - AMOUNT_FRACTIONS[0])
                        for i, frac in enumerate(AMOUNT_FRACTIONS):
                            if abs(ratio - frac) < best_err:
                                best_err = abs(ratio - frac)
                                best_i = i
                        amount = max(100.0, round(AMOUNT_FRACTIONS[best_i] * max_repay, 2))
                else:
                    raw = (self.loans_repay_amount_text or "0").strip()
                    try:
                        amount = float(raw.replace(",", "")) if raw else self.loans_repay_amount
                    except ValueError:
                        amount = self.loans_repay_amount
                    amount = max(0.0, min(amount, total_debt, cash))
                    if total_debt > 0:
                        amount = max(100.0, amount)
                before = self.state.player.cash
                snap = copy.deepcopy(self.state) if (self.demo_recording and self.demo_recorder is not None) else None
                repay_amt: float | None = None if amount == "all" else float(amount)
                self.state.player.cash = repay(self.state.loans, cash=self.state.player.cash, amount=repay_amt)
                spent = before - self.state.player.cash
                if spent <= 0:
                    self._log("现金不足，无法还款。")
                else:
                    self._log(f"已还款：{spent:.0f} 元。")
                    # 录制：还款（pre-action 状态）
                    if snap is not None:
                        self.demo_recorder.record(
                            snap, self.rng,
                            api_actions.ActionRepay(amount="all" if amount == "all" else float(amount)),
                        )
                    # 还款成功也消耗 1 天
                    result = self._do_advance_day()
                    if result is not None:
                        msgs, _ = result
                        for m in msgs:
                            self._log(m)
                # 重置输入框为 0
                self.loans_repay_amount = 0.0
                self.loans_repay_amount_text = "0"
            return
        if name.startswith("price_info_"):
            if name == "price_info_up":
                self.price_info_scroll = max(0, self.price_info_scroll - 1)
            elif name == "price_info_down":
                self.price_info_scroll += 1
            elif name == "price_info_close":
                self.active_dialog = None
            # 停止拖动
            self.price_info_dragging = False
            return
        
        # 车厂弹窗按钮（维修/买车）
        if name.startswith("factory_"):
            p = self.state.player
            truck_count = max(1, int(getattr(p, "truck_count", 1)))

            if name == "factory_repair_exec":
                self.factory_buy_focused = False
                repair_percent = max(0, int(round(100.0 - p.truck_durability)))
                cost = int(TRUCK_REPAIR_COST_BASE * repair_percent * truck_count)
                if repair_percent <= 0:
                    self._log("无需维修。")
                    return
                if p.cash < cost:
                    self._log("现金不足，无法维修。")
                    return
                p.cash -= cost
                p.truck_durability = 100.0
                for _ in range(TRUCK_REPAIR_DAYS):
                    result = self._do_advance_day()
                    if result is None:
                        return
                    msgs, _ = result
                    for m in msgs:
                        self._log(m)

                self._log(f"已维修：花费 {cost:,.0f} 元（{truck_count} 辆车）。耐久恢复至 100%，时间流逝 {TRUCK_REPAIR_DAYS} 天。")
                return

            if name == "factory_buy_input":
                self.factory_buy_focused = True
                return
            if name == "factory_buy":
                # 解析数量并打开二级确认弹窗
                self.factory_buy_focused = False
                try:
                    qty = max(1, int(str(getattr(self, "factory_buy_text", "1")).strip() or "1"))
                except (ValueError, TypeError):
                    qty = 1
                self.factory_buy_qty = qty
                self.factory_buy_text = str(qty)
                self.dialog_data["factory_buy_pending"] = True
                return

            if name == "factory_buy_cancel":
                self.dialog_data.pop("factory_buy_pending", None)
                return

            if name == "factory_buy_confirm":
                self.dialog_data.pop("factory_buy_pending", None)
                price_per = TRUCK_PURCHASE_PRICE
                qty = max(1, getattr(self, "factory_buy_qty", 1))
                total_cost = price_per * qty
                if p.cash < total_cost:
                    self._log("现金不足，无法购车。")
                    return
                p.cash -= total_cost
                p.truck_total_capacity += TRUCK_CAPACITY_PER_VEHICLE * qty
                try:
                    p.truck_count = int(getattr(p, "truck_count", 1)) + qty
                except Exception:
                    p.truck_count = 1 + qty
                self._log(f"已购入车辆 {qty} 辆，花费 {total_cost:,.0f} 元；总载重 +{TRUCK_CAPACITY_PER_VEHICLE * qty}（现 {p.truck_total_capacity}）。")
                return

            return
        
        # 出海弹窗按钮
        if name.startswith("sail_"):
            if name == "sail_cancel":
                self.active_dialog = None
                self.sail_target = None
            elif name in ("sail_confirm_normal", "sail_confirm_fast"):
                if not self.sail_target:
                    self._log("请先在左侧选择目的地。")
                else:
                    fast = name == "sail_confirm_fast"
                    self._sail(self.sail_target, fast=fast)
                    self.active_dialog = None
            return

        # 存档弹窗按钮
        if name.startswith("save_"):
            if name.startswith("save_slot_"):
                slot_idx = int(name.replace("save_slot_", ""))
                self.save_slot_selected = slot_idx
            elif name == "save_confirm":
                if self.save_slot_selected is not None:
                    save_game(self.state, f"slot_{self.save_slot_selected + 1}")
                    self._log(f"已保存到存档 {self.save_slot_selected + 1}。")
                    self.active_dialog = None
            elif name == "save_delete":
                if self.save_slot_selected is not None:
                    slot_name = f"slot_{self.save_slot_selected + 1}"
                    delete_game(slot_name)
                    self._log(f"已删除存档 {self.save_slot_selected + 1}。")
                    # 删除后仍留在存档界面，方便重新保存
            elif name == "save_cancel":
                self.active_dialog = None
            return
        
        # 读档弹窗按钮
        if name.startswith("load_"):
            if name.startswith("load_slot_"):
                slot_idx = int(name.replace("load_slot_", ""))
                self.load_slot_selected = slot_idx
            elif name == "load_confirm":
                if self.load_slot_selected is not None:
                    try:
                        self.state = load_game(f"slot_{self.load_slot_selected + 1}")
                        self._log(f"已读取存档 {self.load_slot_selected + 1}。")
                        self.active_dialog = None
                        if self.current_screen == "mode_select":
                            self.current_screen = "playing"
                    except FileNotFoundError:
                        self._log("存档不存在。")
            elif name == "load_cancel":
                self.active_dialog = None
            return
        
        # 区域折叠按钮（出行弹窗）
        if name.startswith("region_"):
            region_name = name.replace("region_", "")
            self.travel_region_expanded[region_name] = not self.travel_region_expanded.get(region_name, True)
            return


def run() -> int:
    window = TradeGameWindow()
    arcade.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(run())