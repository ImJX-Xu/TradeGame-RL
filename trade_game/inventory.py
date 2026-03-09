from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Dict, Iterable, List, Tuple

from .data import PRODUCTS, ProductCategory
from .data import Product
from .game_config import (
    LOSS_REF_DAYS,
    LOSS_REF_KM,
    TRANSPORT_LOSS_RANDOM_FACTOR_MAX,
    TRANSPORT_LOSS_RANDOM_FACTOR_MIN,
)

# 生鲜额外“时间老化”强度（根据商品 id 区分，越大代表越容易变质）
PERISHABLE_AGING_STRENGTH: dict[str, float] = {
    # 设计目标（车辆状态良好时，单次运输 10 天左右）：
    # - 福州带鱼：损失概率接近 95%
    # - 海南芒果：约 80%
    # - 高雄凤梨酥：约 60%
    # - 海岛椰子糖：约 40%
    # - 东北榛蘑：约 25%（干货，最耐放）
    "fz_fish": 3.0,
    "hn_mango": 4.5,
    "ks_pineapple": 5.5,
    "island_candy": 5.5,
    "db_mushroom": 7.5,
}


def _sample_poisson(lam: float, rng: Random) -> int:
    """
    采样 Poisson(λ)（Knuth 算法）。
    仅在 λ 较小（例如 < 30）时使用更合适。
    """
    if lam <= 0:
        return 0
    # Knuth: O(k) where k ~ λ
    from math import exp

    L = exp(-lam)
    k = 0
    p = 1.0
    while p > L:
        k += 1
        p *= rng.random()
    return k - 1


def _sample_binomial(n: int, p: float, rng: Random) -> int:
    """
    采样 Binomial(n, p)。
    - 小 n：逐次伯努利（精确）
    - 小 p 且 n*p 小：Poisson 近似
    - 其余：Normal 近似（截断到 [0, n]）
    """
    if n <= 0:
        return 0
    if p <= 0.0:
        return 0
    if p >= 1.0:
        return n

    # 小规模直接模拟，避免近似误差
    if n <= 64:
        k = 0
        for _ in range(n):
            if rng.random() < p:
                k += 1
        return k

    mean = n * p

    # 小概率、小期望：Poisson 近似更好且更快
    if p <= 0.05 and mean <= 20.0:
        k = _sample_poisson(mean, rng)
        return max(0, min(n, k))

    # 正态近似
    from math import sqrt

    var = n * p * (1.0 - p)
    if var <= 0:
        return int(round(mean))
    k = int(round(rng.gauss(mean, sqrt(var))))
    return max(0, min(n, k))


@dataclass(slots=True)
class CargoLot:
    product_id: str
    quantity: int
    origin_city: str
    shelf_life_remaining_days: int | None = None

    def is_perishable(self) -> bool:
        # 新版枚举中，生鲜特产使用 ProductCategory.PERISHABLE
        return PRODUCTS[self.product_id].category == ProductCategory.PERISHABLE


def cargo_used(lots: Iterable[CargoLot]) -> int:
    return sum(l.quantity for l in lots)


def add_lot(lots: List[CargoLot], lot: CargoLot) -> None:
    """
    合并同类批次（同商品、同产地、同保质期）以简化显示。
    """
    if lot.quantity <= 0:
        return
    for existing in lots:
        if (
            existing.product_id == lot.product_id
            and existing.origin_city == lot.origin_city
            and existing.shelf_life_remaining_days == lot.shelf_life_remaining_days
        ):
            existing.quantity += lot.quantity
            return
    lots.append(lot)


def remove_quantity_fifo(
    lots: List[CargoLot], product_id: str, quantity: int
) -> Tuple[int, List[CargoLot]]:
    """
    以 FIFO 卖出/移除指定数量，返回 (实际移除数量, 本次移除的批次列表[按实际扣减后数量]).
    """
    if quantity <= 0:
        return 0, []

    removed: List[CargoLot] = []
    remaining = quantity
    i = 0
    while i < len(lots) and remaining > 0:
        lot = lots[i]
        if lot.product_id != product_id or lot.quantity <= 0:
            i += 1
            continue

        take = lot.quantity if lot.quantity <= remaining else remaining
        lot.quantity -= take
        remaining -= take
        removed.append(
            CargoLot(
                product_id=lot.product_id,
                quantity=take,
                origin_city=lot.origin_city,
                shelf_life_remaining_days=lot.shelf_life_remaining_days,
            )
        )
        if lot.quantity == 0:
            lots.pop(i)
            continue
        i += 1

    return quantity - remaining, removed


def wipe_to_capacity_lifo(lots: List[CargoLot], max_units: int) -> int:
    """
    超出容量时，从“最新批次”开始无补偿清空，直到占用 <= max_units。
    返回被清空的单位数。
    """
    if max_units < 0:
        max_units = 0
    removed = 0
    excess = cargo_used(lots) - max_units
    while excess > 0 and lots:
        lot = lots[-1]
        take = lot.quantity if lot.quantity <= excess else excess
        lot.quantity -= take
        removed += take
        excess -= take
        if lot.quantity == 0:
            lots.pop()
    return removed


def decay_shelf_life(lots: List[CargoLot], days: int) -> int:
    """
    保质期衰减（按天）。返回“变质清空”的单位数。
    """
    if days <= 0:
        return 0
    spoiled = 0
    i = 0
    while i < len(lots):
        lot = lots[i]
        if lot.shelf_life_remaining_days is None:
            i += 1
            continue
        lot.shelf_life_remaining_days -= days
        if lot.shelf_life_remaining_days <= 0:
            spoiled += lot.quantity
            lots.pop(i)
            continue
        i += 1
    return spoiled


def _transport_loss_multiplier(prod: Product, km: float, days: int) -> float:
    """
    运输损耗乘数：
    - 易损：与里程成正比，mult_e = 1 + km / LOSS_REF_KM
    - 生鲜：
        · 基础时间乘数：1 + days / LOSS_REF_DAYS
        · 额外老化乘数：1 + S_prod × (days / 10)^2 （S_prod 由 PERISHABLE_AGING_STRENGTH 控制）
          使易腐商品在 10 天左右运输时损耗率大幅升高，近似“运途中过期”效果。
    - 普通（基础民生 + 轻工）：沿用易损逻辑，但强度为其 1/3：
        mult_n = 1 + (km / LOSS_REF_KM) / 3
      在相同基础损耗率和里程下，普通商品的有效损耗率约为易损类的 1/3。
    """
    if prod.category == ProductCategory.ELECTRONICS:
        return 1.0 + km / LOSS_REF_KM
    if prod.category == ProductCategory.PERISHABLE:
        base = 1.0 + days / LOSS_REF_DAYS
        strength = PERISHABLE_AGING_STRENGTH.get(prod.id, 0.0)
        if strength <= 0.0 or days <= 0:
            return base
        aging = 1.0 + strength * (days / 10.0) ** 2
        return base * aging
    # 普通商品：BASE / LIGHT_INDUSTRY 等
    return 1.0 + (km / LOSS_REF_KM) / 3.0


def expected_transport_loss_display(km: float, days: int) -> str:
    """
    预计运损展示用（汇总）：
    - 易损：从商品数据取 electronics 类典型基础率，乘以里程乘数
    - 轻工：从商品数据取 light_industry 类典型基础率，乘数 1/3
    - 民用：从商品数据取 base 类典型基础率，乘数 1/3
    （生鲜在 UI 中按商品逐项列出，这里不再汇总）
    """
    # 从实际商品数据动态获取各类别的基础损耗率（取第一个商品的损耗率作为代表）
    base_rate_electronics = 0.05  # 默认值
    base_rate_light = 0.03  # 默认值
    base_rate_base = 0.005  # 默认值
    
    for prod in PRODUCTS.values():
        if prod.category == ProductCategory.ELECTRONICS and base_rate_electronics == 0.05:
            base_rate_electronics = prod.transport_loss_rate
        elif prod.category == ProductCategory.LIGHT_INDUSTRY and base_rate_light == 0.03:
            base_rate_light = prod.transport_loss_rate
        elif prod.category == ProductCategory.BASE and base_rate_base == 0.005:
            base_rate_base = prod.transport_loss_rate
    
    elec_mult = 1.0 + km / LOSS_REF_KM
    mult_n = 1.0 + (km / LOSS_REF_KM) / 3.0
    pct_e = base_rate_electronics * elec_mult * 100
    pct_light = base_rate_light * mult_n * 100
    pct_base = base_rate_base * mult_n * 100
    lines = [
        f"易损：约 {pct_e:.1f}%",
        f"轻工：约 {pct_light:.1f}%",
        f"民用：约 {pct_base:.1f}%",
    ]
    return "\n".join(lines)


def expected_perishable_loss_details(km: float, days: int) -> List[Tuple[str, float]]:
    """
    返回所有生鲜商品在当前路线下的预计损耗率明细（仅用于 UI 展示，不参与结算）。

    使用与实际运输损耗接近的近似：
    - 概率 p ≈ transport_loss_rate × 生鲜时间乘数（不包含随机因子）
    - 并截断到 [0, 0.99]，然后转为百分比。
    """
    details: List[Tuple[str, float]] = []
    for pid, prod in PRODUCTS.items():
        if prod.category != ProductCategory.PERISHABLE:
            continue
        base_rate = float(prod.transport_loss_rate)
        if base_rate <= 0.0:
            continue
        mult = _transport_loss_multiplier(prod, km, days)
        p = max(0.0, min(0.99, base_rate * mult))
        pct = p * 100.0
        details.append((prod.name, pct))
    # 按预计损耗率从高到低排序，方便玩家一眼看到最容易坏的
    details.sort(key=lambda x: x[1], reverse=True)
    return details


def apply_transport_loss(
    lots: List[CargoLot],
    *,
    origin_city: str,
    target_city: str,
    km: float,
    days: int,
    rng: Random,
    loss_stats: Dict[str, int] | None = None,
) -> int:
    """
    运输损耗（按“每趟运输”结算）。
    - 基础：商品 transport_loss_rate，台湾路线 +1%。
    - 预计运损乘数：易损类与里程成正比，生鲜与时间成正比。
    - 实际运损：将“损耗率”视为“单件损坏概率”，对每个批次按二项分布采样损坏数量。
      同时，整体再乘 game_config 中 TRANSPORT_LOSS_RANDOM_FACTOR_MIN～MAX 的随机乘数。
    返回实际损耗掉的单位数。
    """
    is_taiwan = origin_city in ("台北", "高雄") or target_city in ("台北", "高雄")
    random_factor = rng.uniform(TRANSPORT_LOSS_RANDOM_FACTOR_MIN, TRANSPORT_LOSS_RANDOM_FACTOR_MAX)
    lost = 0
    for lot in list(lots):
        if lot.quantity <= 0:
            continue

        prod = PRODUCTS[lot.product_id]
        base_rate = float(prod.transport_loss_rate)
        if is_taiwan:
            base_rate += 0.01
        if base_rate <= 0.0:
            continue

        mult = _transport_loss_multiplier(prod, km, days)
        # “单件损坏概率” = base_rate * mult * random_factor，截断到 [0, 1]
        p = base_rate * mult * random_factor
        if p <= 0.0:
            continue
        if p >= 1.0:
            delta = lot.quantity
        else:
            delta = _sample_binomial(lot.quantity, p, rng)

        if delta <= 0:
            continue

        lot.quantity -= delta
        lost += delta
        if loss_stats is not None:
            loss_stats[prod.id] = loss_stats.get(prod.id, 0) + delta

    lots[:] = [l for l in lots if l.quantity > 0]
    return lost

