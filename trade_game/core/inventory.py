"""不可变库存批次的聚合、追加与 FIFO 扣减函数。"""

from __future__ import annotations

from .models import CargoLot


def cargo_quantity(lots: tuple[CargoLot, ...], product_id: str | None = None) -> int:
    """返回全部库存或指定商品的总数量。"""

    return sum(lot.quantity for lot in lots if product_id is None or lot.product_id == product_id)


def free_capacity(lots: tuple[CargoLot, ...], total_capacity: int) -> int:
    """按当前单位载重计算剩余可用容量。"""

    if total_capacity < 0:
        raise ValueError("总容量不能为负")
    return max(0, total_capacity - cargo_quantity(lots))


def add_cargo(lots: tuple[CargoLot, ...], added_lot: CargoLot) -> tuple[CargoLot, ...]:
    """加入一批货物；仅完全相同的批次元数据可以合并。"""

    merged: list[CargoLot] = []
    found_equivalent_lot = False
    for lot in lots:
        if (
            lot.product_id == added_lot.product_id
            and lot.origin_city == added_lot.origin_city
            and lot.shelf_life_remaining_days == added_lot.shelf_life_remaining_days
            and lot.age_days == added_lot.age_days
        ):
            merged.append(
                CargoLot(
                    product_id=lot.product_id,
                    quantity=lot.quantity + added_lot.quantity,
                    origin_city=lot.origin_city,
                    shelf_life_remaining_days=lot.shelf_life_remaining_days,
                    age_days=lot.age_days,
                )
            )
            found_equivalent_lot = True
        else:
            merged.append(lot)
    if not found_equivalent_lot:
        merged.append(added_lot)
    return tuple(merged)


def remove_cargo_fifo(
    lots: tuple[CargoLot, ...], product_id: str, quantity: int
) -> tuple[tuple[CargoLot, ...], tuple[CargoLot, ...]]:
    """按先进先出规则扣减精确数量，返回剩余和被移除的批次。"""

    if quantity <= 0:
        raise ValueError("扣减数量必须大于 0")
    if cargo_quantity(lots, product_id) < quantity:
        raise ValueError("库存不足")

    remaining_quantity = quantity
    remaining: list[CargoLot] = []
    removed: list[CargoLot] = []
    for lot in lots:
        if lot.product_id != product_id or remaining_quantity == 0:
            remaining.append(lot)
            continue
        taken = min(lot.quantity, remaining_quantity)
        remaining_quantity -= taken
        removed.append(
            CargoLot(
                product_id=lot.product_id,
                quantity=taken,
                origin_city=lot.origin_city,
                shelf_life_remaining_days=lot.shelf_life_remaining_days,
                age_days=lot.age_days,
            )
        )
        if taken < lot.quantity:
            remaining.append(
                CargoLot(
                    product_id=lot.product_id,
                    quantity=lot.quantity - taken,
                    origin_city=lot.origin_city,
                    shelf_life_remaining_days=lot.shelf_life_remaining_days,
                    age_days=lot.age_days,
                )
            )
    return tuple(remaining), tuple(removed)
