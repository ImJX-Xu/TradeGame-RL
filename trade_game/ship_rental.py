from __future__ import annotations

from dataclasses import dataclass

from .data import CITIES


@dataclass(frozen=True, slots=True)
class ShipRates:
    daily_rent_per_ship: float = 500.0
    overdue_multiplier: float = 3.0  # 逾期按标准日租金的 3 倍计费

    @property
    def overdue_fee_per_ship_per_day(self) -> float:
        return self.daily_rent_per_ship * self.overdue_multiplier


RATES = ShipRates()


class ShipRentalError(Exception):
    pass


def has_active_contract(p) -> bool:
    return getattr(p, "ships_rented", 0) > 0


def ensure_port_city(city: str) -> None:
    if not CITIES[city].has_port:
        raise ShipRentalError("当前城市没有港口，无法租/还船")


def remaining_days(p) -> int:
    """
    当前合同剩余有效天数（含今天）。若已逾期则为 0。
    """
    end_day = getattr(p, "ship_contract_end_day", 0)
    today = getattr(p, "day", 1)
    if end_day <= 0:
        return 0
    if today > end_day:
        return 0
    return end_day - today + 1


def rent_new(p, *, ships: int, days: int, city: str) -> float:
    """
    新开租船合同：港口一次性预付租期费用。
    返回本次支付金额。
    """
    ensure_port_city(city)
    if ships <= 0 or days <= 0:
        raise ShipRentalError("租船数量/租期必须>0")
    if has_active_contract(p):
        raise ShipRentalError("已有租船合同，请在同港口加租/续租或先归还")

    cost = ships * days * RATES.daily_rent_per_ship
    if p.cash < cost:
        raise ShipRentalError("现金不足")

    p.cash = round(p.cash - cost, 2)
    p.ships_rented = int(ships)
    p.ship_rental_port = city
    p.ship_contract_end_day = int(p.day + days - 1)
    p.ship_overdue_days = 0
    return float(cost)


def extend_contract(p, *, days: int, city: str) -> float:
    """
    续租：必须在租船港口支付。
    - 未逾期：end_day += days
    - 已逾期：从“今天”重新起算 end_day = today + days - 1
    返回本次支付金额。
    """
    ensure_port_city(city)
    if not has_active_contract(p):
        raise ShipRentalError("当前未租船")
    if city != p.ship_rental_port:
        raise ShipRentalError("必须在租船的同一城市续租")
    if days <= 0:
        raise ShipRentalError("续租天数必须>0")

    cost = p.ships_rented * days * RATES.daily_rent_per_ship
    if p.cash < cost:
        raise ShipRentalError("现金不足")

    p.cash = round(p.cash - cost, 2)
    today = int(p.day)
    if today > int(p.ship_contract_end_day):
        p.ship_contract_end_day = today + days - 1
    else:
        p.ship_contract_end_day = int(p.ship_contract_end_day) + days
    return float(cost)


def add_ships(p, *, ships: int, city: str) -> float:
    """
    加租船只：必须在租船港口支付，且仅允许在未逾期状态下加租。
    新增船只的费用 = ships * 当前合同剩余天数 * 日租金。
    返回本次支付金额。
    """
    ensure_port_city(city)
    if not has_active_contract(p):
        raise ShipRentalError("当前未租船")
    if city != p.ship_rental_port:
        raise ShipRentalError("必须在租船的同一城市加租")
    if ships <= 0:
        raise ShipRentalError("加租数量必须>0")
    rem = remaining_days(p)
    if rem <= 0:
        raise ShipRentalError("合同已逾期，请先续租后再加租")

    cost = ships * rem * RATES.daily_rent_per_ship
    if p.cash < cost:
        raise ShipRentalError("现金不足")

    p.cash = round(p.cash - cost, 2)
    p.ships_rented = int(p.ships_rented) + int(ships)
    return float(cost)


def return_ships(p, *, city: str) -> None:
    """
    归还：必须在租船的同一城市。
    """
    ensure_port_city(city)
    if not has_active_contract(p):
        raise ShipRentalError("当前未租船")
    if city != p.ship_rental_port:
        raise ShipRentalError("必须在租船的同一城市归还")

    p.ships_rented = 0
    p.ship_contract_end_day = 0
    p.ship_overdue_days = 0
    p.ship_rental_port = ""


def apply_overdue_fee_for_today(p) -> float:
    """
    若已逾期，按“今天”收取一次逾期金。返回本次扣费金额（未逾期则 0）。
    """
    if not has_active_contract(p):
        return 0.0
    today = int(p.day)
    end_day = int(getattr(p, "ship_contract_end_day", 0))
    if end_day <= 0 or today <= end_day:
        return 0.0

    fee = p.ships_rented * RATES.overdue_fee_per_ship_per_day
    p.cash = round(p.cash - fee, 2)
    p.ship_overdue_days = int(getattr(p, "ship_overdue_days", 0)) + 1
    return float(fee)

