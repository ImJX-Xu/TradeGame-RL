"""路线、运输报价、旅行状态变换与运输货损规则。"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, replace
from decimal import Decimal
from math import ceil, exp, sqrt
from random import Random

from .catalog import Catalog
from .commands import Travel
from .inventory import cargo_quantity
from .models import CargoLot, GameState, Product, ProductCategory, TransportMode
from .price_functions import money
from .results import CommandRejection, CommandResult, GameEvent, RejectionCode


# 这些平衡参数将在下一阶段迁入 rules.toml；此处集中定义以保证规则没有散落常量。
LAND_COST_PER_KM = Decimal("0.15")
SEA_COST_PER_KM = Decimal("0.60")
LAND_SPEED_KM_PER_DAY = 400
SEA_SPEED_KM_PER_DAY = 250
FAST_TRAVEL_COST_MULTIPLIER = Decimal("3")
FAST_TRAVEL_TIME_DIVISOR = 3
TRUCK_MIN_DURABILITY_FOR_TRAVEL = Decimal("30")
TRUCK_DURABILITY_LOSS_PER_KM = Decimal("0.001")
REMOTE_SALE_MULTIPLIER_MIN = Decimal("1")
REMOTE_SALE_MULTIPLIER_MAX = Decimal("1.5")

_LOSS_REF_KM = Decimal("2000")
_LOSS_REF_DAYS = Decimal("5")
_LOSS_RANDOM_FACTOR_MIN = Decimal("0.9")
_LOSS_RANDOM_FACTOR_MAX = Decimal("1.1")
_PERISHABLE_AGING_STRENGTH = {
    "fz_fish": Decimal("3"),
    "hn_mango": Decimal("4.5"),
    "ks_pineapple": Decimal("5.5"),
    "island_candy": Decimal("5.5"),
    "db_mushroom": Decimal("7.5"),
}


class RouteNotFound(ValueError):
    """用户选择的运输方式无法到达目标城市。"""


@dataclass(frozen=True, slots=True)
class TravelQuote:
    """一次可执行旅行的结算前报价。"""

    origin: str
    destination: str
    mode: TransportMode
    distance_km: int
    days: int
    cost: Decimal
    truck_damage_ratio: Decimal
    truck_durability_loss: Decimal


def shortest_distance(
    catalog: Catalog, origin: str, destination: str, mode: TransportMode
) -> int:
    """计算只使用一种运输方式时的最短路线距离。"""

    catalog.city(origin)
    catalog.city(destination)
    if origin == destination:
        return 0
    graph = _route_graph(catalog, mode)
    queue: list[tuple[int, str]] = [(0, origin)]
    distances = {origin: 0}
    while queue:
        distance, city_name = heapq.heappop(queue)
        if city_name == destination:
            return distance
        if distance != distances[city_name]:
            continue
        for next_city, edge_distance in graph.get(city_name, ()):
            candidate = distance + edge_distance
            if candidate < distances.get(next_city, 10**18):
                distances[next_city] = candidate
                heapq.heappush(queue, (candidate, next_city))
    raise RouteNotFound(f"不存在 {mode.value} 路线：{origin} -> {destination}")


def shortest_distance_any(catalog: Catalog, origin: str, destination: str) -> int:
    """计算允许换乘陆运和海运时的最短路线距离，用于市场距离定价。"""

    catalog.city(origin)
    catalog.city(destination)
    if origin == destination:
        return 0
    graph = _route_graph(catalog, None)
    queue: list[tuple[int, str]] = [(0, origin)]
    distances = {origin: 0}
    while queue:
        distance, city_name = heapq.heappop(queue)
        if city_name == destination:
            return distance
        if distance != distances[city_name]:
            continue
        for next_city, edge_distance in graph.get(city_name, ()):
            candidate = distance + edge_distance
            if candidate < distances.get(next_city, 10**18):
                distances[next_city] = candidate
                heapq.heappush(queue, (candidate, next_city))
    raise RuntimeError(f"静态路线图不连通：{origin} -> {destination}")


def remote_sale_distance_multiplier(catalog: Catalog, product: Product, city_name: str) -> Decimal:
    """按产地到销售城市的最远最短路径距离计算异地销售乘数。"""

    if city_name in product.origins:
        return Decimal("1")
    all_distances = [
        shortest_distance_any(catalog, city_a, city_b)
        for index, city_a in enumerate(catalog.cities)
        for city_b in tuple(catalog.cities)[index + 1 :]
    ]
    minimum = min(all_distances)
    maximum = max(all_distances)
    distance = max(shortest_distance_any(catalog, origin, city_name) for origin in product.origins)
    if maximum == minimum:
        return REMOTE_SALE_MULTIPLIER_MIN
    ratio = Decimal(distance - minimum) / Decimal(maximum - minimum)
    return REMOTE_SALE_MULTIPLIER_MIN + ratio * (
        REMOTE_SALE_MULTIPLIER_MAX - REMOTE_SALE_MULTIPLIER_MIN
    )


def quote_travel(catalog: Catalog, state: GameState, command: Travel, rng: Random) -> TravelQuote:
    """根据当前状态和随机源生成一次旅行报价。"""

    origin = state.player.location
    destination = command.destination
    if destination not in catalog.cities:
        raise RouteNotFound(f"未知目标城市：{destination}")
    if origin == destination:
        raise RouteNotFound("目标城市不能是当前位置")
    if command.mode not in catalog.city(origin).modes or command.mode not in catalog.city(destination).modes:
        raise RouteNotFound("起点或终点不支持所选运输方式")
    if command.mode is TransportMode.LAND and state.player.truck_durability <= TRUCK_MIN_DURABILITY_FOR_TRAVEL:
        raise RouteNotFound(f"货车耐久度不高于 {TRUCK_MIN_DURABILITY_FOR_TRAVEL}%")

    distance = shortest_distance(catalog, origin, destination, command.mode)
    days = _sample_travel_days(command.mode, distance, rng)
    truck_damage_ratio = Decimal("0")
    truck_durability_loss = Decimal("0")
    if command.mode is TransportMode.LAND:
        truck_damage_ratio = (Decimal("100") - state.player.truck_durability) / Decimal("100")
        days = max(1, int(round(days * float(Decimal("1") + Decimal("0.5") * truck_damage_ratio))))
        truck_durability_loss = Decimal(distance) * TRUCK_DURABILITY_LOSS_PER_KM
        cost = Decimal(distance) * LAND_COST_PER_KM * state.player.truck_count
    else:
        capacity = state.player.truck_total_capacity
        load_multiplier = Decimal("1") + Decimal(cargo_quantity(state.player.cargo_lots)) / Decimal(capacity)
        cost = Decimal(distance) * SEA_COST_PER_KM * load_multiplier
    if command.fast:
        days = max(1, days // FAST_TRAVEL_TIME_DIVISOR)
        cost *= FAST_TRAVEL_COST_MULTIPLIER
    return TravelQuote(
        origin=origin,
        destination=destination,
        mode=command.mode,
        distance_km=distance,
        days=days,
        cost=money(cost),
        truck_damage_ratio=truck_damage_ratio,
        truck_durability_loss=truck_durability_loss,
    )


def travel(catalog: Catalog, state: GameState, command: Travel, rng: Random) -> CommandResult:
    """执行一次旅行，扣除成本、更新地点、耐久和运输货损。"""

    try:
        quote = quote_travel(catalog, state, command, rng)
    except RouteNotFound as error:
        return _reject(command, state, RejectionCode.NOT_ALLOWED, str(error))
    if state.player.cash < quote.cost:
        return _reject(command, state, RejectionCode.INSUFFICIENT_CASH, "现金不足以支付运输成本")

    cargo_lots, lost_by_product = _apply_transport_loss(
        catalog, state.player.cargo_lots, quote, rng
    )
    updated_losses = dict(state.loss_by_product)
    for product_id, quantity in lost_by_product.items():
        updated_losses[product_id] = updated_losses.get(product_id, 0) + quantity
    player = replace(
        state.player,
        cash=state.player.cash - quote.cost,
        location=quote.destination,
        truck_durability=max(Decimal("0"), state.player.truck_durability - quote.truck_durability_loss),
        cargo_lots=cargo_lots,
    )
    next_state = replace(
        state,
        player=player,
        day=state.day + quote.days,
        visited_cities=state.visited_cities | {quote.destination},
        loss_by_product=updated_losses,
    )
    events = [
        GameEvent(
            "travel_completed",
            {
                "origin": quote.origin,
                "destination": quote.destination,
                "mode": quote.mode.value,
                "distance_km": quote.distance_km,
                "days": quote.days,
                "cost": quote.cost,
            },
        )
    ]
    if lost_by_product:
        events.append(GameEvent("cargo_lost_in_transit", {"quantity": sum(lost_by_product.values())}))
    return CommandResult.succeed(command, next_state, *events)


def _route_graph(catalog: Catalog, mode: TransportMode | None) -> dict[str, tuple[tuple[str, int], ...]]:
    graph: dict[str, list[tuple[str, int]]] = {}
    for route in catalog.routes:
        if mode is not None and route.mode is not mode:
            continue
        graph.setdefault(route.from_city, []).append((route.to_city, route.distance_km))
        graph.setdefault(route.to_city, []).append((route.from_city, route.distance_km))
    return {city_name: tuple(edges) for city_name, edges in graph.items()}


def _sample_travel_days(mode: TransportMode, distance_km: int, rng: Random) -> int:
    speed = LAND_SPEED_KM_PER_DAY if mode is TransportMode.LAND else SEA_SPEED_KM_PER_DAY
    base_days = max(1, ceil(distance_km / speed))
    standard_deviation = 0.15 if mode is TransportMode.LAND else 0.20
    minimum_factor = 0.55 if mode is TransportMode.LAND else 0.40
    maximum_factor = 1.45 if mode is TransportMode.LAND else 1.60
    factor = min(maximum_factor, max(minimum_factor, 1.0 + rng.gauss(0.0, standard_deviation)))
    return max(1, ceil(base_days * factor))


def _apply_transport_loss(
    catalog: Catalog, lots: tuple[CargoLot, ...], quote: TravelQuote, rng: Random
) -> tuple[tuple[CargoLot, ...], dict[str, int]]:
    retained: list[CargoLot] = []
    losses: dict[str, int] = {}
    damage_ratio = quote.truck_damage_ratio if quote.mode is TransportMode.LAND else Decimal("0")
    for lot in lots:
        product = catalog.product(lot.product_id)
        probability = _transport_loss_probability(product, quote, rng)
        if quote.mode is TransportMode.LAND and damage_ratio > 0:
            probability = min(Decimal("1"), probability + product.transport_loss_rate * damage_ratio)
        lost = _sample_binomial(lot.quantity, probability, rng)
        if lost:
            losses[lot.product_id] = losses.get(lot.product_id, 0) + lost
        if lost < lot.quantity:
            retained.append(replace(lot, quantity=lot.quantity - lost))
    return tuple(retained), losses


def _transport_loss_probability(product: Product, quote: TravelQuote, rng: Random) -> Decimal:
    distance = Decimal(quote.distance_km)
    days = Decimal(quote.days)
    if product.category is ProductCategory.ELECTRONICS:
        multiplier = Decimal("1") + distance / _LOSS_REF_KM
    elif product.category is ProductCategory.PERISHABLE:
        multiplier = Decimal("1") + days / _LOSS_REF_DAYS
        strength = _PERISHABLE_AGING_STRENGTH[product.id]
        multiplier *= Decimal("1") + strength * (days / Decimal("10")) ** 2
    else:
        multiplier = Decimal("1") + (distance / _LOSS_REF_KM) / Decimal("3")
    rate = product.transport_loss_rate
    random_factor = Decimal(str(rng.uniform(float(_LOSS_RANDOM_FACTOR_MIN), float(_LOSS_RANDOM_FACTOR_MAX))))
    return min(Decimal("1"), max(Decimal("0"), rate * multiplier * random_factor))


def _sample_binomial(total: int, probability: Decimal, rng: Random) -> int:
    if probability <= 0:
        return 0
    if probability >= 1:
        return total
    if total <= 64:
        return sum(rng.random() < float(probability) for _ in range(total))
    mean = total * float(probability)
    if mean <= 20 and probability <= Decimal("0.05"):
        return min(total, _sample_poisson(mean, rng))
    variance = total * float(probability) * (1.0 - float(probability))
    return max(0, min(total, round(rng.gauss(mean, sqrt(variance)))))


def _sample_poisson(expected: float, rng: Random) -> int:
    threshold = exp(-expected)
    product = 1.0
    count = 0
    while product > threshold:
        count += 1
        product *= rng.random()
    return count - 1


def _reject(command: Travel, state: GameState, code: RejectionCode, message: str) -> CommandResult:
    return CommandResult.reject(command, state, CommandRejection(code, message))
