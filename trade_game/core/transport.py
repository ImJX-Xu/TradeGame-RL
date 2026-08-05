"""路线、运输报价、旅行状态变换与运输货损规则。"""

from __future__ import annotations

import heapq
from collections import OrderedDict
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
from .rules import GameRules


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


@dataclass(frozen=True, slots=True)
class TravelEstimate:
    """不消耗随机数的路线估算，供界面和智能体比较候选运输方案。"""

    origin: str
    destination: str
    mode: TransportMode
    distance_km: int
    standard_days: int
    fast_days: int
    standard_cost: Decimal
    fast_cost: Decimal
    truck_durability_loss: Decimal


_ROUTE_GRAPH_CACHE: OrderedDict[
    tuple[int, TransportMode | None],
    tuple[Catalog, dict[str, tuple[tuple[str, int], ...]]],
] = OrderedDict()
_ROUTE_GRAPH_CACHE_LIMIT = 64


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


def remote_sale_distance_premium(
    catalog: Catalog, rules: GameRules, origin_city: str, destination_city: str
) -> Decimal:
    """按真实产地到销售城市的最短路径计算固定单位流通溢价。"""

    catalog.city(origin_city)
    catalog.city(destination_city)
    if origin_city == destination_city:
        return Decimal("0")
    distance = shortest_distance_any(catalog, origin_city, destination_city)
    return money(Decimal(distance) * rules.pricing.remote_sale_premium_per_km)


def reference_sale_origin(catalog: Catalog, product: Product, destination_city: str) -> str:
    """为未持有货物的行情板选择最近货源作为出售参考价。"""

    catalog.city(destination_city)
    return min(
        product.origins,
        key=lambda origin_city: (shortest_distance_any(catalog, origin_city, destination_city), origin_city),
    )


def travel_cost(catalog: Catalog, rules: GameRules, state: GameState, command: Travel) -> Decimal:
    """返回一次旅行的确定性现金成本，并校验路线是否可行。"""

    distance = _travel_distance(catalog, rules, state, command)
    return _travel_cost(rules, state, command, distance)


def can_travel(catalog: Catalog, rules: GameRules, state: GameState, command: Travel) -> bool:
    """判断一条旅行命令在当前状态下是否可执行。"""

    try:
        return state.player.cash >= travel_cost(catalog, rules, state, command)
    except RouteNotFound:
        return False


def estimate_travel(
    catalog: Catalog, rules: GameRules, state: GameState, command: Travel
) -> TravelEstimate:
    """估算普通和快速运输的确定性成本、基准时间与耐久损耗。"""

    distance = _travel_distance(catalog, rules, state, command)
    standard_command = Travel(destination=command.destination, mode=command.mode)
    fast_command = Travel(destination=command.destination, mode=command.mode, fast=True)
    standard_days = _baseline_travel_days(rules, state, command.mode, distance)
    return TravelEstimate(
        origin=state.player.location,
        destination=command.destination,
        mode=command.mode,
        distance_km=distance,
        standard_days=standard_days,
        fast_days=max(1, standard_days // rules.transport.fast_time_divisor),
        standard_cost=_travel_cost(rules, state, standard_command, distance),
        fast_cost=_travel_cost(rules, state, fast_command, distance),
        truck_durability_loss=(
            Decimal(distance) * rules.transport.truck_durability_loss_per_km
            if command.mode is TransportMode.LAND
            else Decimal("0")
        ),
    )


def quote_travel(
    catalog: Catalog, rules: GameRules, state: GameState, command: Travel, rng: Random
) -> TravelQuote:
    """根据当前状态和随机源生成一次旅行报价。"""

    origin = state.player.location
    destination = command.destination
    distance = _travel_distance(catalog, rules, state, command)
    days = _sample_travel_days(rules, command.mode, distance, rng)
    truck_damage_ratio = Decimal("0")
    truck_durability_loss = Decimal("0")
    if command.mode is TransportMode.LAND:
        truck_damage_ratio = (Decimal("100") - state.player.truck_durability) / Decimal("100")
        days = max(
            1,
            int(
                round(
                    days
                    * float(
                        Decimal("1")
                        + rules.transport.truck_damage_time_multiplier * truck_damage_ratio
                    )
                )
            ),
        )
        truck_durability_loss = Decimal(distance) * rules.transport.truck_durability_loss_per_km
    if command.fast:
        days = max(1, days // rules.transport.fast_time_divisor)
    return TravelQuote(
        origin=origin,
        destination=destination,
        mode=command.mode,
        distance_km=distance,
        days=days,
        cost=_travel_cost(rules, state, command, distance),
        truck_damage_ratio=truck_damage_ratio,
        truck_durability_loss=truck_durability_loss,
    )


def _travel_distance(catalog: Catalog, rules: GameRules, state: GameState, command: Travel) -> int:
    origin = state.player.location
    destination = command.destination
    if destination not in catalog.cities:
        raise RouteNotFound(f"未知目标城市：{destination}")
    if origin == destination:
        raise RouteNotFound("目标城市不能是当前位置")
    if command.mode not in catalog.city(origin).modes or command.mode not in catalog.city(destination).modes:
        raise RouteNotFound("起点或终点不支持所选运输方式")
    if command.mode is TransportMode.LAND and state.player.truck_durability <= rules.transport.truck_min_durability:
        raise RouteNotFound(f"货车耐久度不高于 {rules.transport.truck_min_durability}%")
    return shortest_distance(catalog, origin, destination, command.mode)


def _travel_cost(rules: GameRules, state: GameState, command: Travel, distance: int) -> Decimal:
    if command.mode is TransportMode.LAND:
        cost = Decimal(distance) * rules.transport.land.cost_per_km * state.player.truck_count
    else:
        capacity = state.player.truck_total_capacity
        load_multiplier = Decimal("1") + Decimal(cargo_quantity(state.player.cargo_lots)) / Decimal(capacity)
        cost = Decimal(distance) * rules.transport.sea.cost_per_km * load_multiplier
    if command.fast:
        cost *= rules.transport.fast_cost_multiplier
    return money(cost)


def travel(
    catalog: Catalog, rules: GameRules, state: GameState, command: Travel, rng: Random
) -> CommandResult:
    """执行一次旅行，扣除成本、更新地点、耐久和运输货损。"""

    try:
        quote = quote_travel(catalog, rules, state, command, rng)
    except RouteNotFound as error:
        return _reject(command, state, RejectionCode.NOT_ALLOWED, str(error))
    if state.player.cash < quote.cost:
        return _reject(command, state, RejectionCode.INSUFFICIENT_CASH, "现金不足以支付运输成本")

    cargo_lots, lost_by_product = _apply_transport_loss(
        catalog, rules, state.player.cargo_lots, quote, rng
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
    key = (id(catalog), mode)
    cached = _ROUTE_GRAPH_CACHE.get(key)
    if cached is not None and cached[0] is catalog:
        _ROUTE_GRAPH_CACHE.move_to_end(key)
        return cached[1]

    graph: dict[str, list[tuple[str, int]]] = {}
    for route in catalog.routes:
        if mode is not None and route.mode is not mode:
            continue
        graph.setdefault(route.from_city, []).append((route.to_city, route.distance_km))
        graph.setdefault(route.to_city, []).append((route.from_city, route.distance_km))
    frozen_graph = {city_name: tuple(edges) for city_name, edges in graph.items()}
    _ROUTE_GRAPH_CACHE[key] = (catalog, frozen_graph)
    _ROUTE_GRAPH_CACHE.move_to_end(key)
    if len(_ROUTE_GRAPH_CACHE) > _ROUTE_GRAPH_CACHE_LIMIT:
        _ROUTE_GRAPH_CACHE.popitem(last=False)
    return frozen_graph


def _sample_travel_days(rules: GameRules, mode: TransportMode, distance_km: int, rng: Random) -> int:
    mode_rules = rules.transport.land if mode is TransportMode.LAND else rules.transport.sea
    base_days = max(1, ceil(distance_km / mode_rules.speed_km_per_day))
    factor = min(
        float(mode_rules.travel_day_max_factor),
        max(
            float(mode_rules.travel_day_min_factor),
            1.0 + rng.gauss(0.0, float(mode_rules.travel_day_standard_deviation)),
        ),
    )
    return max(1, ceil(base_days * factor))


def _baseline_travel_days(
    rules: GameRules, state: GameState, mode: TransportMode, distance_km: int
) -> int:
    mode_rules = rules.transport.land if mode is TransportMode.LAND else rules.transport.sea
    days = max(1, ceil(distance_km / mode_rules.speed_km_per_day))
    if mode is TransportMode.LAND:
        truck_damage_ratio = (Decimal("100") - state.player.truck_durability) / Decimal("100")
        days = max(
            1,
            int(
                round(
                    days
                    * float(
                        Decimal("1")
                        + rules.transport.truck_damage_time_multiplier * truck_damage_ratio
                    )
                )
            ),
        )
    return days


def _apply_transport_loss(
    catalog: Catalog, rules: GameRules, lots: tuple[CargoLot, ...], quote: TravelQuote, rng: Random
) -> tuple[tuple[CargoLot, ...], dict[str, int]]:
    retained: list[CargoLot] = []
    losses: dict[str, int] = {}
    damage_ratio = quote.truck_damage_ratio if quote.mode is TransportMode.LAND else Decimal("0")
    for lot in lots:
        product = catalog.product(lot.product_id)
        probability = _transport_loss_probability(rules, product, quote, rng)
        if quote.mode is TransportMode.LAND and damage_ratio > 0:
            probability = min(Decimal("1"), probability + product.transport_loss_rate * damage_ratio)
        lost = _sample_binomial(lot.quantity, probability, rng)
        if lost:
            losses[lot.product_id] = losses.get(lot.product_id, 0) + lost
        if lost < lot.quantity:
            retained.append(replace(lot, quantity=lot.quantity - lost))
    return tuple(retained), losses


def _transport_loss_probability(rules: GameRules, product: Product, quote: TravelQuote, rng: Random) -> Decimal:
    distance = Decimal(quote.distance_km)
    days = Decimal(quote.days)
    if product.category is ProductCategory.ELECTRONICS:
        multiplier = Decimal("1") + distance / rules.transport.loss.reference_km
    elif product.category is ProductCategory.PERISHABLE:
        multiplier = Decimal("1") + days / rules.transport.loss.reference_days
        strength = product.perishable_aging_strength
        multiplier *= Decimal("1") + strength * (days / rules.transport.loss.perishable_aging_day_scale) ** 2
    else:
        multiplier = Decimal("1") + (
            distance / rules.transport.loss.reference_km
        ) / rules.transport.loss.normal_distance_divisor
    rate = product.transport_loss_rate
    random_factor = Decimal(
        str(rng.uniform(float(rules.transport.loss.random_factor_min), float(rules.transport.loss.random_factor_max)))
    )
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
