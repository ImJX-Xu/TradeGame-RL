"""评估城市与商品对路线、风险层级和市场决策的综合必要性。"""

from __future__ import annotations

from dataclasses import dataclass
from math import log10, sqrt
from statistics import median
from typing import Mapping

from trade_game.core.catalog import Catalog
from trade_game.core.models import City, Product, SpecialtyScope
from trade_game.core.rules import GameRules
from trade_game.core.transport import shortest_distance_any


@dataclass(frozen=True, slots=True)
class CityNecessity:
    city_name: str
    score: float
    supply_identity: float
    network_value: float
    demand_value: float
    event_value: float
    distinctness: float
    observed_usage: float
    assessment: str


@dataclass(frozen=True, slots=True)
class ProductNecessity:
    product_id: str
    score: float
    supply_identity: float
    demand_coverage: float
    economic_identity: float
    market_dynamics: float
    distinctness: float
    observed_usage: float
    assessment: str


@dataclass(frozen=True, slots=True)
class NecessityReport:
    cities: tuple[CityNecessity, ...]
    products: tuple[ProductNecessity, ...]


def analyze_necessity(
    catalog: Catalog,
    rules: GameRules,
    *,
    city_usage: Mapping[str, int] | None = None,
    product_usage: Mapping[str, int] | None = None,
) -> NecessityReport:
    """将静态结构与可选的试玩使用率合并为 0-100 的必要性评分。"""

    city_usage = city_usage or {}
    product_usage = product_usage or {}
    city_rows = _city_components(catalog, rules, city_usage)
    product_rows = _product_components(catalog, rules, product_usage)
    return NecessityReport(
        cities=tuple(
            sorted(
                (
                    CityNecessity(
                        city_name=city_name,
                        score=_city_score(values, bool(city_usage)),
                        assessment=_assessment(_city_score(values, bool(city_usage))),
                        **values,
                    )
                    for city_name, values in city_rows.items()
                ),
                key=lambda item: item.score,
                reverse=True,
            )
        ),
        products=tuple(
            sorted(
                (
                    ProductNecessity(
                        product_id=product_id,
                        score=_product_score(values, bool(product_usage)),
                        assessment=_assessment(_product_score(values, bool(product_usage))),
                        **values,
                    )
                    for product_id, values in product_rows.items()
                ),
                key=lambda item: item.score,
                reverse=True,
            )
        ),
    )


def _city_components(
    catalog: Catalog,
    rules: GameRules,
    usage: Mapping[str, int],
) -> dict[str, dict[str, float]]:
    produced = {
        city_name: tuple(product for product in catalog.products.values() if city_name in product.origins)
        for city_name in catalog.cities
    }
    supply = {
        city_name: sum(
            (1.25 if product.specialty_scope is SpecialtyScope.CITY else 0.80) / len(product.origins)
            for product in products
        )
        for city_name, products in produced.items()
    }
    degree_raw = {
        city_name: len(
            {
                route.to_city if route.from_city == city_name else route.from_city
                for route in catalog.routes
                if city_name in (route.from_city, route.to_city)
            }
        )
        for city_name in catalog.cities
    }
    closeness_raw = {
        city_name: 1.0
        / sum(
            shortest_distance_any(catalog, city_name, destination)
            for destination in catalog.cities
            if destination != city_name
        )
        for city_name in catalog.cities
    }
    degree = _normalize(degree_raw)
    closeness = _normalize(closeness_raw)
    network = {
        city_name: 0.55 * degree[city_name] + 0.45 * closeness[city_name]
        for city_name in catalog.cities
    }
    demand_raw = {
        city_name: sum(
            bool(city.market_roles & product.demand_roles) and city_name not in product.origins
            for product in catalog.products.values()
        )
        for city_name, city in catalog.cities.items()
    }
    shortage_share = float(rules.market.shortage_probability)
    surplus_share = 1.0 - shortage_share
    event_raw = {
        city_name: sum(
            float(product.event_weight) * surplus_share / len(product.origins)
            for product in produced[city_name]
        )
        + sum(
            float(product.event_weight) * shortage_share
            for product in catalog.products.values()
            if city_name not in product.origins and catalog.city(city_name).market_roles & product.demand_roles
        )
        for city_name in catalog.cities
    }
    distinctness = {
        city_name: 1.0
        - max(
            _city_similarity(catalog.city(city_name), catalog.city(other), produced[city_name], produced[other])
            for other in catalog.cities
            if other != city_name
        )
        for city_name in catalog.cities
    }
    demand = _normalize(demand_raw)
    event = _normalize(event_raw)
    observed = _normalize({city_name: usage.get(city_name, 0) for city_name in catalog.cities})
    return {
        city_name: {
            "supply_identity": min(1.0, supply[city_name] / 1.25),
            "network_value": network[city_name],
            "demand_value": demand[city_name],
            "event_value": event[city_name],
            "distinctness": distinctness[city_name],
            "observed_usage": observed[city_name],
        }
        for city_name in catalog.cities
    }


def _product_components(
    catalog: Catalog,
    rules: GameRules,
    usage: Mapping[str, int],
) -> dict[str, dict[str, float]]:
    products = tuple(catalog.products.values())
    log_prices = {product.id: log10(float(product.base_purchase_price)) for product in products}
    price_median = median(log_prices.values())
    price_span = max(log_prices.values()) - min(log_prices.values())
    margins = {product.id: float(product.profit_margin_rate) for product in products}
    margin_median = median(margins.values())
    margin_span = max(margins.values()) - min(margins.values())
    supply = {
        product.id: 1.0 if product.specialty_scope is SpecialtyScope.CITY else 0.55
        for product in products
    }
    demand_raw = {
        product.id: sum(
            city_name not in product.origins and bool(city.market_roles & product.demand_roles)
            for city_name, city in catalog.cities.items()
        )
        for product in products
    }
    economic_raw = {
        product.id: (
            abs(log_prices[product.id] - price_median) / price_span
            + abs(margins[product.id] - margin_median) / margin_span
        )
        for product in products
    }
    dynamics_raw = {}
    for product in products:
        trend_scale = (
            float(product.trend_sigma)
            / sqrt(1.0 - float(product.trend_persistence) ** 2)
            * float(rules.market.trend_range_share)
        )
        local_scale = float(product.local_spread_sigma) / sqrt(
            1.0 - float(rules.market.local_spread_persistence) ** 2
        )
        average_event_amplitude = float(
            (product.event_amplitude_min + product.event_amplitude_max) / 2
        )
        event_scale = (
            float(product.event_weight)
            * average_event_amplitude
            * float(rules.market.event_spawn_probability)
        )
        dynamics_raw[product.id] = trend_scale + local_scale + event_scale
    distinctness = {
        product.id: 1.0
        - max(
            _product_similarity(product, other, log_prices, margins)
            for other in products
            if other.id != product.id
        )
        for product in products
    }
    demand = _normalize(demand_raw)
    economic = _normalize(economic_raw)
    dynamics = _normalize(dynamics_raw)
    observed = _normalize({product.id: usage.get(product.id, 0) for product in products})
    return {
        product.id: {
            "supply_identity": supply[product.id],
            "demand_coverage": demand[product.id],
            "economic_identity": economic[product.id],
            "market_dynamics": dynamics[product.id],
            "distinctness": distinctness[product.id],
            "observed_usage": observed[product.id],
        }
        for product in products
    }


def _city_score(values: Mapping[str, float], with_usage: bool) -> float:
    weights = {
        "supply_identity": 0.25,
        "network_value": 0.18,
        "demand_value": 0.16,
        "event_value": 0.14,
        "distinctness": 0.17,
        "observed_usage": 0.10 if with_usage else 0.0,
    }
    scale = sum(weights.values())
    return round(100 * sum(values[name] * weight for name, weight in weights.items()) / scale, 2)


def _product_score(values: Mapping[str, float], with_usage: bool) -> float:
    weights = {
        "supply_identity": 0.20,
        "demand_coverage": 0.16,
        "economic_identity": 0.14,
        "market_dynamics": 0.14,
        "distinctness": 0.26,
        "observed_usage": 0.10 if with_usage else 0.0,
    }
    scale = sum(weights.values())
    return round(100 * sum(values[name] * weight for name, weight in weights.items()) / scale, 2)


def _city_similarity(
    city: City,
    other: City,
    products: tuple[Product, ...],
    other_products: tuple[Product, ...],
) -> float:
    product_categories = {product.category for product in products}
    other_categories = {product.category for product in other_products}
    return (
        0.35 * _jaccard(city.market_roles, other.market_roles)
        + 0.25 * _jaccard(city.modes, other.modes)
        + 0.25 * _jaccard(product_categories, other_categories)
        + 0.15 * float(city.region == other.region)
    )


def _product_similarity(
    product: Product,
    other: Product,
    log_prices: Mapping[str, float],
    margins: Mapping[str, float],
) -> float:
    price_similarity = 1.0 / (1.0 + abs(log_prices[product.id] - log_prices[other.id]))
    margin_similarity = 1.0 / (1.0 + 8.0 * abs(margins[product.id] - margins[other.id]))
    return (
        0.28 * float(product.category is other.category)
        + 0.20 * price_similarity
        + 0.17 * margin_similarity
        + 0.20 * _jaccard(product.demand_roles, other.demand_roles)
        + 0.15 * float(product.specialty_scope is other.specialty_scope)
    )


def _jaccard(left, right) -> float:
    union = set(left) | set(right)
    return len(set(left) & set(right)) / len(union) if union else 0.0


def _normalize(values: Mapping[str, float | int]) -> dict[str, float]:
    minimum = min(float(value) for value in values.values())
    maximum = max(float(value) for value in values.values())
    if maximum == minimum:
        return {name: 0.5 for name in values}
    return {name: (float(value) - minimum) / (maximum - minimum) for name, value in values.items()}


def _assessment(score: float) -> str:
    if score >= 65:
        return "核心"
    if score >= 45:
        return "有效"
    if score >= 30:
        return "可优化"
    return "需审视"
