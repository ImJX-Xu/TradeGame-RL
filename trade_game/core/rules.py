"""加载、校验并冻结游戏全局数值规则。"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from importlib.resources import files
from pathlib import Path
import tomllib
from typing import Any, Mapping

from .catalog import Catalog


class GameRulesError(ValueError):
    """rules.toml 缺失、格式错误或不满足数值约束时抛出。"""


@dataclass(frozen=True, slots=True)
class InitialStateRules:
    initial_cash: Decimal
    initial_location: str
    initial_day: int
    initial_truck_count: int
    initial_truck_capacity: int
    initial_truck_durability: Decimal


@dataclass(frozen=True, slots=True)
class GameLimits:
    challenge_max_days: int


@dataclass(frozen=True, slots=True)
class PricingRules:
    high_consumption_multiplier: Decimal
    remote_sale_premium_per_km: Decimal


@dataclass(frozen=True, slots=True)
class MarketRules:
    trend_range_share: Decimal
    local_spread_persistence: Decimal
    event_spawn_probability: Decimal
    max_active_events: int
    event_ramp_days: int
    shortage_probability: Decimal
    regional_scope_probability: Decimal
    purchase_cooldown_days: int


@dataclass(frozen=True, slots=True)
class TransportModeRules:
    speed_km_per_day: int
    cost_per_km: Decimal
    travel_day_standard_deviation: Decimal
    travel_day_min_factor: Decimal
    travel_day_max_factor: Decimal


@dataclass(frozen=True, slots=True)
class TransportLossRules:
    reference_km: Decimal
    reference_days: Decimal
    random_factor_min: Decimal
    random_factor_max: Decimal
    normal_distance_divisor: Decimal
    perishable_aging_day_scale: Decimal


@dataclass(frozen=True, slots=True)
class TransportRules:
    land: TransportModeRules
    sea: TransportModeRules
    fast_cost_multiplier: Decimal
    fast_time_divisor: int
    truck_min_durability: Decimal
    truck_durability_loss_per_km: Decimal
    truck_damage_time_multiplier: Decimal
    loss: TransportLossRules


@dataclass(frozen=True, slots=True)
class VehicleRules:
    purchase_price: Decimal
    capacity_per_vehicle: int
    repair_cost_per_percent: Decimal
    repair_days: int
    daily_labor_cost_per_extra_truck: Decimal


@dataclass(frozen=True, slots=True)
class FinanceRules:
    daily_interest_rate: Decimal
    loan_asset_multiplier: Decimal


@dataclass(frozen=True, slots=True)
class SettlementRules:
    additional_truck_residual_value: Decimal


@dataclass(frozen=True, slots=True)
class GameRules:
    """所有非实体级、可调的游戏规则数值。"""

    schema_version: int
    balance_version: str
    initial: InitialStateRules
    limits: GameLimits
    pricing: PricingRules
    market: MarketRules
    transport: TransportRules
    vehicles: VehicleRules
    finance: FinanceRules
    settlement: SettlementRules


def load_default_game_rules(catalog: Catalog) -> GameRules:
    """加载包内唯一默认规则文件。"""

    return load_game_rules(catalog, Path(str(files("trade_game.core.data") / "rules.toml")))


def load_game_rules(catalog: Catalog, path: Path) -> GameRules:
    """从 TOML 加载全局规则并验证其与静态目录一致。"""

    try:
        with path.open("rb") as file:
            raw = tomllib.load(file)
    except FileNotFoundError as error:
        raise GameRulesError(f"缺少游戏规则文件：{path}") from error
    except tomllib.TOMLDecodeError as error:
        raise GameRulesError(f"{path.name} TOML 格式错误：{error}") from error

    _expect_exact_keys(
        raw,
        {
            "schema_version",
            "balance_version",
            "game",
            "pricing",
            "market",
            "transport",
            "vehicles",
            "finance",
            "settlement",
        },
        "根配置",
    )
    schema_version = _integer(raw["schema_version"], "schema_version", minimum=1)
    if schema_version != 1:
        raise GameRulesError(f"不支持的 rules.toml 版本：{schema_version}")
    balance_version = _text(raw["balance_version"], "balance_version")
    game = _mapping(raw["game"], "game")
    pricing = _mapping(raw["pricing"], "pricing")
    market = _mapping(raw["market"], "market")
    transport = _mapping(raw["transport"], "transport")
    vehicles = _mapping(raw["vehicles"], "vehicles")
    finance = _mapping(raw["finance"], "finance")
    settlement = _mapping(raw["settlement"], "settlement")

    initial, limits = _parse_game(game, catalog)
    return GameRules(
        schema_version=schema_version,
        balance_version=balance_version,
        initial=initial,
        limits=limits,
        pricing=_parse_pricing(pricing),
        market=_parse_market(market),
        transport=_parse_transport(transport),
        vehicles=_parse_vehicles(vehicles),
        finance=_parse_finance(finance),
        settlement=_parse_settlement(settlement),
    )


def _parse_game(raw: Mapping[str, Any], catalog: Catalog) -> tuple[InitialStateRules, GameLimits]:
    _expect_exact_keys(
        raw,
        {
            "initial_cash",
            "initial_location",
            "initial_day",
            "initial_truck_count",
            "initial_truck_capacity",
            "initial_truck_durability",
            "challenge_max_days",
        },
        "game",
    )
    location = _text(raw["initial_location"], "game.initial_location")
    if location not in catalog.cities:
        raise GameRulesError(f"初始城市不存在：{location}")
    initial = InitialStateRules(
        initial_cash=_decimal(raw["initial_cash"], "game.initial_cash", minimum=Decimal("0")),
        initial_location=location,
        initial_day=_integer(raw["initial_day"], "game.initial_day", minimum=1),
        initial_truck_count=_integer(raw["initial_truck_count"], "game.initial_truck_count", minimum=1),
        initial_truck_capacity=_integer(raw["initial_truck_capacity"], "game.initial_truck_capacity", minimum=1),
        initial_truck_durability=_decimal(
            raw["initial_truck_durability"], "game.initial_truck_durability", minimum=Decimal("0"), maximum=Decimal("100")
        ),
    )
    return initial, GameLimits(
        challenge_max_days=_integer(raw["challenge_max_days"], "game.challenge_max_days", minimum=1)
    )


def _parse_pricing(raw: Mapping[str, Any]) -> PricingRules:
    _expect_exact_keys(raw, {"high_consumption_multiplier", "remote_sale_premium_per_km"}, "pricing")
    return PricingRules(
        high_consumption_multiplier=_decimal(
            raw["high_consumption_multiplier"], "pricing.high_consumption_multiplier", minimum=Decimal("1")
        ),
        remote_sale_premium_per_km=_decimal(
            raw["remote_sale_premium_per_km"],
            "pricing.remote_sale_premium_per_km",
            minimum=Decimal("0"),
        ),
    )


def _parse_market(raw: Mapping[str, Any]) -> MarketRules:
    _expect_exact_keys(
        raw,
        {
            "trend_range_share",
            "local_spread_persistence",
            "event_spawn_probability",
            "max_active_events",
            "event_ramp_days",
            "shortage_probability",
            "regional_scope_probability",
            "purchase_cooldown_days",
        },
        "market",
    )
    return MarketRules(
        trend_range_share=_decimal(
            raw["trend_range_share"],
            "market.trend_range_share",
            minimum=Decimal("0"),
            maximum=Decimal("1"),
        ),
        local_spread_persistence=_decimal(
            raw["local_spread_persistence"],
            "market.local_spread_persistence",
            minimum=Decimal("0"),
            maximum=Decimal("1"),
        ),
        event_spawn_probability=_decimal(
            raw["event_spawn_probability"],
            "market.event_spawn_probability",
            minimum=Decimal("0"),
            maximum=Decimal("1"),
        ),
        max_active_events=_integer(raw["max_active_events"], "market.max_active_events", minimum=1),
        event_ramp_days=_integer(raw["event_ramp_days"], "market.event_ramp_days", minimum=1),
        shortage_probability=_decimal(
            raw["shortage_probability"],
            "market.shortage_probability",
            minimum=Decimal("0"),
            maximum=Decimal("1"),
        ),
        regional_scope_probability=_decimal(
            raw["regional_scope_probability"],
            "market.regional_scope_probability",
            minimum=Decimal("0"),
            maximum=Decimal("1"),
        ),
        purchase_cooldown_days=_integer(
            raw["purchase_cooldown_days"], "market.purchase_cooldown_days", minimum=0
        ),
    )


def _parse_transport(raw: Mapping[str, Any]) -> TransportRules:
    _expect_exact_keys(
        raw,
        {
            "land",
            "sea",
            "fast_cost_multiplier",
            "fast_time_divisor",
            "truck_min_durability",
            "truck_durability_loss_per_km",
            "truck_damage_time_multiplier",
            "loss",
        },
        "transport",
    )
    loss = _mapping(raw["loss"], "transport.loss")
    _expect_exact_keys(
        loss,
        {
            "reference_km",
            "reference_days",
            "random_factor_min",
            "random_factor_max",
            "normal_distance_divisor",
            "perishable_aging_day_scale",
        },
        "transport.loss",
    )
    random_min = _decimal(loss["random_factor_min"], "transport.loss.random_factor_min", minimum=Decimal("0"))
    random_max = _decimal(loss["random_factor_max"], "transport.loss.random_factor_max", minimum=random_min)
    return TransportRules(
        land=_parse_transport_mode(_mapping(raw["land"], "transport.land"), "transport.land"),
        sea=_parse_transport_mode(_mapping(raw["sea"], "transport.sea"), "transport.sea"),
        fast_cost_multiplier=_decimal(raw["fast_cost_multiplier"], "transport.fast_cost_multiplier", minimum=Decimal("1")),
        fast_time_divisor=_integer(raw["fast_time_divisor"], "transport.fast_time_divisor", minimum=1),
        truck_min_durability=_decimal(raw["truck_min_durability"], "transport.truck_min_durability", minimum=Decimal("0"), maximum=Decimal("100")),
        truck_durability_loss_per_km=_decimal(
            raw["truck_durability_loss_per_km"], "transport.truck_durability_loss_per_km", minimum=Decimal("0")
        ),
        truck_damage_time_multiplier=_decimal(
            raw["truck_damage_time_multiplier"], "transport.truck_damage_time_multiplier", minimum=Decimal("0")
        ),
        loss=TransportLossRules(
            reference_km=_decimal(loss["reference_km"], "transport.loss.reference_km", minimum=Decimal("1")),
            reference_days=_decimal(loss["reference_days"], "transport.loss.reference_days", minimum=Decimal("1")),
            random_factor_min=random_min,
            random_factor_max=random_max,
            normal_distance_divisor=_decimal(
                loss["normal_distance_divisor"], "transport.loss.normal_distance_divisor", minimum=Decimal("1")
            ),
            perishable_aging_day_scale=_decimal(
                loss["perishable_aging_day_scale"], "transport.loss.perishable_aging_day_scale", minimum=Decimal("1")
            ),
        ),
    )


def _parse_transport_mode(raw: Mapping[str, Any], name: str) -> TransportModeRules:
    _expect_exact_keys(
        raw,
        {
            "speed_km_per_day",
            "cost_per_km",
            "travel_day_standard_deviation",
            "travel_day_min_factor",
            "travel_day_max_factor",
        },
        name,
    )
    minimum = _decimal(raw["travel_day_min_factor"], f"{name}.travel_day_min_factor", minimum=Decimal("0"))
    maximum = _decimal(raw["travel_day_max_factor"], f"{name}.travel_day_max_factor", minimum=minimum)
    return TransportModeRules(
        speed_km_per_day=_integer(raw["speed_km_per_day"], f"{name}.speed_km_per_day", minimum=1),
        cost_per_km=_decimal(raw["cost_per_km"], f"{name}.cost_per_km", minimum=Decimal("0")),
        travel_day_standard_deviation=_decimal(
            raw["travel_day_standard_deviation"], f"{name}.travel_day_standard_deviation", minimum=Decimal("0")
        ),
        travel_day_min_factor=minimum,
        travel_day_max_factor=maximum,
    )


def _parse_vehicles(raw: Mapping[str, Any]) -> VehicleRules:
    _expect_exact_keys(
        raw,
        {
            "purchase_price",
            "capacity_per_vehicle",
            "repair_cost_per_percent",
            "repair_days",
            "daily_labor_cost_per_extra_truck",
        },
        "vehicles",
    )
    return VehicleRules(
        purchase_price=_decimal(raw["purchase_price"], "vehicles.purchase_price", minimum=Decimal("0.01")),
        capacity_per_vehicle=_integer(raw["capacity_per_vehicle"], "vehicles.capacity_per_vehicle", minimum=1),
        repair_cost_per_percent=_decimal(raw["repair_cost_per_percent"], "vehicles.repair_cost_per_percent", minimum=Decimal("0")),
        repair_days=_integer(raw["repair_days"], "vehicles.repair_days", minimum=1),
        daily_labor_cost_per_extra_truck=_decimal(
            raw["daily_labor_cost_per_extra_truck"],
            "vehicles.daily_labor_cost_per_extra_truck",
            minimum=Decimal("0"),
        ),
    )


def _parse_finance(raw: Mapping[str, Any]) -> FinanceRules:
    _expect_exact_keys(raw, {"daily_interest_rate", "loan_asset_multiplier"}, "finance")
    return FinanceRules(
        daily_interest_rate=_decimal(raw["daily_interest_rate"], "finance.daily_interest_rate", minimum=Decimal("0")),
        loan_asset_multiplier=_decimal(raw["loan_asset_multiplier"], "finance.loan_asset_multiplier", minimum=Decimal("0")),
    )


def _parse_settlement(raw: Mapping[str, Any]) -> SettlementRules:
    _expect_exact_keys(raw, {"additional_truck_residual_value"}, "settlement")
    return SettlementRules(
        additional_truck_residual_value=_decimal(
            raw["additional_truck_residual_value"],
            "settlement.additional_truck_residual_value",
            minimum=Decimal("0"),
        )
    )


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise GameRulesError(f"{name} 必须是 TOML 表")
    return value


def _expect_exact_keys(raw: Mapping[str, Any], expected: set[str], name: str) -> None:
    actual = set(raw)
    missing = expected - actual
    unknown = actual - expected
    if missing or unknown:
        pieces = []
        if missing:
            pieces.append(f"缺少：{', '.join(sorted(missing))}")
        if unknown:
            pieces.append(f"未知：{', '.join(sorted(unknown))}")
        raise GameRulesError(f"{name} 字段不匹配（{'；'.join(pieces)}）")


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise GameRulesError(f"{name} 必须是非空字符串")
    return value.strip()


def _integer(value: object, name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise GameRulesError(f"{name} 必须是不小于 {minimum} 的整数")
    return value


def _decimal(
    value: object,
    name: str,
    *,
    minimum: Decimal,
    maximum: Decimal | None = None,
) -> Decimal:
    if not isinstance(value, (str, int)) or isinstance(value, bool):
        raise GameRulesError(f"{name} 必须是十进制字符串或整数")
    try:
        parsed = Decimal(str(value))
    except InvalidOperation as error:
        raise GameRulesError(f"{name} 不是有效十进制数") from error
    if not parsed.is_finite() or parsed < minimum or (maximum is not None and parsed > maximum):
        bounds = f"不小于 {minimum}" if maximum is None else f"位于 {minimum} 到 {maximum}"
        raise GameRulesError(f"{name} 必须{bounds}")
    return parsed
