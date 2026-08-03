"""从 CSV 加载和校验静态游戏目录。"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from importlib.resources import files
from pathlib import Path
from types import MappingProxyType
from typing import Iterable, Mapping

from .models import City, Product, ProductCategory, Route, SpecialtyScope, TransportMode


class CatalogDataError(ValueError):
    """静态数据缺失、格式错误或引用不一致时抛出。"""


@dataclass(frozen=True, slots=True)
class Catalog:
    """经校验的城市、商品和路线目录。"""

    cities: Mapping[str, City]
    products: Mapping[str, Product]
    routes: tuple[Route, ...]

    def city(self, name: str) -> City:
        """按名称查询城市，不存在时给出明确错误。"""

        try:
            return self.cities[name]
        except KeyError as error:
            raise KeyError(f"未知城市：{name}") from error

    def product(self, product_id: str) -> Product:
        """按 ID 查询商品，不存在时给出明确错误。"""

        try:
            return self.products[product_id]
        except KeyError as error:
            raise KeyError(f"未知商品：{product_id}") from error


_CITY_COLUMNS = frozenset(
    {
        "name",
        "region",
        "modes",
        "has_bank",
        "has_port",
        "lat",
        "lon",
        "is_high_consumption",
    }
)
_PRODUCT_COLUMNS = frozenset(
    {
        "id",
        "name",
        "category",
        "base_purchase_price",
        "profit_margin_rate",
        "origins",
        "specialty_scope",
        "specialty_region",
        "perishable_shelf_life_days",
        "lambda_min",
        "lambda_max",
        "lambda_alpha",
        "lambda_sigma",
        "transport_loss_rate",
    }
)
_ROUTE_COLUMNS = frozenset({"from_city", "to_city", "mode", "distance_km"})


def load_default_catalog() -> Catalog:
    """加载随 Python 包分发的唯一一套游戏数据。"""

    data_directory = files("trade_game.core.data")
    return load_catalog(Path(str(data_directory)))


def load_catalog(data_directory: Path) -> Catalog:
    """从指定目录加载三个 CSV，并验证所有交叉引用。"""

    cities = _load_cities(data_directory / "cities.csv")
    products = _load_products(data_directory / "products.csv")
    routes = _load_routes(data_directory / "routes.csv")
    _validate_references(cities, products, routes)
    return Catalog(
        cities=MappingProxyType(cities),
        products=MappingProxyType(products),
        routes=tuple(routes),
    )


def _read_rows(path: Path, required_columns: frozenset[str]) -> Iterable[tuple[int, dict[str, str]]]:
    if not path.is_file():
        raise CatalogDataError(f"缺少静态数据文件：{path}")

    try:
        with path.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            columns = frozenset(reader.fieldnames or ())
            missing = required_columns - columns
            if missing:
                names = ", ".join(sorted(missing))
                raise CatalogDataError(f"{path.name} 缺少字段：{names}")
            yield from enumerate(reader, start=2)
    except UnicodeDecodeError as error:
        raise CatalogDataError(f"{path.name} 不是 UTF-8 编码") from error
    except csv.Error as error:
        raise CatalogDataError(f"{path.name} CSV 格式错误：{error}") from error


def _value(row: Mapping[str, str], name: str, path: Path, line: int) -> str:
    value = row[name].strip()
    if not value:
        raise CatalogDataError(f"{path.name}:{line} 的 {name} 不能为空")
    return value


def _decimal(value: str, path: Path, line: int, field: str) -> Decimal:
    try:
        parsed = Decimal(value)
    except InvalidOperation as error:
        raise CatalogDataError(f"{path.name}:{line} 的 {field} 不是有效数字") from error
    if not parsed.is_finite():
        raise CatalogDataError(f"{path.name}:{line} 的 {field} 必须是有限数字")
    return parsed


def _integer(value: str, path: Path, line: int, field: str) -> int:
    try:
        return int(value)
    except ValueError as error:
        raise CatalogDataError(f"{path.name}:{line} 的 {field} 不是有效整数") from error


def _boolean(value: str, path: Path, line: int, field: str) -> bool:
    normalized = value.casefold()
    if normalized in {"1", "true"}:
        return True
    if normalized in {"0", "false"}:
        return False
    raise CatalogDataError(f"{path.name}:{line} 的 {field} 必须为 0、1、true 或 false")


def _load_cities(path: Path) -> dict[str, City]:
    cities: dict[str, City] = {}
    for line, row in _read_rows(path, _CITY_COLUMNS):
        name = _value(row, "name", path, line)
        if name in cities:
            raise CatalogDataError(f"{path.name}:{line} 的城市重复：{name}")
        mode_names = _value(row, "modes", path, line).split("+")
        try:
            modes = frozenset(TransportMode(mode) for mode in mode_names)
        except ValueError as error:
            raise CatalogDataError(f"{path.name}:{line} 包含未知运输方式") from error
        if not modes:
            raise CatalogDataError(f"{path.name}:{line} 的 modes 不能为空")
        has_port = _boolean(_value(row, "has_port", path, line), path, line, "has_port")
        if has_port != (TransportMode.SEA in modes):
            raise CatalogDataError(f"{path.name}:{line} 的 has_port 与 modes 不一致")
        cities[name] = City(
            name=name,
            region=_value(row, "region", path, line),
            modes=modes,
            has_bank=_boolean(_value(row, "has_bank", path, line), path, line, "has_bank"),
            has_port=has_port,
            latitude=_decimal(_value(row, "lat", path, line), path, line, "lat"),
            longitude=_decimal(_value(row, "lon", path, line), path, line, "lon"),
            is_high_consumption=_boolean(
                _value(row, "is_high_consumption", path, line), path, line, "is_high_consumption"
            ),
        )
    if not cities:
        raise CatalogDataError(f"{path.name} 至少需要一个城市")
    return cities


def _load_products(path: Path) -> dict[str, Product]:
    products: dict[str, Product] = {}
    for line, row in _read_rows(path, _PRODUCT_COLUMNS):
        product_id = _value(row, "id", path, line)
        if product_id in products:
            raise CatalogDataError(f"{path.name}:{line} 的商品 ID 重复：{product_id}")
        try:
            category = ProductCategory(_value(row, "category", path, line))
            scope = SpecialtyScope(_value(row, "specialty_scope", path, line))
        except ValueError as error:
            raise CatalogDataError(f"{path.name}:{line} 的商品类别或产地范围无效") from error
        shelf_life_raw = row["perishable_shelf_life_days"].strip()
        shelf_life = (
            _integer(shelf_life_raw, path, line, "perishable_shelf_life_days")
            if shelf_life_raw
            else None
        )
        if shelf_life is not None and shelf_life <= 0:
            raise CatalogDataError(f"{path.name}:{line} 的保质期必须大于 0")
        origins = frozenset(part.strip() for part in _value(row, "origins", path, line).split(";") if part.strip())
        region = row["specialty_region"].strip() or None
        if scope is SpecialtyScope.REGION and region is None:
            raise CatalogDataError(f"{path.name}:{line} 的区域特产必须指定 specialty_region")
        if scope is SpecialtyScope.CITY and region is not None:
            raise CatalogDataError(f"{path.name}:{line} 的城市特产不能指定 specialty_region")
        product = Product(
            id=product_id,
            name=_value(row, "name", path, line),
            category=category,
            base_purchase_price=_decimal(
                _value(row, "base_purchase_price", path, line), path, line, "base_purchase_price"
            ),
            profit_margin_rate=_decimal(
                _value(row, "profit_margin_rate", path, line), path, line, "profit_margin_rate"
            ),
            origins=origins,
            specialty_scope=scope,
            specialty_region=region,
            perishable_shelf_life_days=shelf_life,
            lambda_min=_decimal(_value(row, "lambda_min", path, line), path, line, "lambda_min"),
            lambda_max=_decimal(_value(row, "lambda_max", path, line), path, line, "lambda_max"),
            lambda_alpha=_decimal(_value(row, "lambda_alpha", path, line), path, line, "lambda_alpha"),
            lambda_sigma=_decimal(_value(row, "lambda_sigma", path, line), path, line, "lambda_sigma"),
            transport_loss_rate=_decimal(
                _value(row, "transport_loss_rate", path, line), path, line, "transport_loss_rate"
            ),
        )
        if product.base_purchase_price <= 0:
            raise CatalogDataError(f"{path.name}:{line} 的 base_purchase_price 必须大于 0")
        if product.profit_margin_rate < 0:
            raise CatalogDataError(f"{path.name}:{line} 的 profit_margin_rate 不能为负")
        if product.lambda_min > product.lambda_max:
            raise CatalogDataError(f"{path.name}:{line} 的 lambda_min 不能大于 lambda_max")
        if not Decimal("0") <= product.transport_loss_rate <= Decimal("1"):
            raise CatalogDataError(f"{path.name}:{line} 的 transport_loss_rate 必须在 0 到 1 之间")
        products[product_id] = product
    if not products:
        raise CatalogDataError(f"{path.name} 至少需要一个商品")
    return products


def _load_routes(path: Path) -> list[Route]:
    routes: list[Route] = []
    seen: set[tuple[str, str, TransportMode]] = set()
    for line, row in _read_rows(path, _ROUTE_COLUMNS):
        from_city = _value(row, "from_city", path, line)
        to_city = _value(row, "to_city", path, line)
        if from_city == to_city:
            raise CatalogDataError(f"{path.name}:{line} 的路线起点和终点不能相同")
        try:
            mode = TransportMode(_value(row, "mode", path, line))
        except ValueError as error:
            raise CatalogDataError(f"{path.name}:{line} 包含未知运输方式") from error
        distance = _integer(_value(row, "distance_km", path, line), path, line, "distance_km")
        if distance <= 0:
            raise CatalogDataError(f"{path.name}:{line} 的 distance_km 必须大于 0")
        route_key = (*sorted((from_city, to_city)), mode)
        if route_key in seen:
            raise CatalogDataError(f"{path.name}:{line} 的无向路线重复：{from_city}-{to_city}-{mode}")
        seen.add(route_key)
        routes.append(Route(from_city=from_city, to_city=to_city, mode=mode, distance_km=distance))
    if not routes:
        raise CatalogDataError(f"{path.name} 至少需要一条路线")
    return routes


def _validate_references(
    cities: Mapping[str, City], products: Mapping[str, Product], routes: Iterable[Route]
) -> None:
    for product in products.values():
        missing_origins = product.origins - cities.keys()
        if missing_origins:
            names = ", ".join(sorted(missing_origins))
            raise CatalogDataError(f"商品 {product.id} 引用了不存在的产地：{names}")
        if product.specialty_scope is SpecialtyScope.REGION:
            assert product.specialty_region is not None
            invalid_origins = [
                city for city in product.origins if cities[city].region != product.specialty_region
            ]
            if invalid_origins:
                names = ", ".join(sorted(invalid_origins))
                raise CatalogDataError(f"商品 {product.id} 的产地不属于区域 {product.specialty_region}：{names}")
    for route in routes:
        for city_name in (route.from_city, route.to_city):
            if city_name not in cities:
                raise CatalogDataError(f"路线引用了不存在的城市：{city_name}")
            if route.mode not in cities[city_name].modes:
                raise CatalogDataError(
                    f"路线 {route.from_city}-{route.to_city} 使用了 {city_name} 不支持的 {route.mode}"
                )
