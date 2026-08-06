"""将结构化智能体观测整理为 PyTorch 批量张量。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import torch
from torch import Tensor

from trade_game.agent import (
    CARGO_FEATURE_NAMES,
    CITY_FEATURE_NAMES,
    GLOBAL_FEATURE_NAMES,
    MARKET_HISTORY_OFFSETS,
    PRODUCT_CATEGORY_NAMES,
    PRODUCT_FEATURE_NAMES,
    ROUTE_FEATURE_NAMES,
    ActionMask,
    AgentObservation,
)


class ObservationBatchError(ValueError):
    """观测批次与固定观测协议不一致时抛出。"""


@dataclass(frozen=True, slots=True)
class ObservationSpec:
    """训练数据、网络配置与检查点共享的固定观测规格。"""

    region_names: tuple[str, ...]
    category_names: tuple[str, ...]
    market_history_offsets: tuple[int, ...]
    city_count: int
    product_count: int
    transport_count: int
    global_feature_count: int
    city_feature_count: int
    product_feature_count: int
    route_feature_count: int
    cargo_feature_count: int

    @classmethod
    def from_observation(cls, observation: AgentObservation) -> "ObservationSpec":
        """从一个合法观测中固化当前目录与特征维度。"""

        return cls(
            region_names=observation.region_names,
            category_names=PRODUCT_CATEGORY_NAMES,
            market_history_offsets=MARKET_HISTORY_OFFSETS,
            city_count=len(observation.cities),
            product_count=len(observation.products),
            transport_count=len(observation.routes[0]),
            global_feature_count=len(GLOBAL_FEATURE_NAMES),
            city_feature_count=len(CITY_FEATURE_NAMES),
            product_feature_count=len(PRODUCT_FEATURE_NAMES),
            route_feature_count=len(ROUTE_FEATURE_NAMES),
            cargo_feature_count=len(CARGO_FEATURE_NAMES),
        )

    def validate(self, observation: AgentObservation) -> None:
        """校验一个观测仍遵循创建模型时使用的协议。"""

        if observation.region_names != self.region_names:
            raise ObservationBatchError("观测区域词表与批处理规格不一致")
        if (
            self.category_names != PRODUCT_CATEGORY_NAMES
            or self.market_history_offsets != MARKET_HISTORY_OFFSETS
        ):
            raise ObservationBatchError("批处理规格与当前观测协议不一致")
        if (
            len(observation.cities) != self.city_count
            or len(observation.products) != self.product_count
            or len(observation.market_quotes) != self.city_count
            or len(observation.routes) != self.city_count
        ):
            raise ObservationBatchError("观测实体数量与批处理规格不一致")
        if len(observation.global_state.features) != self.global_feature_count:
            raise ObservationBatchError("全局特征维度与批处理规格不一致")
        if len(observation.market_history_valid) != len(self.market_history_offsets):
            raise ObservationBatchError("市场时间点与批处理规格不一致")
        for city, market_row, route_row in zip(
            observation.cities,
            observation.market_quotes,
            observation.routes,
            strict=True,
        ):
            if len(city.features) != self.city_feature_count:
                raise ObservationBatchError("城市特征维度与批处理规格不一致")
            if len(market_row) != self.product_count or len(route_row) != self.transport_count:
                raise ObservationBatchError("市场或路线实体数量与批处理规格不一致")
            for route in route_row:
                if len(route.features) != self.route_feature_count:
                    raise ObservationBatchError("路线特征维度与批处理规格不一致")
            for quote in market_row:
                if len(quote.sale_log_history) != len(self.market_history_offsets):
                    raise ObservationBatchError("市场价格时间维度与批处理规格不一致")
        for product in observation.products:
            if len(product.features) != self.product_feature_count:
                raise ObservationBatchError("商品特征维度与批处理规格不一致")
        for lot in observation.cargo_lots:
            if len(lot.features) != self.cargo_feature_count:
                raise ObservationBatchError("货物批次特征维度与批处理规格不一致")


@dataclass(frozen=True, slots=True)
class ObservationBatch:
    """可直接传入状态编码网络的批量观测张量。"""

    spec: ObservationSpec
    global_features: Tensor
    current_city_ids: Tensor
    city_ids: Tensor
    city_region_ids: Tensor
    city_features: Tensor
    product_ids: Tensor
    product_category_ids: Tensor
    product_features: Tensor
    market_sale_history: Tensor
    market_can_purchase: Tensor
    market_purchase_cooldown: Tensor
    market_history_valid: Tensor
    route_available: Tensor
    route_features: Tensor
    cargo_product_ids: Tensor
    cargo_origin_city_ids: Tensor
    cargo_features: Tensor
    cargo_valid: Tensor

    @classmethod
    def from_observations(
        cls,
        observations: Sequence[AgentObservation],
        *,
        spec: ObservationSpec | None = None,
        device: torch.device | str | None = None,
    ) -> "ObservationBatch":
        """将同一观测协议下的状态快照堆叠为一个 batch。"""

        items = tuple(observations)
        if not items:
            raise ObservationBatchError("观测批次不能为空")
        resolved_spec = spec or ObservationSpec.from_observation(items[0])
        for observation in items:
            resolved_spec.validate(observation)

        batch_size = len(items)
        cargo_count = max(1, max(len(observation.cargo_lots) for observation in items))
        batch = cls(
            spec=resolved_spec,
            global_features=torch.tensor(
                [observation.global_state.features for observation in items], dtype=torch.float32
            ),
            current_city_ids=torch.tensor(
                [observation.global_state.current_city_index for observation in items], dtype=torch.long
            ),
            city_ids=torch.tensor(
                [[city.city_index for city in observation.cities] for observation in items], dtype=torch.long
            ),
            city_region_ids=torch.tensor(
                [[city.region_index for city in observation.cities] for observation in items], dtype=torch.long
            ),
            city_features=torch.tensor(
                [[city.features for city in observation.cities] for observation in items], dtype=torch.float32
            ),
            product_ids=torch.tensor(
                [[product.product_index for product in observation.products] for observation in items],
                dtype=torch.long,
            ),
            product_category_ids=torch.tensor(
                [[product.category_index for product in observation.products] for observation in items],
                dtype=torch.long,
            ),
            product_features=torch.tensor(
                [[product.features for product in observation.products] for observation in items],
                dtype=torch.float32,
            ),
            market_sale_history=torch.tensor(
                [
                    [[quote.sale_log_history for quote in market_row] for market_row in observation.market_quotes]
                    for observation in items
                ],
                dtype=torch.float32,
            ),
            market_can_purchase=torch.tensor(
                [
                    [[quote.can_purchase for quote in market_row] for market_row in observation.market_quotes]
                    for observation in items
                ],
                dtype=torch.bool,
            ),
            market_purchase_cooldown=torch.tensor(
                [
                    [
                        [quote.purchase_cooldown_fraction for quote in market_row]
                        for market_row in observation.market_quotes
                    ]
                    for observation in items
                ],
                dtype=torch.float32,
            ),
            market_history_valid=torch.tensor(
                [observation.market_history_valid for observation in items], dtype=torch.bool
            ),
            route_available=torch.tensor(
                [
                    [[route.available for route in route_row] for route_row in observation.routes]
                    for observation in items
                ],
                dtype=torch.bool,
            ),
            route_features=torch.tensor(
                [
                    [[route.features for route in route_row] for route_row in observation.routes]
                    for observation in items
                ],
                dtype=torch.float32,
            ),
            cargo_product_ids=torch.zeros((batch_size, cargo_count), dtype=torch.long),
            cargo_origin_city_ids=torch.zeros((batch_size, cargo_count), dtype=torch.long),
            cargo_features=torch.zeros(
                (batch_size, cargo_count, resolved_spec.cargo_feature_count), dtype=torch.float32
            ),
            cargo_valid=torch.zeros((batch_size, cargo_count), dtype=torch.bool),
        )
        for batch_index, observation in enumerate(items):
            for cargo_index, lot in enumerate(observation.cargo_lots):
                batch.cargo_product_ids[batch_index, cargo_index] = lot.product_index
                batch.cargo_origin_city_ids[batch_index, cargo_index] = lot.origin_city_index
                batch.cargo_features[batch_index, cargo_index] = torch.tensor(
                    lot.features, dtype=torch.float32
                )
                batch.cargo_valid[batch_index, cargo_index] = True
        return batch if device is None else batch.to(device)

    def to(self, device: torch.device | str, *, non_blocking: bool = False) -> "ObservationBatch":
        """返回迁移到指定设备后的新批次，规格保持在 Python 侧。"""

        return replace(
            self,
            global_features=self.global_features.to(device, non_blocking=non_blocking),
            current_city_ids=self.current_city_ids.to(device, non_blocking=non_blocking),
            city_ids=self.city_ids.to(device, non_blocking=non_blocking),
            city_region_ids=self.city_region_ids.to(device, non_blocking=non_blocking),
            city_features=self.city_features.to(device, non_blocking=non_blocking),
            product_ids=self.product_ids.to(device, non_blocking=non_blocking),
            product_category_ids=self.product_category_ids.to(device, non_blocking=non_blocking),
            product_features=self.product_features.to(device, non_blocking=non_blocking),
            market_sale_history=self.market_sale_history.to(device, non_blocking=non_blocking),
            market_can_purchase=self.market_can_purchase.to(device, non_blocking=non_blocking),
            market_purchase_cooldown=self.market_purchase_cooldown.to(device, non_blocking=non_blocking),
            market_history_valid=self.market_history_valid.to(device, non_blocking=non_blocking),
            route_available=self.route_available.to(device, non_blocking=non_blocking),
            route_features=self.route_features.to(device, non_blocking=non_blocking),
            cargo_product_ids=self.cargo_product_ids.to(device, non_blocking=non_blocking),
            cargo_origin_city_ids=self.cargo_origin_city_ids.to(device, non_blocking=non_blocking),
            cargo_features=self.cargo_features.to(device, non_blocking=non_blocking),
            cargo_valid=self.cargo_valid.to(device, non_blocking=non_blocking),
        )

    def index_select(self, indices: Tensor) -> "ObservationBatch":
        """浠庡凡鎵归噺鍖栫殑瑙傛祴涓鍙栦竴涓皬鎵规銆?"""

        return replace(
            self,
            global_features=self.global_features.index_select(0, indices),
            current_city_ids=self.current_city_ids.index_select(0, indices),
            city_ids=self.city_ids.index_select(0, indices),
            city_region_ids=self.city_region_ids.index_select(0, indices),
            city_features=self.city_features.index_select(0, indices),
            product_ids=self.product_ids.index_select(0, indices),
            product_category_ids=self.product_category_ids.index_select(0, indices),
            product_features=self.product_features.index_select(0, indices),
            market_sale_history=self.market_sale_history.index_select(0, indices),
            market_can_purchase=self.market_can_purchase.index_select(0, indices),
            market_purchase_cooldown=self.market_purchase_cooldown.index_select(0, indices),
            market_history_valid=self.market_history_valid.index_select(0, indices),
            route_available=self.route_available.index_select(0, indices),
            route_features=self.route_features.index_select(0, indices),
            cargo_product_ids=self.cargo_product_ids.index_select(0, indices),
            cargo_origin_city_ids=self.cargo_origin_city_ids.index_select(0, indices),
            cargo_features=self.cargo_features.index_select(0, indices),
            cargo_valid=self.cargo_valid.index_select(0, indices),
        )


@dataclass(frozen=True, slots=True)
class ActionMaskBatch:
    """保持动作掩码层级的批量布尔张量，不混入状态特征。"""

    action: Tensor
    buy_product: Tensor
    sell_product: Tensor
    buy_quantity: Tensor
    sell_quantity: Tensor
    travel_city: Tensor
    travel_transport: Tensor
    travel_fast: Tensor
    borrow_quantity: Tensor
    repay_quantity: Tensor
    buy_truck_quantity: Tensor

    @classmethod
    def from_masks(
        cls,
        masks: Sequence[ActionMask],
        *,
        device: torch.device | str | None = None,
    ) -> "ActionMaskBatch":
        """将同一动作词表下的条件掩码堆叠为批量布尔张量。"""

        items = tuple(masks)
        if not items:
            raise ObservationBatchError("动作掩码批次不能为空")
        batch = cls(
            action=torch.tensor([mask.action for mask in items], dtype=torch.bool),
            buy_product=torch.tensor([mask.buy_product for mask in items], dtype=torch.bool),
            sell_product=torch.tensor([mask.sell_product for mask in items], dtype=torch.bool),
            buy_quantity=torch.tensor([mask.buy_quantity for mask in items], dtype=torch.bool),
            sell_quantity=torch.tensor([mask.sell_quantity for mask in items], dtype=torch.bool),
            travel_city=torch.tensor([mask.travel_city for mask in items], dtype=torch.bool),
            travel_transport=torch.tensor([mask.travel_transport for mask in items], dtype=torch.bool),
            travel_fast=torch.tensor([mask.travel_fast for mask in items], dtype=torch.bool),
            borrow_quantity=torch.tensor([mask.borrow_quantity for mask in items], dtype=torch.bool),
            repay_quantity=torch.tensor([mask.repay_quantity for mask in items], dtype=torch.bool),
            buy_truck_quantity=torch.tensor([mask.buy_truck_quantity for mask in items], dtype=torch.bool),
        )
        return batch if device is None else batch.to(device)

    def to(self, device: torch.device | str, *, non_blocking: bool = False) -> "ActionMaskBatch":
        """返回迁移到指定设备后的新动作掩码批次。"""

        return replace(
            self,
            action=self.action.to(device, non_blocking=non_blocking),
            buy_product=self.buy_product.to(device, non_blocking=non_blocking),
            sell_product=self.sell_product.to(device, non_blocking=non_blocking),
            buy_quantity=self.buy_quantity.to(device, non_blocking=non_blocking),
            sell_quantity=self.sell_quantity.to(device, non_blocking=non_blocking),
            travel_city=self.travel_city.to(device, non_blocking=non_blocking),
            travel_transport=self.travel_transport.to(device, non_blocking=non_blocking),
            travel_fast=self.travel_fast.to(device, non_blocking=non_blocking),
            borrow_quantity=self.borrow_quantity.to(device, non_blocking=non_blocking),
            repay_quantity=self.repay_quantity.to(device, non_blocking=non_blocking),
            buy_truck_quantity=self.buy_truck_quantity.to(device, non_blocking=non_blocking),
        )

    def index_select(self, indices: Tensor) -> "ActionMaskBatch":
        """浠庡凡鎵归噺鍖栫殑鎺╃爜涓鍙栦竴涓皬鎵规銆?"""

        return replace(
            self,
            action=self.action.index_select(0, indices),
            buy_product=self.buy_product.index_select(0, indices),
            sell_product=self.sell_product.index_select(0, indices),
            buy_quantity=self.buy_quantity.index_select(0, indices),
            sell_quantity=self.sell_quantity.index_select(0, indices),
            travel_city=self.travel_city.index_select(0, indices),
            travel_transport=self.travel_transport.index_select(0, indices),
            travel_fast=self.travel_fast.index_select(0, indices),
            borrow_quantity=self.borrow_quantity.index_select(0, indices),
            repay_quantity=self.repay_quantity.index_select(0, indices),
            buy_truck_quantity=self.buy_truck_quantity.index_select(0, indices),
        )


__all__ = [
    "ActionMaskBatch",
    "ObservationBatch",
    "ObservationBatchError",
    "ObservationSpec",
]
