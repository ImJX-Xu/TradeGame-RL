"""将智能体事实矩阵堆叠为 PyTorch 训练批次。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import torch
from torch import Tensor

from trade_game.agent import (
    CARGO_LOT_FEATURE_NAMES,
    CITY_FEATURE_NAMES,
    GLOBAL_FEATURE_NAMES,
    PRODUCT_FEATURE_NAMES,
    ROUTE_FEATURE_NAMES,
    ActionMask,
    AgentObservation,
    market_feature_names,
)


class ObservationBatchError(ValueError):
    """观测与已创建模型的矩阵协议不一致。"""


@dataclass(frozen=True, slots=True)
class ObservationSpec:
    """模型、rollout 和检查点共享的事实矩阵规格。"""

    market_history_offsets: tuple[int, ...]
    city_count: int
    product_count: int
    transport_count: int
    global_feature_count: int
    city_feature_count: int
    product_feature_count: int
    market_feature_count: int
    route_feature_count: int
    cargo_lot_feature_count: int

    @classmethod
    def from_observation(cls, observation: AgentObservation) -> "ObservationSpec":
        return cls(
            market_history_offsets=observation.market_history_offsets,
            city_count=len(observation.city_features),
            product_count=len(observation.product_features),
            transport_count=len(observation.route_features[0][0]),
            global_feature_count=len(GLOBAL_FEATURE_NAMES),
            city_feature_count=len(CITY_FEATURE_NAMES),
            product_feature_count=len(PRODUCT_FEATURE_NAMES),
            market_feature_count=len(market_feature_names(observation.market_history_offsets)),
            route_feature_count=len(ROUTE_FEATURE_NAMES),
            cargo_lot_feature_count=len(CARGO_LOT_FEATURE_NAMES),
        )

    def validate(self, observation: AgentObservation) -> None:
        if observation.market_history_offsets != self.market_history_offsets:
            raise ObservationBatchError("市场历史采样配置与观测规格不一致")
        if (
            len(observation.global_state.features) != self.global_feature_count
            or len(observation.current_city_flags) != self.city_count
            or len(observation.city_features) != self.city_count
            or len(observation.product_features) != self.product_count
            or len(observation.market_features) != self.city_count
            or len(observation.route_features) != self.city_count
            or len(observation.route_available) != self.city_count
            or len(observation.cargo_lot_table) != self.product_count
        ):
            raise ObservationBatchError("观测矩阵尺寸与模型规格不一致")
        for city_features, market_row, route_origins, route_available in zip(
            observation.city_features,
            observation.market_features,
            observation.route_features,
            observation.route_available,
            strict=True,
        ):
            if len(city_features) != self.city_feature_count:
                raise ObservationBatchError("城市特征维度与模型规格不一致")
            if len(market_row) != self.product_count:
                raise ObservationBatchError("市场商品轴与模型规格不一致")
            if len(route_origins) != self.city_count or len(route_available) != self.city_count:
                raise ObservationBatchError("路线城市轴与模型规格不一致")
            for market_cell in market_row:
                if len(market_cell) != self.market_feature_count:
                    raise ObservationBatchError("市场特征维度与模型规格不一致")
            for route_modes, available_modes in zip(route_origins, route_available, strict=True):
                if len(route_modes) != self.transport_count or len(available_modes) != self.transport_count:
                    raise ObservationBatchError("路线运输轴与模型规格不一致")
                for route_features in route_modes:
                    if len(route_features) != self.route_feature_count:
                        raise ObservationBatchError("路线特征维度与模型规格不一致")
        for product_features, origin_rows in zip(
            observation.product_features, observation.cargo_lot_table, strict=True
        ):
            if len(product_features) != self.product_feature_count:
                raise ObservationBatchError("商品特征维度与模型规格不一致")
            if len(origin_rows) != self.city_count:
                raise ObservationBatchError("库存产地轴与模型规格不一致")
            for lots in origin_rows:
                for lot in lots:
                    if len(lot) != self.cargo_lot_feature_count:
                        raise ObservationBatchError("库存批次特征维度与模型规格不一致")


@dataclass(frozen=True, slots=True)
class ObservationBatch:
    """网络输入张量。

    ``cargo_product_axes`` 与 ``cargo_origin_city_axes`` 仅用于从商品、城市和
    路线矩阵取得对应行，不会传入 MLP 或作为可学习的 ID 特征。
    """

    spec: ObservationSpec
    global_features: Tensor
    current_city_flags: Tensor
    city_features: Tensor
    product_features: Tensor
    market_features: Tensor
    route_available: Tensor
    route_features: Tensor
    cargo_lot_features: Tensor
    cargo_product_axes: Tensor
    cargo_origin_city_axes: Tensor
    cargo_valid: Tensor

    @classmethod
    def from_observations(
        cls,
        observations: Sequence[AgentObservation],
        *,
        spec: ObservationSpec | None = None,
        device: torch.device | str | None = None,
    ) -> "ObservationBatch":
        items = tuple(observations)
        if not items:
            raise ObservationBatchError("观测批次不能为空")
        resolved_spec = spec or ObservationSpec.from_observation(items[0])
        for observation in items:
            resolved_spec.validate(observation)

        cargo_entries = tuple(_cargo_entries(observation) for observation in items)
        batch_size = len(items)
        cargo_count = max(1, max(len(entries) for entries in cargo_entries))
        batch = cls(
            spec=resolved_spec,
            global_features=torch.tensor(
                [item.global_state.features for item in items], dtype=torch.float32
            ),
            current_city_flags=torch.tensor(
                [item.current_city_flags for item in items], dtype=torch.float32
            ),
            city_features=torch.tensor([item.city_features for item in items], dtype=torch.float32),
            product_features=torch.tensor(
                [item.product_features for item in items], dtype=torch.float32
            ),
            market_features=torch.tensor(
                [item.market_features for item in items], dtype=torch.float32
            ),
            route_available=torch.tensor(
                [item.route_available for item in items], dtype=torch.bool
            ),
            route_features=torch.tensor(
                [item.route_features for item in items], dtype=torch.float32
            ),
            cargo_lot_features=torch.zeros(
                (batch_size, cargo_count, resolved_spec.cargo_lot_feature_count), dtype=torch.float32
            ),
            cargo_product_axes=torch.zeros((batch_size, cargo_count), dtype=torch.long),
            cargo_origin_city_axes=torch.zeros((batch_size, cargo_count), dtype=torch.long),
            cargo_valid=torch.zeros((batch_size, cargo_count), dtype=torch.bool),
        )
        for batch_index, entries in enumerate(cargo_entries):
            for cargo_index, (product_axis, origin_axis, features) in enumerate(entries):
                batch.cargo_lot_features[batch_index, cargo_index] = torch.tensor(
                    features, dtype=torch.float32
                )
                batch.cargo_product_axes[batch_index, cargo_index] = product_axis
                batch.cargo_origin_city_axes[batch_index, cargo_index] = origin_axis
                batch.cargo_valid[batch_index, cargo_index] = True
        return batch if device is None else batch.to(device)

    def to(self, device: torch.device | str, *, non_blocking: bool = False) -> "ObservationBatch":
        return replace(
            self,
            global_features=self.global_features.to(device, non_blocking=non_blocking),
            current_city_flags=self.current_city_flags.to(device, non_blocking=non_blocking),
            city_features=self.city_features.to(device, non_blocking=non_blocking),
            product_features=self.product_features.to(device, non_blocking=non_blocking),
            market_features=self.market_features.to(device, non_blocking=non_blocking),
            route_available=self.route_available.to(device, non_blocking=non_blocking),
            route_features=self.route_features.to(device, non_blocking=non_blocking),
            cargo_lot_features=self.cargo_lot_features.to(device, non_blocking=non_blocking),
            cargo_product_axes=self.cargo_product_axes.to(device, non_blocking=non_blocking),
            cargo_origin_city_axes=self.cargo_origin_city_axes.to(
                device, non_blocking=non_blocking
            ),
            cargo_valid=self.cargo_valid.to(device, non_blocking=non_blocking),
        )

    def index_select(self, indices: Tensor) -> "ObservationBatch":
        return replace(
            self,
            global_features=self.global_features.index_select(0, indices),
            current_city_flags=self.current_city_flags.index_select(0, indices),
            city_features=self.city_features.index_select(0, indices),
            product_features=self.product_features.index_select(0, indices),
            market_features=self.market_features.index_select(0, indices),
            route_available=self.route_available.index_select(0, indices),
            route_features=self.route_features.index_select(0, indices),
            cargo_lot_features=self.cargo_lot_features.index_select(0, indices),
            cargo_product_axes=self.cargo_product_axes.index_select(0, indices),
            cargo_origin_city_axes=self.cargo_origin_city_axes.index_select(0, indices),
            cargo_valid=self.cargo_valid.index_select(0, indices),
        )


def _cargo_entries(
    observation: AgentObservation,
) -> tuple[tuple[int, int, tuple[float, ...]], ...]:
    entries: list[tuple[int, int, tuple[float, ...]]] = []
    for product_axis, origin_rows in enumerate(observation.cargo_lot_table):
        for origin_axis, lots in enumerate(origin_rows):
            entries.extend((product_axis, origin_axis, features) for features in lots)
    return tuple(entries)


@dataclass(frozen=True, slots=True)
class ActionMaskBatch:
    """动作掩码保持为核心规则的独立输出。"""

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
            travel_transport=torch.tensor(
                [mask.travel_transport for mask in items], dtype=torch.bool
            ),
            travel_fast=torch.tensor([mask.travel_fast for mask in items], dtype=torch.bool),
            borrow_quantity=torch.tensor(
                [mask.borrow_quantity for mask in items], dtype=torch.bool
            ),
            repay_quantity=torch.tensor([mask.repay_quantity for mask in items], dtype=torch.bool),
            buy_truck_quantity=torch.tensor(
                [mask.buy_truck_quantity for mask in items], dtype=torch.bool
            ),
        )
        return batch if device is None else batch.to(device)

    def to(self, device: torch.device | str, *, non_blocking: bool = False) -> "ActionMaskBatch":
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
