"""从事实矩阵生成策略与价值网络共享的状态表示。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from trade_game.agent import ROUTE_FEATURE_NAMES

from .batching import ObservationBatch, ObservationSpec


_ROUTE_FEATURE_INDEX = {name: index for index, name in enumerate(ROUTE_FEATURE_NAMES)}


@dataclass(frozen=True, slots=True)
class StateEncoderConfig:
    """事实矩阵编码器的维度配置。"""

    row_dim: int = 64
    state_dim: int = 128
    hidden_dim: int = 128
    dropout: float = 0.0


@dataclass(frozen=True, slots=True)
class StateEncoding:
    """全局状态及供条件动作头比较的候选向量。"""

    state: Tensor
    product_tokens: Tensor
    route_city_tokens: Tensor
    route_mode_tokens: Tensor
    route_option_tokens: Tensor


class StateEncoder(nn.Module):
    """编码一次记录的事实，并在网络内部组合经营候选。"""

    def __init__(
        self,
        spec: ObservationSpec,
        config: StateEncoderConfig | None = None,
    ) -> None:
        super().__init__()
        self.spec = spec
        self.config = config or StateEncoderConfig()
        row_dim = self.config.row_dim
        hidden_dim = self.config.hidden_dim
        dropout = self.config.dropout

        self.global_encoder = _FeatureEncoder(
            spec.global_feature_count, row_dim, hidden_dim, dropout
        )
        self.city_encoder = _FeatureEncoder(spec.city_feature_count, row_dim, hidden_dim, dropout)
        self.product_encoder = _FeatureEncoder(
            spec.product_feature_count, row_dim, hidden_dim, dropout
        )
        self.market_encoder = _FeatureEncoder(
            spec.market_feature_count, row_dim, hidden_dim, dropout
        )
        self.route_offer_encoder = _FeatureEncoder(4, row_dim, hidden_dim, dropout)
        self.cargo_lot_encoder = _FeatureEncoder(
            spec.cargo_lot_feature_count, row_dim, hidden_dim, dropout
        )

        self.global_context_fusion = _FeatureEncoder(2 * row_dim, row_dim, hidden_dim, dropout)
        self.route_offer_fusion = _FeatureEncoder(2 * row_dim, row_dim, hidden_dim, dropout)
        self.route_mode_fusion = _FeatureEncoder(2 * row_dim, row_dim, hidden_dim, dropout)
        self.route_city_fusion = _FeatureEncoder(2 * row_dim, row_dim, hidden_dim, dropout)
        self.purchase_candidate_encoder = _FeatureEncoder(4 * row_dim, row_dim, hidden_dim, dropout)
        self.purchase_product_fusion = _FeatureEncoder(2 * row_dim, row_dim, hidden_dim, dropout)
        self.purchase_route_fusion = _FeatureEncoder(2 * row_dim, row_dim, hidden_dim, dropout)
        self.cargo_lot_fusion = _FeatureEncoder(3 * row_dim, row_dim, hidden_dim, dropout)
        self.cargo_route_candidate_encoder = _FeatureEncoder(3 * row_dim, row_dim, hidden_dim, dropout)
        self.cargo_route_fusion = _FeatureEncoder(2 * row_dim, row_dim, hidden_dim, dropout)
        self.cargo_sale_fusion = _FeatureEncoder(3 * row_dim, row_dim, hidden_dim, dropout)
        self.cargo_product_fusion = _FeatureEncoder(2 * row_dim, row_dim, hidden_dim, dropout)
        self.product_fusion = _FeatureEncoder(4 * row_dim, row_dim, hidden_dim, dropout)
        self.route_option_fusion = _FeatureEncoder(3 * row_dim, row_dim, hidden_dim, dropout)
        self.state_fusion = _FeatureEncoder(
            9 * row_dim, self.config.state_dim, hidden_dim, dropout
        )

    def forward(self, batch: ObservationBatch) -> StateEncoding:
        """将一批事实快照编码为动作候选和固定维度状态。"""

        if batch.spec != self.spec:
            raise ValueError("观测批次规格与状态编码器不一致")

        global_token = self.global_encoder(batch.global_features)
        city_tokens = self.city_encoder(batch.city_features)
        product_base_tokens = self.product_encoder(batch.product_features)
        market_tokens = self.market_encoder(batch.market_features)
        current_city_token = torch.einsum("bc,bcr->br", batch.current_city_flags, city_tokens)
        global_context = self.global_context_fusion(
            torch.cat((global_token, current_city_token), dim=-1)
        )
        current_market_tokens = torch.einsum(
            "bc,bcpr->bpr", batch.current_city_flags, market_tokens
        )

        full_route_options = self.route_offer_encoder(_route_offer_features(batch.route_features))
        full_route_valid = batch.route_available.unsqueeze(-1).expand(
            -1, -1, -1, -1, 2
        )
        full_route_modes = self.route_mode_fusion(
            _masked_mean_max_pool(full_route_options, full_route_valid, dims=(4,))
        )
        current_route_options = torch.einsum(
            "bo,bodtfr->bdtfr", batch.current_city_flags, full_route_options
        )
        current_route_valid = torch.einsum(
            "bo,bodt->bdt", batch.current_city_flags, batch.route_available.float()
        ).bool()
        current_route_options = self.route_offer_fusion(
            torch.cat(
                (
                    current_route_options,
                    city_tokens[:, :, None, None, :].expand(
                        -1,
                        -1,
                        self.spec.transport_count,
                        2,
                        -1,
                    ),
                ),
                dim=-1,
            )
        )
        current_route_modes = self.route_mode_fusion(
            _masked_mean_max_pool(
                current_route_options,
                current_route_valid.unsqueeze(-1).expand(-1, -1, -1, 2),
                dims=(3,),
            )
        )

        purchase_candidates = self._purchase_candidates(
            product_base_tokens,
            current_market_tokens,
            market_tokens,
            current_route_modes,
        )
        purchase_valid = current_route_valid.unsqueeze(1).expand(
            -1, self.spec.product_count, -1, -1
        )
        purchase_product_plan = self.purchase_product_fusion(
            _masked_mean_max_pool(purchase_candidates, purchase_valid, dims=(2, 3))
        )
        purchase_route_plan = self.purchase_route_fusion(
            _masked_mean_max_pool(purchase_candidates, purchase_valid, dims=(1,))
        )

        cargo_product_summary, cargo_route_plan, cargo_context = self._cargo_plans(
            batch,
            city_tokens,
            product_base_tokens,
            current_market_tokens,
            market_tokens,
            full_route_modes,
        )
        product_tokens = self.product_fusion(
            torch.cat(
                (
                    product_base_tokens,
                    current_market_tokens,
                    purchase_product_plan,
                    cargo_product_summary,
                ),
                dim=-1,
            )
        )
        route_option_tokens = self.route_option_fusion(
            torch.cat(
                (
                    current_route_options,
                    cargo_route_plan.unsqueeze(3).expand(-1, -1, -1, 2, -1),
                    purchase_route_plan.unsqueeze(3).expand(-1, -1, -1, 2, -1),
                ),
                dim=-1,
            )
        )
        route_mode_tokens = self.route_mode_fusion(
            _masked_mean_max_pool(
                route_option_tokens,
                current_route_valid.unsqueeze(-1).expand(-1, -1, -1, 2),
                dims=(3,),
            )
        )
        route_city_tokens = self.route_city_fusion(
            _masked_mean_max_pool(route_mode_tokens, current_route_valid, dims=(2,))
        )

        market_context = _masked_mean_max_pool(
            market_tokens,
            torch.ones_like(batch.market_features[..., 0], dtype=torch.bool),
            dims=(1, 2),
        )
        product_context = _masked_mean_max_pool(
            product_tokens,
            torch.ones_like(product_tokens[..., 0], dtype=torch.bool),
            dims=(1,),
        )
        route_context = _masked_mean_max_pool(
            route_mode_tokens, current_route_valid, dims=(1, 2)
        )
        state = self.state_fusion(
            torch.cat(
                (global_context, market_context, product_context, route_context, cargo_context), dim=-1
            )
        )
        return StateEncoding(
            state=state,
            product_tokens=product_tokens,
            route_city_tokens=route_city_tokens,
            route_mode_tokens=route_mode_tokens,
            route_option_tokens=route_option_tokens,
        )

    def _purchase_candidates(
        self,
        product_tokens: Tensor,
        current_market_tokens: Tensor,
        market_tokens: Tensor,
        route_mode_tokens: Tensor,
    ) -> Tensor:
        target_market_tokens = market_tokens.permute(0, 2, 1, 3)
        return self.purchase_candidate_encoder(
            torch.cat(
                (
                    product_tokens[:, :, None, None, :].expand(
                        -1, -1, self.spec.city_count, self.spec.transport_count, -1
                    ),
                    current_market_tokens[:, :, None, None, :].expand(
                        -1, -1, self.spec.city_count, self.spec.transport_count, -1
                    ),
                    target_market_tokens[:, :, :, None, :].expand(
                        -1, -1, -1, self.spec.transport_count, -1
                    ),
                    route_mode_tokens[:, None, :, :, :].expand(
                        -1, self.spec.product_count, -1, -1, -1
                    ),
                ),
                dim=-1,
            )
        )

    def _cargo_plans(
        self,
        batch: ObservationBatch,
        city_tokens: Tensor,
        product_base_tokens: Tensor,
        current_market_tokens: Tensor,
        market_tokens: Tensor,
        full_route_modes: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        cargo_tokens = self.cargo_lot_encoder(batch.cargo_lot_features)
        cargo_product_tokens = _gather_axis(product_base_tokens, batch.cargo_product_axes)
        cargo_origin_tokens = _gather_axis(city_tokens, batch.cargo_origin_city_axes)
        cargo_tokens = self.cargo_lot_fusion(
            torch.cat((cargo_tokens, cargo_product_tokens, cargo_origin_tokens), dim=-1)
        )
        current_market_for_cargo = _gather_axis(
            current_market_tokens, batch.cargo_product_axes
        )
        target_market_for_cargo = _gather_market_product_axis(
            market_tokens, batch.cargo_product_axes
        )
        origin_routes_for_cargo = _gather_axis(
            full_route_modes, batch.cargo_origin_city_axes
        )
        origin_route_available = _gather_axis(
            batch.route_available, batch.cargo_origin_city_axes
        )
        cargo_route_candidates = self.cargo_route_candidate_encoder(
            torch.cat(
                (
                    cargo_tokens[:, :, None, None, :].expand(
                        -1, -1, self.spec.city_count, self.spec.transport_count, -1
                    ),
                    target_market_for_cargo[:, :, :, None, :].expand(
                        -1, -1, -1, self.spec.transport_count, -1
                    ),
                    origin_routes_for_cargo,
                ),
                dim=-1,
            )
        )
        cargo_route_valid = batch.cargo_valid[:, :, None, None] & origin_route_available
        cargo_route_plan = self.cargo_route_fusion(
            _masked_mean_max_pool(cargo_route_candidates, cargo_route_valid, dims=(1,))
        )

        origin_to_current_route = torch.einsum(
            "bd,bldtr->bltr", batch.current_city_flags, origin_routes_for_cargo
        )
        origin_to_current_valid = torch.einsum(
            "bd,bldt->blt", batch.current_city_flags, origin_route_available.float()
        ).bool()
        cargo_current_route = self.route_mode_fusion(
            _masked_mean_max_pool(origin_to_current_route, origin_to_current_valid, dims=(2,))
        )
        cargo_sale_tokens = self.cargo_sale_fusion(
            torch.cat((cargo_tokens, current_market_for_cargo, cargo_current_route), dim=-1)
        )
        cargo_product_summary = self.cargo_product_fusion(
            _pool_cargo_by_product(
                cargo_sale_tokens,
                batch.cargo_product_axes,
                batch.cargo_valid,
                self.spec.product_count,
            )
        )
        cargo_context = _masked_mean_max_pool(
            cargo_sale_tokens, batch.cargo_valid, dims=(1,)
        )
        return cargo_product_summary, cargo_route_plan, cargo_context


class _FeatureEncoder(nn.Module):
    """将一行数值事实或内部候选投影为统一向量。"""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.layers(features)


def _route_offer_features(route_features: Tensor) -> Tensor:
    standard = torch.stack(
        (
            route_features[..., _ROUTE_FEATURE_INDEX["standard_fare_log"]],
            route_features[..., _ROUTE_FEATURE_INDEX["standard_travel_time_fraction"]],
            route_features[..., _ROUTE_FEATURE_INDEX["truck_durability_loss_fraction"]],
            route_features[..., _ROUTE_FEATURE_INDEX["economic_distance_fraction"]],
        ),
        dim=-1,
    )
    express = torch.stack(
        (
            route_features[..., _ROUTE_FEATURE_INDEX["express_fare_log"]],
            route_features[..., _ROUTE_FEATURE_INDEX["express_travel_time_fraction"]],
            route_features[..., _ROUTE_FEATURE_INDEX["truck_durability_loss_fraction"]],
            route_features[..., _ROUTE_FEATURE_INDEX["economic_distance_fraction"]],
        ),
        dim=-1,
    )
    return torch.stack((standard, express), dim=-2)


def _masked_mean_max_pool(values: Tensor, valid: Tensor, *, dims: tuple[int, ...]) -> Tensor:
    """沿指定轴汇总有效行的均值与最大值。"""

    valid_values = valid.unsqueeze(-1)
    weights = valid_values.to(dtype=values.dtype)
    count = weights.sum(dim=dims).clamp_min(1.0)
    mean = (values * weights).sum(dim=dims) / count
    masked = values.masked_fill(~valid_values, torch.finfo(values.dtype).min)
    maximum = masked.amax(dim=dims)
    has_valid = valid.any(dim=dims).unsqueeze(-1)
    maximum = torch.where(has_valid, maximum, torch.zeros_like(maximum))
    return torch.cat((mean, maximum), dim=-1)


def _gather_axis(values: Tensor, axes: Tensor) -> Tensor:
    """用结构轴索引取行；索引本身不进入网络特征。"""

    index = axes.reshape(axes.size(0), axes.size(1), *([1] * (values.ndim - 2))).expand(
        axes.size(0), axes.size(1), *values.shape[2:]
    )
    return torch.gather(values, dim=1, index=index)


def _gather_market_product_axis(market_tokens: Tensor, product_axes: Tensor) -> Tensor:
    by_product = market_tokens.permute(0, 2, 1, 3)
    return _gather_axis(by_product, product_axes)


def _pool_cargo_by_product(
    values: Tensor,
    product_axes: Tensor,
    valid: Tensor,
    product_count: int,
) -> Tensor:
    product_slots = torch.arange(product_count, device=values.device).view(1, product_count, 1)
    matching = valid[:, None, :] & (product_axes[:, None, :] == product_slots)
    expanded_values = values[:, None, :, :].expand(-1, product_count, -1, -1)
    return _masked_mean_max_pool(expanded_values, matching, dims=(2,))


__all__ = ["StateEncoder", "StateEncoderConfig", "StateEncoding"]
