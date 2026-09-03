"""从事实矩阵生成策略与价值网络共享的状态表示。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .batching import ObservationBatch, ObservationSpec


@dataclass(frozen=True, slots=True)
class StateEncoderConfig:
    """多分支 MLP 编码器和共享层的尺寸。"""

    global_dim: int = 64
    city_dim: int = 64
    product_dim: int = 96
    market_dim: int = 128
    route_dim: int = 128
    cargo_dim: int = 96
    fusion_hidden_dim: int = 256
    state_dim: int = 128
    hidden_dim: int = 128
    dropout: float = 0.0


@dataclass(frozen=True, slots=True)
class StateEncoding:
    """供 Actor 和 Critic 共享的状态表示。"""

    state: Tensor


class StateEncoder(nn.Module):
    """六路展平 MLP、Concat 融合和共享状态编码器。"""

    def __init__(
        self,
        spec: ObservationSpec,
        config: StateEncoderConfig | None = None,
    ) -> None:
        super().__init__()
        self.spec = spec
        self.config = config or StateEncoderConfig()
        c = self.config
        route_input = spec.city_count * spec.city_count * spec.transport_count * (
            spec.route_feature_count + 1
        )
        cargo_input = (
            spec.product_count
            * spec.city_count
            * spec.cargo_lot_slots
            * (spec.cargo_lot_feature_count + 1)
        )
        self.global_encoder = _BranchMLP(spec.global_feature_count, c.global_dim, c)
        self.city_encoder = _BranchMLP(
            spec.city_count * (1 + spec.city_feature_count), c.city_dim, c
        )
        self.product_encoder = _BranchMLP(
            spec.product_count * spec.product_feature_count, c.product_dim, c
        )
        self.market_encoder = _BranchMLP(
            spec.city_count * spec.product_count * spec.market_feature_count,
            c.market_dim,
            c,
        )
        self.route_encoder = _BranchMLP(route_input, c.route_dim, c)
        self.cargo_encoder = _BranchMLP(cargo_input, c.cargo_dim, c)
        concat_dim = c.global_dim + c.city_dim + c.product_dim + c.market_dim + c.route_dim + c.cargo_dim
        self.shared_encoder = nn.Sequential(
            nn.LayerNorm(concat_dim),
            nn.Linear(concat_dim, c.fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(c.dropout),
            nn.Linear(c.fusion_hidden_dim, c.state_dim),
            nn.LayerNorm(c.state_dim),
        )
        _orthogonal_init(self)

    def forward(self, batch: ObservationBatch) -> StateEncoding:
        """展平各事实分支并生成固定维度共享状态。"""

        if batch.spec != self.spec:
            raise ValueError("观测批次规格与状态编码器不一致")
        city_features = torch.cat(
            (batch.current_city_flags.unsqueeze(-1), batch.city_features), dim=-1
        )
        route_features = torch.cat(
            (
                batch.route_features,
                batch.route_available.unsqueeze(-1).to(batch.route_features.dtype),
            ),
            dim=-1,
        )
        cargo_features = torch.cat(
            (
                batch.cargo_lot_features,
                batch.cargo_valid.unsqueeze(-1).to(batch.cargo_lot_features.dtype),
            ),
            dim=-1,
        )
        branches = (
            self.global_encoder(_flatten(batch.global_features)),
            self.city_encoder(_flatten(city_features)),
            self.product_encoder(_flatten(batch.product_features)),
            self.market_encoder(_flatten(batch.market_features)),
            self.route_encoder(_flatten(route_features)),
            self.cargo_encoder(_flatten(cargo_features)),
        )
        return StateEncoding(state=self.shared_encoder(torch.cat(branches, dim=-1)))

class _BranchMLP(nn.Module):
    """一个展平事实分支的两层 MLP。"""

    def __init__(self, input_dim: int, output_dim: int, config: StateEncoderConfig) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.layers(features)


def _flatten(values: Tensor) -> Tensor:
    return values.reshape(values.size(0), -1)


def _orthogonal_init(module: nn.Module) -> None:
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            nn.init.orthogonal_(layer.weight, gain=2**0.5)
            nn.init.zeros_(layer.bias)


__all__ = ["StateEncoder", "StateEncoderConfig", "StateEncoding"]
