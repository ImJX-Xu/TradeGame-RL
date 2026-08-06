"""将结构化观测张量编码为供策略与价值网络使用的状态表示。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .batching import ObservationBatch, ObservationSpec


@dataclass(frozen=True, slots=True)
class StateEncoderConfig:
    """状态编码器的维度配置。"""

    embedding_dim: int = 32
    entity_dim: int = 128
    state_dim: int = 256
    hidden_dim: int = 256
    dropout: float = 0.0


@dataclass(frozen=True, slots=True)
class StateEncoding:
    """状态编码结果及各对象集合相对当前经营状态的注意力。"""

    state: Tensor
    market_tokens: Tensor
    route_tokens: Tensor
    global_context: Tensor
    market_context: Tensor
    route_context: Tensor
    cargo_context: Tensor
    market_attention: Tensor
    route_attention: Tensor
    cargo_attention: Tensor


class StateEncoder(nn.Module):
    """以实体嵌入和目标注意力融合经营、行情、路线及库存信息。"""

    def __init__(
        self,
        spec: ObservationSpec,
        config: StateEncoderConfig | None = None,
    ) -> None:
        super().__init__()
        self.spec = spec
        self.config = config or StateEncoderConfig()
        embedding_dim = self.config.embedding_dim
        entity_dim = self.config.entity_dim
        hidden_dim = self.config.hidden_dim

        self.city_embedding = nn.Embedding(spec.city_count, embedding_dim)
        self.region_embedding = nn.Embedding(len(spec.region_names), embedding_dim)
        self.product_embedding = nn.Embedding(spec.product_count, embedding_dim)
        self.category_embedding = nn.Embedding(len(spec.category_names), embedding_dim)
        self.transport_embedding = nn.Embedding(spec.transport_count, embedding_dim)

        self.city_encoder = _FeatureEncoder(
            2 * embedding_dim + spec.city_feature_count,
            entity_dim,
            hidden_dim,
            self.config.dropout,
        )
        self.product_encoder = _FeatureEncoder(
            2 * embedding_dim + spec.product_feature_count,
            entity_dim,
            hidden_dim,
            self.config.dropout,
        )
        self.global_encoder = _FeatureEncoder(
            spec.global_feature_count + entity_dim,
            entity_dim,
            hidden_dim,
            self.config.dropout,
        )
        self.market_encoder = _FeatureEncoder(
            2 * len(spec.market_history_offsets) + 2,
            entity_dim,
            hidden_dim,
            self.config.dropout,
        )
        self.transport_encoder = _FeatureEncoder(
            embedding_dim,
            entity_dim,
            hidden_dim,
            self.config.dropout,
        )
        self.route_encoder = _FeatureEncoder(
            spec.route_feature_count + 1,
            entity_dim,
            hidden_dim,
            self.config.dropout,
        )
        self.cargo_encoder = _FeatureEncoder(
            spec.cargo_feature_count,
            entity_dim,
            hidden_dim,
            self.config.dropout,
        )

        self.market_attention = _TargetAttentionPool(entity_dim)
        self.route_attention = _TargetAttentionPool(entity_dim)
        self.cargo_attention = _TargetAttentionPool(entity_dim)
        self.fusion = _FeatureEncoder(
            4 * entity_dim,
            self.config.state_dim,
            hidden_dim,
            self.config.dropout,
        )

    def forward(self, batch: ObservationBatch) -> StateEncoding:
        """编码一个符合当前目录规格的批量游戏状态。"""

        if batch.spec != self.spec:
            raise ValueError("观测批次规格与状态编码器不一致")

        city_tokens = self._encode_cities(batch)
        product_tokens = self._encode_products(batch)
        current_city_tokens = _gather_entities(city_tokens, batch.current_city_ids.unsqueeze(1)).squeeze(1)
        global_context = self.global_encoder(
            torch.cat((batch.global_features, current_city_tokens), dim=-1)
        )

        market_tokens = self._encode_market(batch, city_tokens, product_tokens)
        market_context, market_attention = self.market_attention(
            market_tokens.flatten(start_dim=1, end_dim=2),
            global_context,
        )

        route_tokens = self._encode_routes(batch, city_tokens)
        route_context, route_attention = self.route_attention(
            route_tokens.flatten(start_dim=1, end_dim=2),
            global_context,
            batch.route_available.flatten(start_dim=1, end_dim=2),
        )

        cargo_tokens = self._encode_cargo(batch, city_tokens, product_tokens)
        cargo_context, cargo_attention = self.cargo_attention(
            cargo_tokens,
            global_context,
            batch.cargo_valid,
        )

        state = self.fusion(
            torch.cat((global_context, market_context, route_context, cargo_context), dim=-1)
        )
        return StateEncoding(
            state=state,
            market_tokens=market_tokens,
            route_tokens=route_tokens,
            global_context=global_context,
            market_context=market_context,
            route_context=route_context,
            cargo_context=cargo_context,
            market_attention=market_attention.unflatten(
                1, (self.spec.city_count, self.spec.product_count)
            ),
            route_attention=route_attention.unflatten(
                1, (self.spec.city_count, self.spec.transport_count)
            ),
            cargo_attention=cargo_attention,
        )

    def _encode_cities(self, batch: ObservationBatch) -> Tensor:
        city_ids = self.city_embedding(batch.city_ids)
        region_ids = self.region_embedding(batch.city_region_ids)
        return self.city_encoder(torch.cat((city_ids, region_ids, batch.city_features), dim=-1))

    def _encode_products(self, batch: ObservationBatch) -> Tensor:
        product_ids = self.product_embedding(batch.product_ids)
        category_ids = self.category_embedding(batch.product_category_ids)
        return self.product_encoder(
            torch.cat((product_ids, category_ids, batch.product_features), dim=-1)
        )

    def _encode_market(
        self,
        batch: ObservationBatch,
        city_tokens: Tensor,
        product_tokens: Tensor,
    ) -> Tensor:
        history_valid = batch.market_history_valid[:, None, None, :].expand_as(
            batch.market_sale_history
        )
        market_features = torch.cat(
            (
                batch.market_sale_history,
                history_valid.to(dtype=batch.market_sale_history.dtype),
                batch.market_can_purchase.unsqueeze(-1).to(dtype=batch.market_sale_history.dtype),
                batch.market_purchase_cooldown.unsqueeze(-1),
            ),
            dim=-1,
        )
        return (
            city_tokens.unsqueeze(2)
            + product_tokens.unsqueeze(1)
            + self.market_encoder(market_features)
        )

    def _encode_routes(self, batch: ObservationBatch, city_tokens: Tensor) -> Tensor:
        transport_tokens = self.transport_encoder(self.transport_embedding.weight)
        route_features = torch.cat(
            (
                batch.route_features,
                batch.route_available.unsqueeze(-1).to(dtype=batch.route_features.dtype),
            ),
            dim=-1,
        )
        return (
            city_tokens.unsqueeze(2)
            + transport_tokens.view(1, 1, self.spec.transport_count, -1)
            + self.route_encoder(route_features)
        )

    def _encode_cargo(
        self,
        batch: ObservationBatch,
        city_tokens: Tensor,
        product_tokens: Tensor,
    ) -> Tensor:
        cargo_products = _gather_entities(product_tokens, batch.cargo_product_ids)
        cargo_origins = _gather_entities(city_tokens, batch.cargo_origin_city_ids)
        return cargo_products + cargo_origins + self.cargo_encoder(batch.cargo_features)


class _FeatureEncoder(nn.Module):
    """把末维数值和嵌入特征投影到统一表示空间。"""

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


class _TargetAttentionPool(nn.Module):
    """以当前全局经营状态为查询，汇聚一个可变或固定对象集合。"""

    def __init__(self, entity_dim: int) -> None:
        super().__init__()
        self.key = nn.Linear(entity_dim, entity_dim, bias=False)
        self.query = nn.Linear(entity_dim, entity_dim, bias=False)
        self.score = nn.Linear(entity_dim, 1, bias=False)

    def forward(
        self,
        values: Tensor,
        query: Tensor,
        valid: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        scores = self.score(torch.tanh(self.key(values) + self.query(query).unsqueeze(1))).squeeze(-1)
        if valid is None:
            attention = torch.softmax(scores, dim=1)
        else:
            masked_scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
            attention = torch.softmax(masked_scores, dim=1) * valid.to(dtype=scores.dtype)
        context = torch.sum(attention.unsqueeze(-1) * values, dim=1)
        return context, attention


def _gather_entities(tokens: Tensor, indices: Tensor) -> Tensor:
    """按批次中的实体索引提取对应的实体表示。"""

    return torch.gather(
        tokens,
        dim=1,
        index=indices.unsqueeze(-1).expand(-1, -1, tokens.size(-1)),
    )


__all__ = ["StateEncoder", "StateEncoderConfig", "StateEncoding"]
