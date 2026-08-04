"""基于结构化状态编码的条件动作策略与全局价值网络。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import sqrt
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.distributions import Categorical

from trade_game.agent import ACTION_TYPES, ActionHead
from trade_game.core import CommandType

from .batching import ActionMaskBatch, ObservationBatch, ObservationSpec
from .encoder import StateEncoder, StateEncoderConfig, StateEncoding


_BUY_ACTION_INDEX = ACTION_TYPES.index(CommandType.BUY)
_SELL_ACTION_INDEX = ACTION_TYPES.index(CommandType.SELL)
_TRAVEL_ACTION_INDEX = ACTION_TYPES.index(CommandType.TRAVEL)
_BORROW_ACTION_INDEX = ACTION_TYPES.index(CommandType.BORROW)
_REPAY_ACTION_INDEX = ACTION_TYPES.index(CommandType.REPAY)
_BUY_TRUCK_ACTION_INDEX = ACTION_TYPES.index(CommandType.BUY_TRUCK)


@dataclass(frozen=True, slots=True)
class ActionBatch:
    """与 ``ActionHead`` 字段顺序一致的批量离散动作。"""

    action_index: Tensor
    product_index: Tensor
    city_index: Tensor
    transport_index: Tensor
    quantity_index: Tensor
    fast_index: Tensor

    @classmethod
    def from_actions(
        cls,
        actions: Sequence[ActionHead],
        *,
        device: torch.device | str | None = None,
    ) -> "ActionBatch":
        """将 Python 动作协议转换为训练和推理使用的整型张量。"""

        items = tuple(actions)
        if not items:
            raise ValueError("动作批次不能为空")
        fields = tuple(zip(*(action.as_tuple() for action in items), strict=True))
        batch = cls(*(torch.tensor(field, dtype=torch.long) for field in fields))
        return batch if device is None else batch.to(device)

    @classmethod
    def from_tensor(cls, values: Tensor) -> "ActionBatch":
        """从轨迹存储的 ``[B, 6]`` 整型动作张量恢复批量动作。"""

        if values.ndim != 2 or values.size(1) != 6:
            raise ValueError("动作张量必须为 [B, 6]")
        return cls(*values.unbind(dim=1))

    @property
    def batch_size(self) -> int:
        return self.action_index.size(0)

    def as_tensor(self) -> Tensor:
        """按动作协议顺序返回 ``[B, 6]`` 整型张量。"""

        return torch.stack(
            (
                self.action_index,
                self.product_index,
                self.city_index,
                self.transport_index,
                self.quantity_index,
                self.fast_index,
            ),
            dim=1,
        )

    def to(self, device: torch.device | str, *, non_blocking: bool = False) -> "ActionBatch":
        """返回迁移到指定设备后的新动作批次。"""

        return replace(
            self,
            action_index=self.action_index.to(device, non_blocking=non_blocking),
            product_index=self.product_index.to(device, non_blocking=non_blocking),
            city_index=self.city_index.to(device, non_blocking=non_blocking),
            transport_index=self.transport_index.to(device, non_blocking=non_blocking),
            quantity_index=self.quantity_index.to(device, non_blocking=non_blocking),
            fast_index=self.fast_index.to(device, non_blocking=non_blocking),
        )


@dataclass(frozen=True, slots=True)
class PolicyLogits:
    """所有条件动作分支的未归一化 logits。"""

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


@dataclass(frozen=True, slots=True)
class PolicySample:
    """一次按掩码采样得到的合法动作及其联合统计量。"""

    action: ActionBatch
    log_prob: Tensor
    entropy: Tensor


@dataclass(frozen=True, slots=True)
class PolicyEvaluation:
    """给定动作在当前策略下的联合统计量。"""

    log_prob: Tensor
    entropy: Tensor


@dataclass(frozen=True, slots=True)
class ActorCriticOutput:
    """训练时一次前向计算产生的策略 logits、价值和状态编码。"""

    encoding: StateEncoding
    policy: PolicyLogits
    value: Tensor


@dataclass(frozen=True, slots=True)
class ActorCriticSample:
    """采样动作与相应价值估计。"""

    action: ActionBatch
    log_prob: Tensor
    entropy: Tensor
    value: Tensor


@dataclass(frozen=True, slots=True)
class ActorCriticEvaluation:
    """给定轨迹动作的策略统计量与价值估计。"""

    log_prob: Tensor
    entropy: Tensor
    value: Tensor


class ActionPolicy(nn.Module):
    """按核心命令参数依赖关系生成六元组动作。"""

    def __init__(self, spec: ObservationSpec, config: StateEncoderConfig) -> None:
        super().__init__()
        self.spec = spec
        self.state_dim = config.state_dim
        self.entity_dim = config.entity_dim
        hidden_dim = config.hidden_dim
        quantity_count = 21

        self.action_head = _FeatureHead(self.state_dim, len(ACTION_TYPES), hidden_dim)
        self.buy_product_query = _QueryHead(self.state_dim, self.entity_dim, hidden_dim)
        self.sell_product_query = _QueryHead(self.state_dim, self.entity_dim, hidden_dim)
        self.travel_city_query = _QueryHead(self.state_dim, self.entity_dim, hidden_dim)
        self.travel_transport_query = _QueryHead(self.state_dim, self.entity_dim, hidden_dim)

        self.buy_quantity_head = _FeatureHead(
            self.state_dim + self.entity_dim,
            quantity_count,
            hidden_dim,
        )
        self.sell_quantity_head = _FeatureHead(
            self.state_dim + self.entity_dim,
            quantity_count,
            hidden_dim,
        )
        self.finance_action_embedding = nn.Embedding(3, self.entity_dim)
        self.finance_quantity_head = _FeatureHead(
            self.state_dim + self.entity_dim,
            quantity_count,
            hidden_dim,
        )
        self.travel_fast_head = _FeatureHead(
            self.state_dim + self.entity_dim,
            2,
            hidden_dim,
        )

    def forward(self, encoding: StateEncoding, batch: ObservationBatch) -> PolicyLogits:
        """计算所有条件分支的 logits，不在此处采样或应用动作掩码。"""

        state = encoding.state
        current_market_tokens = _current_market_tokens(encoding.market_tokens, batch.current_city_ids)
        route_city_tokens = encoding.route_tokens.mean(dim=2)

        buy_product = _score_entities(
            self.buy_product_query(state), current_market_tokens, self.entity_dim
        )
        sell_product = _score_entities(
            self.sell_product_query(state), current_market_tokens, self.entity_dim
        )
        travel_city = _score_entities(
            self.travel_city_query(state), route_city_tokens, self.entity_dim
        )
        travel_transport = _score_routes(
            self.travel_transport_query(state), encoding.route_tokens, self.entity_dim
        )

        buy_quantity = self.buy_quantity_head(
            _join_state_and_entities(state, current_market_tokens)
        )
        sell_quantity = self.sell_quantity_head(
            _join_state_and_entities(state, current_market_tokens)
        )
        finance_quantity = self.finance_quantity_head(
            _join_state_and_entities(
                state,
                self.finance_action_embedding.weight.unsqueeze(0).expand(state.size(0), -1, -1),
            )
        )
        travel_fast = self.travel_fast_head(
            _join_state_and_entities(state, encoding.route_tokens)
        )
        return PolicyLogits(
            action=self.action_head(state),
            buy_product=buy_product,
            sell_product=sell_product,
            buy_quantity=buy_quantity,
            sell_quantity=sell_quantity,
            travel_city=travel_city,
            travel_transport=travel_transport,
            travel_fast=travel_fast,
            borrow_quantity=finance_quantity[:, 0],
            repay_quantity=finance_quantity[:, 1],
            buy_truck_quantity=finance_quantity[:, 2],
        )

    def sample(self, logits: PolicyLogits, masks: ActionMaskBatch) -> PolicySample:
        """从条件分支中依次采样，得到可由动作解码器直接处理的动作。"""

        action_distribution = _masked_categorical(logits.action, masks.action)
        action_index = action_distribution.sample()
        log_prob = action_distribution.log_prob(action_index)
        entropy = action_distribution.entropy()
        batch_size = action_index.size(0)

        buy_selected = action_index == _BUY_ACTION_INDEX
        sell_selected = action_index == _SELL_ACTION_INDEX
        travel_selected = action_index == _TRAVEL_ACTION_INDEX
        borrow_selected = action_index == _BORROW_ACTION_INDEX
        repay_selected = action_index == _REPAY_ACTION_INDEX
        buy_truck_selected = action_index == _BUY_TRUCK_ACTION_INDEX

        buy_product, buy_log_prob, buy_entropy = _sample_selected(
            logits.buy_product, masks.buy_product, buy_selected
        )
        sell_product, sell_log_prob, sell_entropy = _sample_selected(
            logits.sell_product, masks.sell_product, sell_selected
        )
        product_index = torch.where(buy_selected, buy_product, sell_product)
        log_prob = log_prob + buy_log_prob + sell_log_prob
        entropy = entropy + buy_entropy + sell_entropy

        buy_quantity_logits = _select_candidate(logits.buy_quantity, product_index)
        buy_quantity_mask = _select_candidate(masks.buy_quantity, product_index)
        buy_quantity, buy_quantity_log_prob, buy_quantity_entropy = _sample_selected(
            buy_quantity_logits, buy_quantity_mask, buy_selected
        )
        sell_quantity_logits = _select_candidate(logits.sell_quantity, product_index)
        sell_quantity_mask = _select_candidate(masks.sell_quantity, product_index)
        sell_quantity, sell_quantity_log_prob, sell_quantity_entropy = _sample_selected(
            sell_quantity_logits, sell_quantity_mask, sell_selected
        )
        borrow_quantity, borrow_log_prob, borrow_entropy = _sample_selected(
            logits.borrow_quantity, masks.borrow_quantity, borrow_selected
        )
        repay_quantity, repay_log_prob, repay_entropy = _sample_selected(
            logits.repay_quantity, masks.repay_quantity, repay_selected
        )
        truck_quantity, truck_log_prob, truck_entropy = _sample_selected(
            logits.buy_truck_quantity, masks.buy_truck_quantity, buy_truck_selected
        )
        quantity_index = torch.zeros(batch_size, dtype=torch.long, device=action_index.device)
        quantity_index = torch.where(buy_selected, buy_quantity, quantity_index)
        quantity_index = torch.where(sell_selected, sell_quantity, quantity_index)
        quantity_index = torch.where(borrow_selected, borrow_quantity, quantity_index)
        quantity_index = torch.where(repay_selected, repay_quantity, quantity_index)
        quantity_index = torch.where(buy_truck_selected, truck_quantity, quantity_index)
        log_prob = (
            log_prob
            + buy_quantity_log_prob
            + sell_quantity_log_prob
            + borrow_log_prob
            + repay_log_prob
            + truck_log_prob
        )
        entropy = (
            entropy
            + buy_quantity_entropy
            + sell_quantity_entropy
            + borrow_entropy
            + repay_entropy
            + truck_entropy
        )

        city_index, city_log_prob, city_entropy = _sample_selected(
            logits.travel_city, masks.travel_city, travel_selected
        )
        travel_transport_logits = _select_candidate(logits.travel_transport, city_index)
        travel_transport_mask = _select_candidate(masks.travel_transport, city_index)
        transport_index, transport_log_prob, transport_entropy = _sample_selected(
            travel_transport_logits, travel_transport_mask, travel_selected
        )
        travel_fast_logits = _select_candidate(
            _select_candidate(logits.travel_fast, city_index),
            transport_index,
        )
        travel_fast_mask = _select_candidate(
            _select_candidate(masks.travel_fast, city_index),
            transport_index,
        )
        fast_index, fast_log_prob, fast_entropy = _sample_selected(
            travel_fast_logits, travel_fast_mask, travel_selected
        )
        log_prob = log_prob + city_log_prob + transport_log_prob + fast_log_prob
        entropy = entropy + city_entropy + transport_entropy + fast_entropy

        action = ActionBatch(
            action_index=action_index,
            product_index=product_index,
            city_index=city_index,
            transport_index=transport_index,
            quantity_index=quantity_index,
            fast_index=fast_index,
        )
        return PolicySample(action=action, log_prob=log_prob, entropy=entropy)

    def evaluate(
        self,
        logits: PolicyLogits,
        masks: ActionMaskBatch,
        actions: ActionBatch,
    ) -> PolicyEvaluation:
        """计算轨迹中给定动作的联合对数概率和条件熵。"""

        action_distribution = _masked_categorical(logits.action, masks.action)
        log_prob = action_distribution.log_prob(actions.action_index)
        entropy = action_distribution.entropy()

        buy_selected = actions.action_index == _BUY_ACTION_INDEX
        sell_selected = actions.action_index == _SELL_ACTION_INDEX
        travel_selected = actions.action_index == _TRAVEL_ACTION_INDEX
        borrow_selected = actions.action_index == _BORROW_ACTION_INDEX
        repay_selected = actions.action_index == _REPAY_ACTION_INDEX
        buy_truck_selected = actions.action_index == _BUY_TRUCK_ACTION_INDEX

        buy_log_prob, buy_entropy = _evaluate_selected(
            logits.buy_product,
            masks.buy_product,
            actions.product_index,
            buy_selected,
        )
        sell_log_prob, sell_entropy = _evaluate_selected(
            logits.sell_product,
            masks.sell_product,
            actions.product_index,
            sell_selected,
        )
        log_prob = log_prob + buy_log_prob + sell_log_prob
        entropy = entropy + buy_entropy + sell_entropy

        buy_quantity_log_prob, buy_quantity_entropy = _evaluate_selected(
            _select_candidate(logits.buy_quantity, actions.product_index),
            _select_candidate(masks.buy_quantity, actions.product_index),
            actions.quantity_index,
            buy_selected,
        )
        sell_quantity_log_prob, sell_quantity_entropy = _evaluate_selected(
            _select_candidate(logits.sell_quantity, actions.product_index),
            _select_candidate(masks.sell_quantity, actions.product_index),
            actions.quantity_index,
            sell_selected,
        )
        borrow_log_prob, borrow_entropy = _evaluate_selected(
            logits.borrow_quantity,
            masks.borrow_quantity,
            actions.quantity_index,
            borrow_selected,
        )
        repay_log_prob, repay_entropy = _evaluate_selected(
            logits.repay_quantity,
            masks.repay_quantity,
            actions.quantity_index,
            repay_selected,
        )
        truck_log_prob, truck_entropy = _evaluate_selected(
            logits.buy_truck_quantity,
            masks.buy_truck_quantity,
            actions.quantity_index,
            buy_truck_selected,
        )
        log_prob = (
            log_prob
            + buy_quantity_log_prob
            + sell_quantity_log_prob
            + borrow_log_prob
            + repay_log_prob
            + truck_log_prob
        )
        entropy = (
            entropy
            + buy_quantity_entropy
            + sell_quantity_entropy
            + borrow_entropy
            + repay_entropy
            + truck_entropy
        )

        city_log_prob, city_entropy = _evaluate_selected(
            logits.travel_city,
            masks.travel_city,
            actions.city_index,
            travel_selected,
        )
        transport_log_prob, transport_entropy = _evaluate_selected(
            _select_candidate(logits.travel_transport, actions.city_index),
            _select_candidate(masks.travel_transport, actions.city_index),
            actions.transport_index,
            travel_selected,
        )
        fast_log_prob, fast_entropy = _evaluate_selected(
            _select_candidate(
                _select_candidate(logits.travel_fast, actions.city_index),
                actions.transport_index,
            ),
            _select_candidate(
                _select_candidate(masks.travel_fast, actions.city_index),
                actions.transport_index,
            ),
            actions.fast_index,
            travel_selected,
        )
        log_prob = log_prob + city_log_prob + transport_log_prob + fast_log_prob
        entropy = entropy + city_entropy + transport_entropy + fast_entropy
        return PolicyEvaluation(log_prob=log_prob, entropy=entropy)


class ActorCritic(nn.Module):
    """共享状态编码器的条件策略和单一全局状态价值网络。"""

    def __init__(
        self,
        spec: ObservationSpec,
        *,
        encoder_config: StateEncoderConfig | None = None,
    ) -> None:
        super().__init__()
        self.encoder = StateEncoder(spec, encoder_config)
        self.policy = ActionPolicy(spec, self.encoder.config)
        self.value_head = _FeatureHead(
            self.encoder.config.state_dim,
            1,
            self.encoder.config.hidden_dim,
        )

    def forward(self, batch: ObservationBatch) -> ActorCriticOutput:
        """计算策略分支、全局价值和中间状态编码。"""

        encoding = self.encoder(batch)
        return ActorCriticOutput(
            encoding=encoding,
            policy=self.policy(encoding, batch),
            value=self.value_head(encoding.state).squeeze(-1),
        )

    def sample(self, batch: ObservationBatch, masks: ActionMaskBatch) -> ActorCriticSample:
        """从当前策略采样一批合法动作。"""

        output = self(batch)
        sample = self.policy.sample(output.policy, masks)
        return ActorCriticSample(
            action=sample.action,
            log_prob=sample.log_prob,
            entropy=sample.entropy,
            value=output.value,
        )

    def evaluate_actions(
        self,
        batch: ObservationBatch,
        masks: ActionMaskBatch,
        actions: ActionBatch,
    ) -> ActorCriticEvaluation:
        """为 PPO 重放的动作计算新策略统计量和价值。"""

        output = self(batch)
        evaluation = self.policy.evaluate(output.policy, masks, actions)
        return ActorCriticEvaluation(
            log_prob=evaluation.log_prob,
            entropy=evaluation.entropy,
            value=output.value,
        )


class _FeatureHead(nn.Module):
    """策略和价值分支共用的末维 MLP 结构。"""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.layers(features)


class _QueryHead(nn.Module):
    """将全局状态投影为与实体候选相容的查询向量。"""

    def __init__(self, state_dim: int, entity_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.LayerNorm(state_dim),
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, entity_dim),
        )

    def forward(self, state: Tensor) -> Tensor:
        return self.layers(state)


def _current_market_tokens(market_tokens: Tensor, current_city_ids: Tensor) -> Tensor:
    batch_indices = torch.arange(current_city_ids.size(0), device=current_city_ids.device)
    return market_tokens[batch_indices, current_city_ids]


def _score_entities(query: Tensor, entities: Tensor, entity_dim: int) -> Tensor:
    return torch.einsum("bd,bnd->bn", query, entities) / sqrt(entity_dim)


def _score_routes(query: Tensor, routes: Tensor, entity_dim: int) -> Tensor:
    return torch.einsum("bd,bctd->bct", query, routes) / sqrt(entity_dim)


def _join_state_and_entities(state: Tensor, entities: Tensor) -> Tensor:
    state_shape = (state.size(0),) + (1,) * (entities.ndim - 2) + (state.size(-1),)
    expanded_state = state.reshape(state_shape).expand(*entities.shape[:-1], state.size(-1))
    return torch.cat((expanded_state, entities), dim=-1)


def _select_candidate(values: Tensor, indices: Tensor) -> Tensor:
    index_shape = (indices.size(0), 1) + (1,) * (values.ndim - 2)
    expanded_index = indices.reshape(index_shape).expand(
        indices.size(0),
        1,
        *values.shape[2:],
    )
    return torch.gather(values, dim=1, index=expanded_index).squeeze(1)


def _masked_categorical(logits: Tensor, valid: Tensor) -> Categorical:
    if not bool(torch.all(valid.any(dim=-1))):
        raise ValueError("动作掩码不包含可选项")
    masked_logits = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
    return Categorical(logits=masked_logits)


def _sample_selected(
    logits: Tensor,
    valid: Tensor,
    selected: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    samples = torch.zeros(selected.size(0), dtype=torch.long, device=selected.device)
    log_prob = torch.zeros(selected.size(0), dtype=logits.dtype, device=logits.device)
    entropy = torch.zeros(selected.size(0), dtype=logits.dtype, device=logits.device)
    if bool(selected.any()):
        distribution = _masked_categorical(logits[selected], valid[selected])
        selected_samples = distribution.sample()
        samples[selected] = selected_samples
        log_prob[selected] = distribution.log_prob(selected_samples)
        entropy[selected] = distribution.entropy()
    return samples, log_prob, entropy


def _evaluate_selected(
    logits: Tensor,
    valid: Tensor,
    choices: Tensor,
    selected: Tensor,
) -> tuple[Tensor, Tensor]:
    log_prob = torch.zeros(selected.size(0), dtype=logits.dtype, device=logits.device)
    entropy = torch.zeros(selected.size(0), dtype=logits.dtype, device=logits.device)
    if bool(selected.any()):
        distribution = _masked_categorical(logits[selected], valid[selected])
        selected_choices = choices[selected]
        log_prob[selected] = distribution.log_prob(selected_choices)
        entropy[selected] = distribution.entropy()
    return log_prob, entropy


__all__ = [
    "ActionBatch",
    "ActionPolicy",
    "ActorCritic",
    "ActorCriticEvaluation",
    "ActorCriticOutput",
    "ActorCriticSample",
    "PolicyEvaluation",
    "PolicyLogits",
    "PolicySample",
]
