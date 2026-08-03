"""命令执行结果与领域事件的稳定返回格式。"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import StrEnum
from types import MappingProxyType
from typing import Mapping, TypeAlias

from .commands import Command
from .models import GameState


EventValue: TypeAlias = str | int | bool | Decimal


class RejectionCode(StrEnum):
    """规则层拒绝一条格式正确命令时使用的稳定原因代码。"""

    GAME_FINISHED = "game_finished"
    UNKNOWN_ENTITY = "unknown_entity"
    NOT_ALLOWED = "not_allowed"
    INSUFFICIENT_CASH = "insufficient_cash"
    INSUFFICIENT_CAPACITY = "insufficient_capacity"
    INVALID_STATE = "invalid_state"


@dataclass(frozen=True, slots=True)
class GameEvent:
    """一条可供 UI、日志与轨迹消费的领域事件。"""

    name: str
    attributes: Mapping[str, EventValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalized_name = self.name.strip()
        if not normalized_name:
            raise ValueError("事件名称不能为空")
        object.__setattr__(self, "name", normalized_name)
        object.__setattr__(self, "attributes", MappingProxyType(dict(self.attributes)))


@dataclass(frozen=True, slots=True)
class CommandRejection:
    """一条格式正确但不满足当前游戏规则的命令拒绝信息。"""

    code: RejectionCode
    message: str

    def __post_init__(self) -> None:
        normalized_message = self.message.strip()
        if not normalized_message:
            raise ValueError("拒绝原因不能为空")
        object.__setattr__(self, "message", normalized_message)


@dataclass(frozen=True, slots=True)
class CommandResult:
    """规则层处理命令后的完整结果，始终附带当前状态快照。"""

    command: Command
    state: GameState
    events: tuple[GameEvent, ...] = ()
    rejection: CommandRejection | None = None

    def __post_init__(self) -> None:
        if self.rejection is not None and self.events:
            raise ValueError("被拒绝的命令不能同时产生领域事件")

    @property
    def accepted(self) -> bool:
        """命令是否已被规则层接受并产生新的或等价状态。"""

        return self.rejection is None

    @classmethod
    def succeed(
        cls, command: Command, state: GameState, *events: GameEvent
    ) -> "CommandResult":
        """构造一条成功命令的结果。"""

        return cls(command=command, state=state, events=events)

    @classmethod
    def reject(
        cls, command: Command, state: GameState, rejection: CommandRejection
    ) -> "CommandResult":
        """构造一条被规则拒绝的命令结果，状态保持原样。"""

        return cls(command=command, state=state, rejection=rejection)


__all__ = [
    "CommandRejection",
    "CommandResult",
    "EventValue",
    "GameEvent",
    "RejectionCode",
]
