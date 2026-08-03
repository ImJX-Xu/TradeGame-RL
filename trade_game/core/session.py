"""游戏会话：唯一拥有状态并分发玩家或智能体命令。"""

from __future__ import annotations

from dataclasses import replace
from random import Random

from .catalog import Catalog, load_default_catalog
from .commands import Borrow, Buy, BuyTruck, Command, NextDay, RepairTruck, Repay, Sell, Travel
from .finance import borrow, repay
from .models import GameMode, GameState
from .results import CommandRejection, CommandResult, GameEvent, RejectionCode
from .rules import GameRules, load_default_game_rules
from .settlement import conclude_if_needed
from .setup import create_initial_state
from .timeflow import settle_elapsed_days
from .trading import buy, sell
from .transport import travel
from .vehicles import buy_truck, repair_truck


class GameSession:
    """封装一局游戏的状态、随机源和统一命令执行入口。"""

    def __init__(self, catalog: Catalog, rules: GameRules, state: GameState, rng: Random) -> None:
        self.catalog = catalog
        self.rules = rules
        self.state = state
        self._rng = rng

    def dispatch(self, command: Command) -> CommandResult:
        """执行一条命令；成功命令的耗时统一触发逐日结算和终局判定。"""

        if self.state.outcome is not None:
            return CommandResult.reject(
                command,
                self.state,
                CommandRejection(RejectionCode.GAME_FINISHED, "本局游戏已经结束"),
            )

        result = self._dispatch_command(command)
        if not result.accepted:
            return result

        elapsed_days = result.state.day - self.state.day
        next_state = result.state
        events = result.events
        if elapsed_days:
            next_state, settlement_events = settle_elapsed_days(
                self.catalog, self.rules, next_state, self._rng, elapsed_days
            )
            events = (*events, *settlement_events)
        next_state = conclude_if_needed(self.catalog, self.rules, next_state)
        if next_state.outcome is not None:
            events = (
                *events,
                GameEvent(
                    "game_finished",
                    {
                        "reason": next_state.outcome.reason.value,
                        "final_assets": next_state.outcome.final_assets,
                    },
                ),
            )
        self.state = next_state
        return CommandResult.succeed(command, next_state, *events)

    def _dispatch_command(self, command: Command) -> CommandResult:
        if isinstance(command, NextDay):
            next_state = replace(self.state, day=self.state.day + 1)
            return CommandResult.succeed(command, next_state, GameEvent("day_advanced", {"day": next_state.day}))
        if isinstance(command, Buy):
            return buy(self.catalog, self.rules, self.state, command)
        if isinstance(command, Sell):
            return sell(self.catalog, self.rules, self.state, command)
        if isinstance(command, Travel):
            return travel(self.catalog, self.rules, self.state, command, self._rng)
        if isinstance(command, RepairTruck):
            return repair_truck(self.rules, self.state, command)
        if isinstance(command, BuyTruck):
            return buy_truck(self.rules, self.state, command)
        if isinstance(command, Borrow):
            return borrow(self.catalog, self.rules, self.state, command)
        if isinstance(command, Repay):
            return repay(self.catalog, self.rules, self.state, command)
        raise TypeError(f"不支持的命令类型：{type(command).__name__}")


def create_game_session(*, seed: int | None = None, mode: GameMode = GameMode.FREE) -> GameSession:
    """使用包内默认目录和规则创建一局独立游戏。"""

    catalog = load_default_catalog()
    rules = load_default_game_rules(catalog)
    return GameSession(catalog, rules, create_initial_state(catalog, rules, mode=mode), Random(seed))
