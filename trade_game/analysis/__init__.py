"""面向游戏设计与数值维护的离线分析工具。"""

from .necessity import CityNecessity, NecessityReport, ProductNecessity, analyze_necessity
from .greedy import (
    GreedyConfig,
    GreedyEpisode,
    GreedyEvaluation,
    GreedyPolicy,
    GreedyStep,
    evaluate_greedy,
    play_greedy,
)

__all__ = [
    "CityNecessity",
    "GreedyConfig",
    "GreedyEpisode",
    "GreedyEvaluation",
    "GreedyPolicy",
    "GreedyStep",
    "NecessityReport",
    "ProductNecessity",
    "analyze_necessity",
    "evaluate_greedy",
    "play_greedy",
]
