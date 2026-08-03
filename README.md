# TradeGame-RL

一个以交易规则为核心、支持人类游玩与强化学习研究的回合制贸易游戏。

本仓库从干净历史开始重构。实现将分为四个严格单向依赖的层级：

1. `trade_game.core`：唯一的游戏规则与状态变更来源。
2. `trade_game.ui`：人类界面，只将交互转换为核心命令。
3. `trade_game.agent`：智能体动作、观测、合法动作投影与数据协议。
4. `trade_game.learning`：PyTorch 网络、行为克隆、DAgger、PPO 与评估。
