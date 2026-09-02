# 智能体结构

智能体由事实矩阵观测、条件离散策略、全局价值估计，以及原生 PyTorch 的 PPO、行为克隆和 DAgger 组成。游戏核心始终通过 `AgentEnvironment` 暴露同一套动作协议与规则约束。

1. [状态编码器](state-encoder.md)：将 `G/X/C/P/M/R/L` 事实矩阵编码为全局状态和商品、路线候选。
2. [Actor-Critic 条件动作网络](actor-critic.md)：从候选行构造操作、商品、数量和旅行参数的联合动作分布。
3. [PPO 训练流程](ppo-training.md)：描述 rollout、GAE、裁剪目标、参数调度和训练指标。

训练配置中的 `[observation]` 段定义市场历史采样点。例如：

```toml
[observation]
market_history_offsets = [6, 4, 2, 0]
```

这些偏移量由配置传入观测器；价格历史保留长度则由游戏规则 `market.price_history_days` 管理。
