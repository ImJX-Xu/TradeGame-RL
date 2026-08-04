# TradeGame-RL

一个以交易规则为核心、支持人类游玩与强化学习研究的回合制贸易游戏。

代码按严格单向依赖组织为四层：

1. `trade_game.core`：唯一的游戏规则与状态变更来源。
2. `trade_game.ui`：人类界面，只将交互转换为核心命令。
3. `trade_game.agent`：智能体动作、观测、合法动作投影与数据协议。
4. `trade_game.learning`：PyTorch 网络、行为克隆、DAgger、PPO 与评估。

## 游玩

图形界面是可选依赖，首次使用前安装：

```powershell
pip install -e ".[ui]"
```

在仓库目录执行：

```powershell
python -m trade_game play
```

可用 `--seed` 固定随机过程，或用 `--mode challenge` 开始 150 天挑战模式：

```powershell
python -m trade_game play --seed 7 --mode challenge
```

图形界面以货运调度台为主屏：在地图或“路线”页选择直达站点并发运；“交易”页可选择商品、调整数量后采购或出售；“车辆”和“融资”页分别处理维修、购车、借款与还款；底部“日结”推进一天。所有图形操作都通过游戏核心命令执行。

终端界面可使用 `--terminal` 启动：

```powershell
python -m trade_game play --terminal
```

终端内使用 `help` 查看命令。常用操作包括：

```text
market
buy zz_flour 10
travel 石家庄 land
borrow 500
next
```
