# TradeGame-RL

一个以交易规则为核心、支持人类游玩与强化学习研究的回合制贸易游戏。

代码按严格单向依赖组织为四层：

1. `trade_game.core`：唯一的游戏规则与状态变更来源。
2. `trade_game.ui`：人类界面，只将交互转换为核心命令。
3. `trade_game.agent`：智能体动作协议与核心命令解码。
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

图形界面以货运调度台为主屏：在地图或“路线”页选择当前运输网络中的可达城市并发运；“交易”页可选择商品、调整数量后采购或出售；“车辆”和“融资”页分别处理维修、购车、借款与还款；底部“日结”推进一天。所有图形操作都通过游戏核心命令执行。

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

## 智能体动作

Agent 使用固定的扁平多头离散动作协议。`action_index` 直接选择 `BUY`、`SELL`、`TRAVEL`、`BORROW`、`REPAY`、`REPAIR_TRUCK`、`BUY_TRUCK`、`NEXT_DAY` 八种核心操作，不再拆分操作大类和子动作。

完整动作由六个固定顺序的索引组成：

```text
(action_index, product_index, city_index, transport_index, quantity_index, fast_index)
```

不同操作只读取自身需要的参数：采购和出售读取商品与数量，运输读取城市、运输方式和加急选项，借款、还款和购车读取数量；维修与推进日期不读取额外参数。商品和城市索引按当前 CSV 目录顺序生成，数量头包含 `1、5%、10% ... 95%、100%` 共 21 档：

```python
from trade_game.agent import (
    ActionHead,
    ActionVocabulary,
    decode_action,
)
from trade_game.core import CommandType, TransportMode, create_game_session

session = create_game_session(seed=7)
vocabulary = ActionVocabulary.from_catalog(session.catalog)
action = ActionHead(
    action_index=vocabulary.action_index(CommandType.TRAVEL),
    city_index=vocabulary.city_index("石家庄"),
    transport_index=vocabulary.transport_index(TransportMode.LAND),
)
command = decode_action(session, action, vocabulary)
result = session.dispatch(command)
```

动作协议只保存稳定索引，不依赖 NumPy、Gymnasium 或 PyTorch。one-hot、embedding 和策略网络属于后续学习层。

推理前使用动作掩码屏蔽当前不能执行的选择。掩码只包含布尔值：先读取 `action`，再按已选命令读取条件参数掩码。例如采购依次读取 `buy_product` 和 `buy_quantity[product_index]`；运输依次读取 `travel_city`、`travel_transport[city_index]`、`travel_fast[city_index][transport_index]`。

```python
from trade_game.agent import build_action_mask

mask = build_action_mask(session, vocabulary)
can_travel = mask.action[vocabulary.action_index(CommandType.TRAVEL)]
```
