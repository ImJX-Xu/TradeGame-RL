# TradeGame-RL

一个以交易规则为核心、支持人类游玩与强化学习研究的回合制贸易游戏。

代码按严格单向依赖组织为三层：

1. `trade_game.core`：唯一的游戏规则与状态变更来源。
2. `trade_game.ui`：人类界面，只将交互转换为核心命令。
3. `trade_game.agent`：智能体动作协议、动作掩码、状态观测与核心命令解码。

后续的 PyTorch 编码器、策略网络和训练算法将只依赖这三层提供的稳定接口。

## 游玩

图形界面是可选依赖，首次使用前安装：

```powershell
pip install -e ".[ui]"
```

智能体张量批处理与后续 PyTorch 网络使用独立可选依赖：

```powershell
pip install -e ".[learning]"
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

## 智能体状态观测

`build_observation` 将 `GameSession` 转换为不可变的结构化状态快照。它不依赖 NumPy、PyTorch 或 Gymnasium；后续训练层负责将其按批次转换为张量。

```python
from trade_game.agent import build_observation

observation = build_observation(session, vocabulary)
print(len(observation.market_quotes))       # 14 个城市
print(len(observation.market_quotes[0]))    # 每城 18 个商品
print(observation.market_history_valid)     # D-6、D-4、D-2、D 的有效位置
```

市场状态以 `城市 x 商品 x 4` 的参考出售价格矩阵表示，四个时间点固定为 `D-6、D-4、D-2、D`。每个市场单元还携带采购可用性；价格统一按商品基础进价取对数比例，避免不同商品量级直接干扰网络训练。

观测还包括：全局经营状态、城市和商品的公开静态属性、从当前位置出发的陆运和海运估算，以及保留真实产地、保质期和 FIFO 顺序的可变长度货物批次。市场电报文本、事件持续时间、趋势项、局部价差和事件振幅均不进入观测；智能体只能从公开报价历史推断市场走势。

动作掩码与状态观测保持分离。策略网络使用状态理解市场和经营状况，再用掩码排除无法提交给 `GameSession.dispatch` 的候选动作。

学习层先将结构化观测批量化为 PyTorch 张量，再由 `StateEncoder` 执行实体嵌入、行情编码与目标注意力汇聚：

```python
from trade_game.learning import ActionMaskBatch, ObservationBatch, StateEncoder

observations = [build_observation(session, vocabulary)]
masks = [build_action_mask(session, vocabulary)]
state_batch = ObservationBatch.from_observations(observations)
mask_batch = ActionMaskBatch.from_masks(masks)
encoder = StateEncoder(state_batch.spec)
state = encoder(state_batch).state  # [batch_size, 256]
```

市场价格保持为 `[B, 城市, 商品, 4]`，货物批次按当前 batch 的最大批次数动态补齐；`cargo_valid` 标记真实批次，动作掩码仍独立于状态张量。编码器以当前全局经营状态为查询，分别从市场、路线和货物实体中汇聚当前最相关的信息，再输出固定维度状态向量。

## 策略与价值网络

`ActorCritic` 共享 `StateEncoder`。策略按命令实际使用的参数顺序生成联合动作概率：采购和出售为“商品、数量”，运输为“城市、运输方式、加急”，借贷和购车为“数量”；维修和推进日期没有额外参数。每个条件分支都在采样前读取对应的动作掩码。

```python
from trade_game.learning import ActorCritic

model = ActorCritic(state_batch.spec)
sample = model.sample(state_batch, mask_batch)
action_tensor = sample.action.as_tensor()  # [batch_size, 6]
```

训练时使用 `model.evaluate_actions(state_batch, mask_batch, actions)` 重算同一批轨迹动作的联合 `log_prob`、条件熵和全局价值 `V(s)`，供后续 PPO 目标函数使用。

## 训练回合

`AgentEnvironment` 默认创建 150 天挑战回合。`reset` 返回初始观测和动作掩码；`step` 只接受 `ActionHead`，并始终通过动作解码器和 `GameSession.dispatch` 执行规则。

默认 `reward_v1` 使用可清算经营资产的对数增量作为密集奖励。现金、货物当前变现价值、货车残值和债务统一计入资产，因此借款不会被误判为利润；挑战终局按最终资产给予额外奖励，破产会受到额外惩罚。每个转移还记录实际经过天数，供训练算法按游戏日而非决策次数折扣。
