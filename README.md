# 风物千程 / TRADE

**风物千程** 是一款带有强机器学习实验属性的贸易决策小游戏，我主要通过Vibe Coding实现的，前端比较原始，灵感来自于而是玩过的一款贸易小游戏：

- **游戏层面**：玩家在 90 年代中国各城市之间采购 / 运输 / 售卖商品，管理借贷与车辆，追求在有限天数内最大化结算资产。
- **研究层面**：这是一个具有明显 **长程决策特性** 的强化学习任务——一次好的收益通常需要「买入 → 跨城运输 → 控制损耗与贷款 → 远期卖出」这一系列长期链条。
- **算法层面**：项目内置基于 **SB3（stable‑baselines3 + sb3_contrib）PPO + ActionMask** 的训练脚本，并支持使用 **人类演示 BC 预热策略** 加速训练。

目前该任务整体难度较高，冷启动周期长、前期无效探索占比大，现有策略的实际表现均不够理想。项目主要提供长程决策与复杂规则环境的仿真场景及算法示例，具体算法实现仍有待进一步优化与迭代。

---

## 1. 游戏与决策任务简介

### 1.1 一个决策贸易小游戏

- **世界观**：以 90 年代中国为背景的贸易模拟，包含若干城市、商品品类与陆运/海运网络。
- **玩家目标**：在给定天数内，通过跨城贸易、借贷、车辆管理等手段，最大化 **结算金额**。
- **核心操作**：
  - 在有市场的城市 **采购/售出** 商品；
  - 选择 **陆运/海运** 路线跨城运输，承担时间与损耗；
  - 在有银行的城市 **借贷/还贷**，控制破产风险；
  - 维修 / 购买卡车，提高运力；
  - 每天根据库存、价格与贷款情况做一次全局决策。

### 1.2 一个长程特性的机器学习决策任务

从 RL 角度看，本环境具有典型的 **长程、稀疏、带约束决策特性**：

- **长程收益链条**：一次赚钱动作通常跨越几十天：
  - Day 1–5：在 A 城低价买入；
  - Day 5–20：分段跨城运输（多次日推进 + 损耗）；
  - Day 20+：在高消费城市 B 卖出；
  - 期间需要控制现金流、贷款和破产风险。
- **强约束 / 不可逆性**：
  - 容量、耐久、贷款上限、是否有银行/港口等限制使得可行动作空间高度状态相关；
  - 一次错误的「满仓同城卖出+高杠杆」组合可能迅速导致破产。
- **冷启动极难**：
  - 随机策略几乎永远在原地「刷天数」，真实盈利轨迹极少；
  - 因此我额外提供了 **行为克隆预热（BC warm‑start）** 与 **动作 mask**，尽量减少完全无效探索。

---

## 2. 强化学习：PPO 训练与 BC 预热

### 2.1 环境封装与动作空间

强化学习部分主要位于：

- `trade_game/api.py`：环境的“干净 API”，定义 `ActionBuy/ActionSell/ActionBorrow/ActionRepay/...` 等结构化动作，以及 `reset/step/get_observation`。
- `trade_game/sb3_env.py`：
  - `TradeGameMaskedEnv`：离散动作 + **ActionMask** 版本 Gymnasium 环境，配合 `sb3_contrib.MaskablePPO` 使用。
  - `EnvConfig`：环境配置（`game_mode`, `max_days`, `max_steps`, `max_travel_choices` 等）。

**动作空间（MaskedEnv，Discrete N）：**

- `0`：`next_day`
- 一段连续区间：`buy` 槽位（商品 × **8 档数量变体**）
- 一段连续区间：`sell` 槽位（商品 × **8 档数量变体**）
- 两段：陆运 / 海运 travel 槽位（满足拓扑与港口规则）
- 借贷相关：
  - `borrow`：按当前可借额度的 **1/3、2/3、3/3** 三档金额；
  - `repay_all`；
  - `repay`：按当前可还额度的 **1/3、2/3、3/3** 三档（3/3 等价全还）。

**数量变体（8 档）**统一用于买入 / 卖出 / GUI 录制模式：

- 上限（可买/可卖上限）的：`1/5, 2/5, 3/5, 4/5, 5/5`
- 固定数量：`1, 2, 3`

**Action Mask** 负责过滤：

- 容量不足 / 现金不足的买入档位；
- 持仓不足的卖出档位；
- 不满足耐久 / 港口 /路径规则的旅行动作；
- 可借/可还额度不足的借贷动作。

#### 2.1.1 动作空间布局与编码示意

为了便于在论文 / 代码中引用，这里用更工程化的方式描述我的实现。

- **MaskedEnv 的 Discrete 索引布局**（见 `TradeGameMaskedEnv.__init__`）：
  - `a = 0`：`next_day`
  - 区间 `[1, B)`：`buy` 槽位，内部展平为  
    `slot = base_idx * 8 + qty_variant`  
    其中 `base_idx` 是第几个候选商品，`qty_variant ∈ {0..7}` 表示 8 档数量。
  - 之后 `[B, S)`：`sell` 槽位，同样是 “候选商品 × 8 档数量”。
  - 紧接着两段长度相同的区间：`travel_land` 与 `travel_sea` 槽位（每个元素就是一个具体目的地城市）。
  - 借贷相关索引：
    - `[borrow_start, borrow_end)`：`borrow` 三档金额（1/3, 2/3, 3/3）；
    - `repay_all_idx`：`repay_all`；
    - `[repay_start, repay_end)`：按 1/3, 2/3, 3/3 可还额度生成的还款动作。
- **数量变体的具体计算**（见 `_compute_qty_variant`）：
  - 给定当前可买/可卖上限 `M` 与变体索引 `k ∈ {0..7}`：
    - `k=0..4`：`qty = round_to_int( (k+1)/5 * M )`，并裁剪到 `[1, M]`；
    - `k=5,6,7`：分别是 `1, 2, 3`，再裁剪到 `[1, M]`。
  - 买入时的上限 `M` 同时考虑 **容量** 与 **现金**：  
    `M = min(剩余容量, floor(cash / price))`。
  - 卖出时的上限 `M = 当前持仓数量`。
- **借贷金额的计算**：
  - 对于借款：  
    先根据净资产 `net_assets = cash - total_debt` 与已借本金 `principal_total` 计算  
    `max_loan = max(0, net_assets - principal_total)`，  
    然后金额为 `max(100, frac * max_loan)`，`frac ∈ {1/3, 2/3, 3/3}`。
  - 对于还款：  
    上限 `max_repay = min(cash, total_debt)`，金额为 `max(100, frac * max_repay)`，  
    其中 3/3 档在实现上直接视为 `"all"`。
- **人类动作到离散 index 的编码**：
  - `trade_game/sb3_env.py::TradeGameMaskedEnv.encode_api_action(...)`：
    - 先调用 `action_mask()` 刷新当前可行动作，并缓存各类候选：
      - 可买商品列表 `_last_buy_cands`；
      - 可卖商品列表 `_last_sell_cands`；
      - 陆运/海运可达城市列表等。
    - 对于 `ActionBuy/ActionSell`：
      - 在当前状态下重新计算 `max_qty`（容量/现金/持仓约束）；
      - 在 8 个变体里搜索会产生 **与给定数量完全相同** 的 `qty_variant`，没有则返回 `None`；
      - 将 `(base_idx, qty_variant)` 映射回全局离散 index。
    - 对于 `ActionBorrow/ActionRepay`：
      - 根据当前 `max_loan/max_repay` 计算给定金额的比例；
      - 在 `{1/3,2/3,3/3}` 中找最接近的 fraction 槽位；
      - 3/3 + 比例≥0.99 时统一编码为「全部」。
    - 对于 `ActionTravel`：只在当前缓存的候选城市列表中查找索引。
  - 如果编码失败（数量不落在 8 档之一、商品不在候选集合、额度不足等），我会返回 `None`，  
    `HumanDemoRecorder` 会把这些动作统计到 `meta["dropped_details"]` 中，方便分析「人类操作与 RL 动作空间不兼容」的具体原因。

### 2.2 使用 SB3 平台的 PPO 训练智能体

训练脚本：`train_ppo.py`  
依赖：`sb3_contrib` (MaskablePPO), `stable-baselines3`, `gymnasium`, `torch`。

主要流程：

1. 使用 `EnvConfig(game_mode="free", max_days=90, max_steps=90)` 构造环境。
2. 通过 `make_vec_env(env_fn, n_envs=4)` 创建并行环境，外层包 `ActionMasker`。
3. 初始化 `MaskablePPO("MlpPolicy", env, ...)`。
4. 可选：使用 **基线脚本策略** 采样 demo 做少量 BC 预热（见下面 2.3）。
5. 通过 `model.learn(total_timesteps=...)` 进行 PPO 训练，过程中写入：
   - TensorBoard 日志：`runs/ppo_trade_game/tb/`
   - 周期性 checkpoint：`runs/ppo_trade_game/ppo_ckpt_*.zip`
   - 最终模型：`runs/ppo_trade_game/ppo_trade_game.zip`
6. 训练结束后，可选择用该模型自动游玩一局，在终端打印 `day/cash/reward` 轨迹。

> **重要说明**：  
> 由于任务长程且约束繁多，**从随机初始化直接 PPO 训练的冷启动极慢**，前期大量时间会浪费在「原地刷天数」和无效买卖上。因此推荐搭配第 2.3 节的 **行为克隆预热** 一起使用。

### 2.3 使用 BC 预热策略的 PPO 加速训练

我提供了两条行为克隆（BC）路径：

- 基于脚本基线策略自动采样 demo：
  - `trade_game/baseline_policy.py`：一个手写、保守的“玩家基线策略”（优先跨城卖出、其次移动、再次买入）。
  - `trade_game/ppo_warmstart.py::pretrain_policy_from_baseline`：  
    在单环境上用基线策略采样 `(obs, action)`，然后对 PPO policy 做若干轮监督训练（最小化 `-log π(a|s)`）。
- 基于 **人类录制轨迹** 的 demo：
  - `trade_game/human_demo.py` + `trade_game/arcade_app.py`：  
    在“玩家演示模式”或 GUI 中按 F8 录制人类游玩，轨迹保存为 `runs/demos/human_demo_*.npz`，内容为：
    - `obs`：与 `TradeGameMaskedEnv` 相同的观测向量；
    - `action`：离散动作 index；
    - `meta`：步数、丢弃的非法动作数量及原因、环境配置等。
  - `train_bc_from_demo.py`：**仅 BC 训练脚本**
    - 自动扫描并合并所有 `runs/demos/*.npz`；
    - 用合并后的 `(obs, action)` 对 MaskablePPO policy 做多轮 BC 训练；
    - 保存模型到 `runs/bc_trade_game/bc_model.zip`；
    - 可选：用该模型自动游玩一局并在终端打印进度。

**推荐训练流程：**

1. 人类在 GUI 中多次游玩，自动录制到 `runs/demos/`。
2. 运行 `python train_bc_from_demo.py`：
   - 合并所有 demo，做若干 epoch 的行为克隆；
   - 得到一个「类似人类风格，但不完美」的起始策略 `bc_model.zip`。
3. 运行 `python train_ppo.py`：
   - 选择“从已有模型继续训练”，默认加载 `runs/bc_trade_game/bc_model.zip`；
   - PPO 从 BC 起点继续训练，探索会比随机冷启动快很多。

---

## 3. 运行与安装

### 3.1 环境安装

建议使用虚拟环境（conda / venv），然后：

```bash
pip install -r requirements.txt
```

主要依赖包括：

- `arcade`：图形界面和输入处理；
- `gymnasium`、`stable-baselines3`、`sb3_contrib`：RL 训练框架；
- `numpy`, `torch` 等基础数值库。

建议使用 **Python 3.10–3.12**，并在有 CUDA 的环境下运行 RL 训练脚本（`train_bc_from_demo.py` / `train_ppo.py`），否则训练时间会明显变长。

### 3.2 启动游戏（人类游玩）

```bash
python start_game.py
```

- 菜单中可以选择：
  - 正常模式（自由 / 挑战 / 读档）；
  - 玩家演示模式（自动录制 RL demo）。
- 在 GUI 中按 F8 可手动开始/停止录制，轨迹保存到 `runs/demos/*.npz`。

### 3.3 行为克隆（仅 BC）

```bash
python train_bc_from_demo.py
```

- 自动合并 `runs/demos` 内所有 `.npz`，做 BC 训练。
- 生成模型：`runs/bc_trade_game/bc_model.zip`。
- 可选自动试玩一局并打印进度。

### 3.4 PPO 训练

```bash
python train_ppo.py
```

交互过程：

- 输入 PPO 的总训练步数；
- 选择是否从已有模型继续训练（推荐选择 `runs/bc_trade_game/bc_model.zip`）；
- 可选：是否在 PPO 前再做一轮「基线脚本 BC 预热」；
- 训练完成后，可让模型自动跑一局并打印 `step/day/cash/reward`。

### 3.5 用 PPO 模型自动游玩（带 GUI）

除了 `train_ppo.py` 里提供的「终端打印版试玩」，我还提供了一个单独的脚本 `play_ppo.py`，用于在 GUI 中让智能体自动游玩并观察效果：

```bash
python play_ppo.py
```

脚本会：

- 加载你指定的 PPO 模型（默认是 `runs/ppo_trade_game/ppo_trade_game.zip` 或某个 checkpoint）；
- 启动与人类相同的 Arcade GUI；
- 由智能体自动选择动作，你可以在界面上直观看到它的买入 / 出行 / 卖出等行为。

> 再强调一次：**这是一个复杂长程任务，目前所有策略都不算理想**。  
> 期望它在少量训练步数内“学会稳定赚钱”是不现实的，更适合作为研究长程决策与探索难度的案例。

---

## 4. 代码结构概览

```text
TRADE/
├── start_game.py                 # 游戏启动入口（GUI + 菜单）
├── train_bc_from_demo.py         # 基于全部人类 demo 的纯行为克隆训练脚本
├── train_ppo.py                  # 基于 PPO 的训练脚本，可从 BC 模型继续训练
├── check_demo.py                 # 检查 runs/demos/*.npz 是否与当前环境兼容
├── test_prices.py                # 价格系统 / 波动范围测试
├── trade_game/
│   ├── api.py                    # 环境 API：Action 定义、reset/step/get_observation
│   ├── arcade_app.py             # Arcade 图形界面与人类交互逻辑
│   ├── cli.py                    # 命令行界面（备用）
│   ├── data.py                   # 商品 / 城市 / 路线静态数据
│   ├── economy.py                # 价格模型与 λ 波动
│   ├── inventory.py              # 库存与损耗
│   ├── transport.py              # 路网、运输时间与模式校验
│   ├── loans.py                  # 借贷系统（Loan、利息、破产判定）
│   ├── state.py                  # GameState / PlayerState 定义
│   ├── save_load.py              # 存档 / 读档
│   ├── timeflow.py               # 日推进逻辑与结算触发
│   ├── capacity_utils.py         # 货车容量计算
│   ├── game_config.py            # 普通游戏平衡参数（运输成本、车辆、价格波动等）
│   ├── train_config.py           # 训练 / CLI / 录制模式规则（数量档位、天数上限、结算评分等）
│   ├── sb3_env.py                # Gymnasium 环境封装（MaskedEnv + CompactEnv）
│   ├── baseline_policy.py        # 手写基线策略（脚本玩家）
│   ├── human_demo.py             # 人类演示录制与保存
│   ├── ppo_warmstart.py          # BC 预热工具（基线 / demo 文件）
│   └── ...                       # 其它辅助模块
└── runs/
    ├── demos/                    # 人类录制的 demo（*.npz）
    ├── bc_trade_game/            # BC 模型输出
    └── ppo_trade_game/           # PPO 训练日志与模型
```

### 4.1 可执行脚本一览

| 脚本                     | 作用简介 |
|--------------------------|----------|
| `start_game.py`          | 启动 Arcade GUI，让人类游玩（支持玩家演示模式自动录制 demo）。 |
| `train_bc_from_demo.py`  | 合并 `runs/demos/*.npz` 中所有人类轨迹，对 PPO policy 做纯行为克隆预训练，保存为 `runs/bc_trade_game/bc_model.zip`。 |
| `train_ppo.py`           | 基于 `TradeGameMaskedEnv` 的 MaskablePPO 训练脚本，可从 BC 模型继续训练，并提供终端版自动试玩。 |
| `play_ppo.py`            | 在 GUI 中加载某个 PPO 模型，让智能体自动游玩，方便可视化观察策略行为。 |
| `check_demo.py`          | 检查单个 demo `.npz` 是否与当前环境兼容（obs 维度、动作范围、丢弃动作统计等）。 |
| `test_prices.py`         | 价格与 λ 波动测试脚本，用于快速检查商品价格区间与成本。 |


---

## 5. 状态与贡献

- 本仓库目前主要用于 **实验与教学**，而不是「已经调好的一键商战 AI」。
- 环境本身规则复杂、冷启动困难、探索效率低，欢迎：
  - 尝试不同的 RL 算法（例如分层 RL、Model‑Based RL、分布式探索等）；
  - 改进 reward shaping / curriculum / imitation 方案；
  - 优化 GUI 与可视化，让人类和智能体更容易对比行为。

如有改进意见或实验结果，欢迎提 issue / PR，一起把这个 **长程贸易决策任务** 打磨成更通用的开源基准。  

