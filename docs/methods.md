# 协同进化方法 / Co-evolution methods

六个连续更新方法共享同一个三 Agent 运行时，主要在**优势估计**与**策略项**上不同。
用 `python scripts/coevolve.py --list-methods` 查看当前注册表。
本仓库不提供离线训练流程或初始模型权重；连续更新从用户提供的
三个角色初始 LoRA 适配器开始。

## 共同设定

连续投放接口由 `scripts/run_ladder.py` 定义，并由
`scripts/online_rl/collector.py` 执行：Planner 输出账户剩余预算比例 `alpha`。

- 每周期三角色共同执行一条事实投放轨迹：Planner 在 `t=12` 分别规划各账户内单元的剩余预算，
  Optimizer 在 `t=12..47` 每 tick 每账户输出 `HOLD` 或 tCPA 倍率，
  Inspector 在周期末生成诊断与 guidance；
- 结算后一周期只调度**一个角色**训练，新版本从下一周期生效；
- 版本化行为策略：重要性比与 KL 都以生成该轨迹的版本为参考；
- 角色 LoRA 独立，PrefixValue / optimizer 状态跨更新持久化。

Planner 与 Optimizer 使用基础业务奖励

$$
Q(F,C,N)=\frac{F-\eta(C-\rho N)}{s}.
$$

其中，$F$ 为 ADV，$C$ 为实际成本，$N$ 为成熟转化数，
$\rho$ 为按成熟转化数加权的单元初始 tCPA。
代码直接使用目标成本额度 $M=\sum_u d_uN_u=\rho N$，
因此零转化时也无需计算 $\rho=M/N$。默认 $\eta=1$、$s=10000$，
账户与整体奖励按 $r=Q_{\mathrm{account}}+0.2Q_{\mathrm{all}}$ 组合。
Planner 使用观察阶段结束至周期末的结果，Optimizer 使用逐时间步的增量结果。
Inspector 使用相邻周期的标准化 ADV、成熟转化、CPA 余量及 forced-HOLD 反馈之差。
四项冻结统计量由 `--inspector-stats` 提供。

信用分配按账户隔离：Planner 每个账户在周期内只有一个规划节点，
Inspector 每条待结算诊断记录只有一个节点，二者使用该节点奖励减去各自的
`PrefixValueModel` 预测值作为优势。Optimizer 将同一账户的全部已结算决策节点
（包括非法回复后执行的 HOLD）按实际 tick 排列，只沿该账户的时间序列计算
skip-observation GAE；相邻节点的
环境步数由 tick 差重新计算，不把同一 tick 的其他账户接入递推。
Actor 参数更新只使用动作合法且具有完整 token 记录的节点。

记 `V_j` 为当前节点的价值预测，`r_j` 为其已结算奖励。Planner 与 Inspector
使用 `A_j = r_j - V_j`、`R_j = r_j`。对同一账户的相邻 Optimizer 节点，令
`Δ_j = tick_{j+1} - tick_j`，则
`δ_j = r_j + γ^{Δ_j}(1-d_j)V_{j+1} - V_j`，
`A_j = δ_j + γ^{Δ_j}λ(1-d_j)A_{j+1}`，`R_j = A_j + V_j`。
其中 `d_j` 表示该节点终止；各账户最后一个已结算节点视为终止节点。

## 1. AdCoEvo 连续更新（默认方法）

```bash
python scripts/coevolve.py --method adcoevo --env-profile sparse ...
```

- 优势：Planner / Inspector 使用单节点奖励减价值预测，Optimizer 使用按账户隔离的
  skip-observation GAE；三个角色分别维护标量 `PrefixValueModel`；
- 策略项：token 级双侧 ratio mask，只保留 `0.8 ≤ r_t ≤ 1.2` 的 token；
- 稳定机制：Actor 只优化策略损失；KL 仅用于每个 Actor epoch 前后的监控，
  超过阈值或有效 token 比例过低时回滚该 epoch 并停止，不参与反向传播；
- 发布：PolicyStore 原子版本发布，各角色首次调度仅训练价值模型。

## 2. PPO / CISPO / GSPO

```bash
python scripts/coevolve.py --method ppo   ...
python scripts/coevolve.py --method cispo ...
python scripts/coevolve.py --method gspo  ...
```

三者共用 `scripts/online_rl/learner.py`，差异只在策略项：

| 方法 | 策略项 |
|---|---|
| `ppo` | token 级 `min(r·A, clip(r)·A)` |
| `cispo` | detached 截断重要性权重乘 log-prob 梯度 |
| `gspo` | 序列级 ratio（token 平均）再裁剪 |

采集模式：`ppo` 为同步单轨迹，`cispo`/`gspo` 为异步单轨迹。

三者沿用上述信用分配方式。`gspo` 这一档只取 GSPO 的
**序列级 ratio 与裁剪**（token 平均后再裁剪），**不使用组采样**——协同进化每周期只有一条
事实轨迹，连续投放不会对同一状态采样多条续写。

## 3. SPO

```bash
python scripts/coevolve.py --method spo ...
```

- 不训练神经 Critic，改用**持久 EMA 历史 baseline**，读前写后；
- EMA 更新率随策略漂移自适应；
- 连续投放入口将最小 batch 设为 1，以便在每个调度周期发布新版本。

## 4. SR-PPO

```bash
python scripts/coevolve.py --method srppo \
  --thresholds configs/srppo_business_pass_v1.json ...
```

- `k=1` 单轨迹，用 Monte-Carlo **prefix credit head** 把终局业务信用分配到 token；
- 阈值文件给出每个角色的业务通过判据，必须显式提供。

## 5. 角色调度

一条市场序列是 **50 个连续投放周期**，角色按 **P → O → I 轮转**更新：

```text
period 0   1   2    3   4   5    ... 48  49
       P → O → I → P → O → I → ...  P → O
```

每个周期只调度轮到的角色，新版本从下一周期生效。50 周期下来，
Planner / Optimizer / Inspector 分别被调度 17 / 17 / 16 次；
首次价值模型预热或回滚可能使实际 Actor 参数更新次数更少。
各角色的已结算记录先进入各自缓存：Planner 和 Optimizer 在当期结算后进入，
Inspector 在下一期结果结算后进入。轮到某角色时，学习器一次消费兼容的已结算记录；
尚未结算或行为策略版本不同且未过期的记录保留到后续轮次。

用 `--role-schedule` 切换：

| 取值 | 含义 |
|---|---|
| `round_robin`（默认） | P → O → I 轮转 |
| `optimizer_only` | 每周期只更新 O，P 和 I 参数固定 |
| `po_alternating` | 只轮 P 和 O，Inspector 冻结（消融用） |

以上设置均保留 P、O、I 参与投放和跨周期诊断，只改变参数更新对象。
50 周期内，`optimizer_only` 调度 O 50 次，`po_alternating` 调度 P 和 O 各 25 次。

## 6. 常用可调项

| 参数 | 说明 |
|---|---|
| `--env-profile` | `sparse` 沿用 AuctionNet 原生参数；`general` 使用公开数据校准的参数 |
| `--episodes` | 连续周期范围，一条市场序列是 50 周期，例如 `0-49` 或 `71-120` |
| `--role-schedule` | 角色更新顺序，默认 `round_robin`（P → O → I） |
| `--window W` | 暖机 tick 数，默认 `12`；Agent 决策从 tick `W` 开始 |
| `--quota K` | 各单元每周期的 tCPA 调整次数上限，默认 `8`；超出的 `SET` 变 forced `HOLD` |
| `--pv` | 每周期机会数，正式实验用 `500000` |
| `--seed` | 训练随机性（环境市场主要由 episode 决定） |
| `--temperature` | 采样温度，连续更新与固定评测均默认为 `0.7` |
| `--learner-lr` / `--beta-kl` | 覆盖所选方法的相应参数；`--beta-kl` 不适用于 `adcoevo` |
| `--inspector-stats` | 冻结的 Inspector 奖励标准化统计量 JSON，连续更新时必填 |
| `--gpu` / `--rollout-gpu` / `--learner-gpu` | 推理卡与学习卡分工，单卡时一致即可 |

## 7. 消融与对照

Rule 根据观察阶段的单元 CPA 选择预算倍率：不高于目标时为 `1.1`，高于目标时为 `0.9`，
无转化时为 `1.0`。单元初始预算乘以该倍率后，按账户总预算归一化。
周期内 Rule 使用 `HOLD`、`0.7`、`1.3` 三种动作：CPA 高于目标的 `1.10` 倍时设置
倍率 `0.7`，低于目标的 `0.90` 倍时设置倍率 `1.3`，其余保持 `HOLD`。
无成熟转化或调整次数耗尽时保持 `HOLD`。重复设置当前倍率按执行层的统一规则视为 `HOLD`。

- `--ablation-mode full | no_inspector_feedback | shuffled_feedback | frozen_partners`
  控制反馈使用和参数更新；`frozen_partners` 保留正常诊断，每周期只更新 O；
- 连续运行的逐周期记录包含同 episode 的 native 无运营结果，可用于
  CPA 可行性判断和相对收益计算；独立 `--arm none` 运行是额外核对；
- 只改一个因素、其余固定，是所有比较的前提。

## 8. 固定策略执行

`scripts/evaluate.py` 使用同一套 Planner、Optimizer、Inspector 决策接口，
每周期结算后只传递 Inspector guidance，不调用学习器，也不发布新策略版本。
本地运行需提供基础模型和三个角色的 LoRA 适配器；已有 OpenAI 兼容服务时，
可通过 `--base-url` 与各角色 `--*-model` 指定固定模型。评测输出单独保存，
不代替以上连续更新方法的逐周期结果。
外部模型默认不启用受约束 JSON 生成，仍使用相同的格式解析与动作校验。
Inspector 不接收 No Operation 的参考结果。
