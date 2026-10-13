# 复现协议 / Reproduction protocol

## 1. 范围与固定设置

本仓库提供连续投放与角色策略更新流程。离线训练代码、基础模型和三个角色的初始 LoRA
适配器不在发布范围内。运行前请自行准备与基础模型兼容的适配器；比较不同运行配置时，应
使用相同的初始 checkpoint。
Planner 适配器需使用账户内单元剩余预算比例 `alpha` 作为输出接口。
Inspector 的反馈标准化还需要从离线训练轨迹计算并冻结 `mu_F`、`sigma_F`、
`mu_N`、`sigma_N`，保存为 JSON：

```json
{"mu_F": 0.0, "sigma_F": 1.0, "mu_N": 0.0, "sigma_N": 1.0}
```

上面仅展示字段格式，数字仅为占位示例。实际运行必须替换为相应
训练集的统计结果。同一环境 Profile 的五个市场窗口使用相同统计量；
General 与 Sparse 可分别使用各自训练集的冻结统计量。

| 项目 | 设置 |
|---|---|
| 环境 | AuctionNet simulator，native heterogeneous competitors |
| 底层出价器 | Abid |
| 每周期流量 | `pv=500000` |
| 周期长度 | 48 ticks |
| 单条市场序列 | 50 个连续周期 |
| 角色调度 | 每周期 P → O → I 轮转（`--role-schedule round_robin`） |
| 观察长度 | `W=12`（`--window`） |
| 单元 tCPA 调整次数上限 | `K=8` / 周期（`--quota`） |
| 我方组合 | 3 账户 × 4 同类目单元 = 12 单元 |
| 采样温度 | 连续更新、固定策略和模型对比均为 `0.7` |

tCPA 动作集合：`HOLD, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2, 1.3`。
`HOLD` 保持当前倍率且不消耗调整次数；重复设置当前档被规范化为 `HOLD`；
达到上限后的 `SET` 变成 forced HOLD。

## 2. 主评测：连续执行、结算与更新

以下命令执行一个 50 周期市场序列。每周期共同执行三个角色的决策，结算后只调度
一个角色更新，新策略从后续周期生效：

```bash
export ADCOEVO_BASE_MODEL=/path/to/compatible/base-model
python scripts/coevolve.py \
  --env-profile sparse --method adcoevo \
  --role-schedule round_robin --window 12 --quota 8 \
  --episodes 71-120 --pv 500000 --seed 30380001 \
  --run-dir runs/sparse_71_120 --gpu 0 --port 8001 \
  --inspector-stats /path/to/inspector_reward_stats.json \
  --planner-init /path/to/planner-lora \
  --optimizer-init /path/to/optimizer-lora \
  --inspector-init /path/to/inspector-lora
```

逐周期事实结果写入 `runs/sparse_71_120/eval/arms.jsonl`。
`coevolve_manifest.json`、`run_manifest.json` 和 `policies/manifest.json`
记录配置及策略版本。主评测直接汇总 `arms.jsonl`，其中同时保存各周期匹配的
native 无运营结果（`native_F`、`native_C`、`native_N`）。
无需在相同周期上用最终冻结模型重新评测来替代主结果。

五个市场窗口示例为 `71-120`、`121-170`、`171-220`、`221-270`
和 `271-320`。每个窗口应从同一初始 checkpoint 开始，使用单独的
`--run-dir`；不要把前一窗口结束时的角色权重作为下一窗口的初始权重。
General 环境使用相同的窗口及 `--env-profile general`，分别保存结果。

## 3. 指标与汇总

对单元 `u` 的曝光集合 `I_u`，固定口径为：

```text
q_u = Σ_{j ∈ I_u} pValue_{u,j}
F   = Σ_u d_u q_u            # ADV，d_u 是模拟器给定的初始 tCPA
CPA = Σ_u Cost_u / Σ_u N_u   # 汇总成本 / 汇总成熟转化
```

Optimizer 改变控制 tCPA 时，评价所用的 `d_u` 不变。
单条 50 周期序列的 ADV/周期为 `sum(F) / 50`，组合 CPA 为
`sum(C) / sum(N)`。跨五个等长窗口汇总时，ADV/周期为全部 250 条记录的
`F` 均值，CPA 仍为全部 `C` 之和除以全部 `N` 之和；
不要对逐周期 CPA 或五个窗口 CPA 取简单平均。

非法输出率 `IR = sum(illegal) / sum(agent_requests)`；固定评测日志的回复总数
字段为 `llm_calls`。分母为模型回复数，不能用 `set_total`（实际执行的调整次数）
或 `set_requests`（单元 SET 请求数）替代。无模型调用的 No Operation 和 Rule 的 IR 为 0。
综合得分为 `Score = (ADV / ADV_0) * (CPA_0 / CPA) - IR`，参照值由相同市场、
环境和预算下的 No Operation 结果汇总得到。公式中的 IR 为小数，`IR_percent`
用于百分比展示。零转化等使 CPA 或得分无定义时，报告为 `null`。

连续更新结果自动汇总到 `eval/summary.json`，固定评测写入 `eval_report.json`，
Rule 和 AdPilot-style 写入 `summary.json`。合并同一方法、环境和预算的五个窗口：

```bash
python scripts/summarize_results.py \
  --input runs/sparse_71_120/eval/arms.jsonl \
          runs/sparse_121_170/eval/arms.jsonl \
          runs/sparse_171_220/eval/arms.jsonl \
          runs/sparse_221_270/eval/arms.jsonl \
          runs/sparse_271_320/eval/arms.jsonl \
  --out runs/sparse_overall_summary.json
```

不要在同一份汇总中混合不同模型、环境或预算条件。

## 4. 固定策略评测（不更新）

需要评测一组三角色权重而不做在线更新时，使用独立入口。以下命令沿用
Planner 暖机后规划、Optimizer 逐 tick 决策、Inspector 期末诊断与跨周期建议的
执行流程；全部周期使用同一组三角色适配器：

```bash
export ADCOEVO_BASE_MODEL=/path/to/compatible/base-model
python scripts/evaluate.py \
  --env-profile sparse --episodes 71-120 --pv 500000 --seed 30380001 \
  --window 12 --quota 8 --temperature 0.7 \
  --out runs/frozen_sparse_71_120 --gpu 0 --port 8001 \
  --planner-init /path/to/planner-lora \
  --optimizer-init /path/to/optimizer-lora \
  --inspector-init /path/to/inspector-lora
```

此入口不启动学习器，也不发布新版本；Inspector guidance 仍会传给下一周期。
逐周期事实结果在 `arms.jsonl`，汇总在 `eval_report.json`，所用配置和
适配器哈希在 `frozen_manifest.json`。`--print-config` 会检查本地基础模型与
适配器并显示命令，但不启动服务或模拟器。已有 OpenAI 兼容推理服务时，
可改用 `--base-url http://localhost:8001/v1` 及
`--planner-model`、`--optimizer-model`、`--inspector-model`，无需本地适配器参数。
此协议不计算训练奖励，因此无需 `--inspector-stats`。
外部 API 默认仅通过提示规定输出格式，不启用受约束 JSON 生成；本地适配器默认启用。
可用 `--constrained-json` / `--no-constrained-json` 显式选择。两种模式都保留相同的
输出解析和动作校验；Inspector 输入只包含系统自身的结算结果与动作记录。
冻结记录也包含同周期 native 无运营参照值，可按第 3 节计算相对收益。

连续更新的主结果已有逐周期匹配的 native 无运营参照值；汇总时仍按第 3 节
口径计算。`--gate` 可在协同进化结束后对**最终冻结版本**执行额外评测。
这两种冻结评测与逐周期持续更新的主结果是不同协议，须单独报告。

## 5. 断点恢复

长实验逐周期落盘。中断后，保留完整的 `--run-dir`（包括策略版本、队列和
`eval/resume_state.json`），用相同参数重运行并加 `--resume`：
若中断发生在周期结算与状态落盘之间，也要保留
`eval/pending_episode.json` 和 `queue.sqlite`；恢复时会完成该周期，
而不会用更新后的策略重跑同一周期。

```bash
python scripts/coevolve.py \
  --env-profile sparse --method adcoevo \
  --role-schedule round_robin --window 12 --quota 8 \
  --episodes 71-120 --pv 500000 --seed 30380001 \
  --run-dir runs/sparse_71_120 --gpu 0 --port 8001 \
  --inspector-stats /path/to/inspector_reward_stats.json \
  --planner-init /path/to/planner-lora \
  --optimizer-init /path/to/optimizer-lora \
  --inspector-init /path/to/inspector-lora \
  --resume
```

恢复流程检查已完成周期前缀、运行配置和已保存的策略版本，
避免把不同设置的结果追加到同一目录。
固定策略评测也支持用相同参数加 `--resume`，并校验环境设置及适配器哈希；
它不需要策略版本目录或更新队列。

## 6. 市场窗口与随机性

AuctionNet 的主要市场随机性由 episode 编号决定，因此比较方法时应使用
**同一窗口的相同 episode**；考察不同市场时使用互不重叠的窗口，
仅修改外部 `--seed` 不会产生一条新的独立市场序列。
`--seed` 仍会影响采样与训练过程。报告方差时，应明确区分市场窗口变化
与训练随机性。
