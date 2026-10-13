# AdCoEvo

AdCoEvo 是一个面向广告投放的协同进化多智能体框架。Planner 规划账户内单元预算，Optimizer 调整单元的控制 tCPA，Inspector 诊断结算结果，为下一周期的账户规划和策略调整提供指导。

框架使用 LangGraph 调度周期内的角色协作，并在连续投放中依据结算反馈更新模型参数。模拟环境采用 AuctionNet，逐流量出价由其 Abid 出价器执行。

[English](README.md)

## 安装

```bash
git clone https://github.com/jd-opensource/AdCoEvo.git adcoevo
cd adcoevo

git clone https://github.com/alimama-tech/AuctionNet

conda create -n adcoevo python=3.10 -y
conda activate adcoevo
pip install -r requirements.txt
```

请使用与 GPU 和 CUDA 环境兼容的 PyTorch、vLLM 版本。AuctionNet 默认位于仓库根目录下的 `AuctionNet/`；若放在其他目录，通过 `ADCOEVO_AUCTIONNET` 指定路径。

## 环境与决策接口

默认控制 3 个账户，每账户包含 4 个单元。每周期有 48 个时间步和 500,000 次流量机会。

| Profile | pValue 生成基准 | 初始 tCPA 范围 | 48 单元总预算 |
|---|---:|---:|---:|
| `sparse` | 0.0005 | 60–130 | 150,000 |
| `general` | 0.005 | 6–12 | 150,000 |

Sparse 沿用 AuctionNet 原生模拟器参数。General 根据公开 General 数据配置 pValue 生成基准、48 维预算和初始 tCPA，其他拍卖机制保持原生实现。配置与来源见 [环境说明](docs/environment_profiles.md)。

| 参数 | 含义 | 默认值 |
|---|---|---:|
| `--window W` | 前 W 个时间步保持初始预算和 tCPA，用于观察；随后开始 Agent 决策 | 12 |
| `--quota K` | 各单元每周期的 tCPA 调整次数上限 | 8 |
| `--budget-scale` | 受控账户初始预算的倍率 | 1.0 |
| `--temperature` | 模型采样温度 | 0.7 |

Planner 输出账户内单元的剩余预算比例 `alpha`。Optimizer 输出 `HOLD` 或 tCPA 倍率 `0.5、0.6、0.7、0.8、1.0、1.2、1.3`；重复设置当前倍率按 HOLD 执行，超过调整次数上限的 SET 按 forced HOLD 执行。Inspector 接收系统自身的结算结果与动作记录，输出诊断和指导。

Rule 对照根据各单元 CPA 选择预算倍率，再归一化为账户总预算；它与 Planner 的剩余预算比例接口分别实现。
周期内 CPA 高于目标的 1.10 倍时设置 tCPA 倍率 `0.7`，低于目标的 0.90 倍时设置 `1.3`，其余保持 `HOLD`。无成熟转化或调整次数耗尽时保持 `HOLD`。

## 协同进化

运行前需准备本地因果语言模型、三个兼容的角色 LoRA 适配器，以及 Inspector 奖励标准化统计量。本仓库提供连续投放与策略更新代码，不包含模型权重和离线训练流程。

```bash
export ADCOEVO_BASE_MODEL=/path/to/base-model
# tokenizer 位于其他目录时设置
export ADCOEVO_TOKENIZER=/path/to/tokenizer
```

从与所选环境匹配的历史结算轨迹计算 ADV 和转化数的均值、标准差，保存为 JSON，并在运行期间保持固定。以下仅展示字段格式，需替换为实际统计量：

```json
{"mu_F": 0.0, "sigma_F": 1.0, "mu_N": 0.0, "sigma_N": 1.0}
```

```bash
python scripts/coevolve.py \
  --env-profile sparse --method adcoevo \
  --episodes 0-49 --window 12 --quota 8 --temperature 0.7 \
  --run-dir runs/coevolve_demo --gpu 0 --port 8001 \
  --inspector-stats /path/to/inspector_reward_stats.json \
  --planner-init /path/to/planner-lora \
  --optimizer-init /path/to/optimizer-lora \
  --inspector-init /path/to/inspector-lora
```

每周期依次完成观察、预算规划、逐步投放调整和期末诊断。结算反馈进入相应角色的训练记录，按 `Planner → Optimizer → Inspector` 轮转调度训练，发布的新策略从下一周期生效。默认 `adcoevo` 方法在各角色首次调度时先训练价值模型，后续调度再更新策略参数。

`--role-schedule optimizer_only` 每周期只更新 Optimizer；`po_alternating` 交替更新 Planner 和 Optimizer。这两种设置均保留三个角色参与投放及诊断指导。

```bash
python scripts/coevolve.py --list-methods
python scripts/coevolve.py --env-profile general --method adcoevo --print-config
```

| 方法 | 更新机制 |
|---|---|
| `adcoevo` | PrefixValueModel、按角色计算优势、token 重要性比筛选与 KL 回滚 |
| `ppo` | token 级 PPO 裁剪 |
| `cispo` | 截断并停止梯度的重要性权重 |
| `gspo` | 序列级重要性比与裁剪 |
| `spo` | 持久 EMA 历史基线 |
| `srppo` | Monte-Carlo prefix credit |

每种方法每周期采集一条实际执行轨迹。`gspo` 使用序列级重要性比与裁剪，连续更新中的优势按角色计算。采用 PrefixValueModel 的方法中，Planner 和 Inspector 使用奖励减价值预测，Optimizer 按账户、按周期分别计算 GAE。默认 `adcoevo` 的 Actor 仅优化策略损失，KL 用于监控与回滚。详细参数见 [方法说明](docs/methods.md)。

## 固定策略与通用模型

固定策略入口执行同一套角色协作流程，保留跨周期诊断指导，不更新模型参数：

```bash
python scripts/evaluate.py \
  --env-profile sparse --episodes 0-49 \
  --window 12 --quota 8 --temperature 0.7 \
  --out runs/frozen_demo --gpu 0 --port 8001 \
  --planner-init /path/to/planner-lora \
  --optimizer-init /path/to/optimizer-lora \
  --inspector-init /path/to/inspector-lora
```

也可使用已有的 OpenAI 兼容 API，无需本地模型或适配器：

```bash
python scripts/evaluate.py \
  --base-url https://your-endpoint/v1 --model your-model \
  --api-key-env YOUR_API_KEY \
  --env-profile sparse --episodes 0-49 \
  --temperature 0.7 --out runs/api_demo
```

`YOUR_API_KEY` 是保存密钥的环境变量名。可通过 `--planner-model`、`--optimizer-model`、`--inspector-model` 分别指定角色模型。

外部 API 默认通过提示规定 JSON 格式，本地适配器默认使用受约束 JSON 生成；可用 `--constrained-json` 或 `--no-constrained-json` 显式选择。两种方式采用相同的输出解析和动作校验。固定策略运行无需 Inspector 奖励统计量。

## 结果与恢复

连续更新的逐周期结果保存在 `eval/arms.jsonl`，汇总在 `eval/summary.json`；固定策略结果保存在 `arms.jsonl` 和 `eval_report.json`。同一方法、环境和预算的多个市场窗口可合并汇总：

```bash
python scripts/summarize_results.py \
  --input runs/window_a/eval/arms.jsonl runs/window_b/eval/arms.jsonl \
  --out runs/overall_summary.json
```

ADV 按逐周期均值汇总，CPA 按总成本除以总成熟转化数计算。非法输出率 `IR` 为非法回复数除以模型回复总数，综合得分为：

```text
Score = (ADV / ADV_0) * (CPA_0 / CPA) - IR
```

参照指标来自同一市场、环境和预算下另行执行的 No Operation 轨迹，仅用于效果评估。公式中的 IR 使用小数，`IR_percent` 用于百分比展示。CPA 或得分无定义时输出 `null`。

中断后保留完整运行目录，用相同命令加 `--resume` 继续。AuctionNet 的主要市场随机性由 episode 决定；比较方法时应使用相同 episode 序列，考察不同市场时使用不同窗口。详细协议见 [运行与汇总说明](docs/reproduction.md)。

## 许可与贡献

本项目采用 [Apache License 2.0](LICENSE)。AuctionNet 完整模拟器和原始数据需单独获取，改编代码的来源及许可见 [NOTICE](NOTICE)。开发环境与贡献方式见 [贡献指南](CONTRIBUTING.zh-CN.md)。
