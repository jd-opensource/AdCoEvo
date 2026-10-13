# 环境 Profile / Environment profiles

## 1. 两种环境设置

`sparse` 沿用 AuctionNet 原生模拟器的 pValue 生成基准、48 个广告主预算和
CPA 约束。本项目将这些原生值显式记录为 profile，便于切换和核对。
`general` 则根据 AuctionNet 公开 General 数据调整这三类参数；其余拍卖与流量生成机制
保持 AuctionNet 原生实现。运行时使用两套**绝对参数**，不做事后缩放。

## 2. 参数来源与核对

| Profile | 数据用途 | 数据包 | period | 转化 | pValue 均值 | CPAConstraint |
|---|---|---|---:|---:|---:|---|
| sparse | 核对原生设置 | `autoBidding_aigb_track_final_data_period_7-8.zip` | 7 / 8 | 1,308 / 1,354 | 0.0004787 / 0.0005423 | [60,130] |
| general | 校准参数 | `autoBidding_aigb_track_data_period_7-8.zip` | 7 / 8 | 12,263 / 12,839 | 0.0047868 / 0.0054231 | [6,12] |

逐行核验结论：

- general 的 `pValue` 与 `pValueSigma` 是 sparse 的**精确 10 倍**；
- 两条轨的 48 维预算与 CPAConstraint 是**各自独立**的数组，不能用统一倍率互相换算；
- 两条轨的 48 个广告主总预算都等于 150,000。

## 3. Profile 已随包提供

Sparse 的原生参数和 General 的校准参数均已写入 `configs/auctionnet_profiles.yaml`，
**开箱可用，无需重新提取数据**。每个 profile 使用 `calibration_id` 和源数据包哈希标识参数来源。

| Profile | `calibration_id` |
|---|---|
| sparse | `aigb-sparse-period-7-8-v1` |
| general | `aigb-general-period-7-8-v1` |

若你需要从上游原始压缩包重新生成（例如换了数据版本），再用
`scripts/prepare_auctionnet_profiles.py`；脚本流式读取，不长期解压数 GB 的 CSV。
平时不需要运行它。

## 4. 运行时如何使用

`env/auctionnet_profile.py` 严格校验 profile（必须 48 维、全正、白名单名称），
`env/auctionnet_sim_env.py` 使用其绝对参数：

- 48 维预算 → 我方单元、native 竞争者、Abid、over-cost 记账、Planner `base_budget`；
- 48 维 CPAConstraint → `desired_cpa`、Abid 基础 CPA、`control_tCPA = d_u × m`；
- `pvalue_mean_base` → **在 PV 生成前**设置生成器基准；Sparse 使用原生值，
  General 使用校准值，不在生成后再乘系数。

保持不变的 AuctionNet 机制：48 ticks、流量时序、pValueSigma、GSP 拍卖、曝光与转化抽样、
native heterogeneous competitor mix、Abid 公式。

明确不做：出清价标定、竞争者 bid 缩放、生成后 pValue 乘法、成本后处理。

## 5. 一键切换

```bash
python scripts/coevolve.py --env-profile sparse  --method adcoevo --print-config
python scripts/coevolve.py --env-profile general --method adcoevo --print-config
```

逐周期记录与 resume state 都会写入 `environment_profile`，
不同 profile 的结果不会被混写。

## 6. 额外压力档（可选）

仓库还保留 `extra_sparse_feedback_stress`：在官方 sparse 之上再做成熟转化 thinning，
用于研究极低反馈密度。它**不是**官方 AuctionNet 轨，默认不使用。

## 7. 跨 profile 比较的正确做法

两个 profile 的 pValue 与 CPAConstraint 都不同，因此：

- 不要直接比较跨 profile 的绝对 ADV 或 CPA；
- 应在同一 profile 内比较各方法，并优先报告相对同周期无运营基线的**相对提升**；
- 若必须跨 profile，只比较方向与相对 uplift，并明确说明参数差异。
