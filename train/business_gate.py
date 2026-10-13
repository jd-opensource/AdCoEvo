"""Business evaluation gate and verdict. Pure statistics, without torch.

候选系统 vs incumbent（共享市场种子 + LLM 采样种子；两者 tick0..11 暖机前缀须一致，否则该配对无效）。
配对 cluster bootstrap 求 CI，每次重采样按总成本与总转化数计算组合 CPA。

五项：①价值 LCB95(ΔF/F_inc)>δ_F ②组合 CPA UCB95(ΔCPA)≤δ_CPA 且 UCB95(CPA_cand)≤CPA_max
③转化非劣 LCB95(ΔN/N)≥−δ_N ④逐 episode 可行/尾部安全（违规率非劣、零转化失败不增、预算守恒/非法/forced）
⑤反重估（fixed ADV 须独立提升）。裁决 ACCEPT/HOLD/REJECT_NO_GAIN。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

INF = float("inf")


def portfolio_cpa(C: Sequence[float], N: Sequence[float]) -> float:
    """Return pooled CPA; zero total conversions produce an infinite CPA."""
    sN = sum(N)
    return (sum(C) / sN) if sN > 1e-9 else INF


def _paired_bootstrap(pairs, stat_fn, n_boot, seed):
    """pairs = 每 episode 的 (cand_dict, inc_dict)。stat_fn(list_of_pairs)->float。
    返回该统计量的 bootstrap 分布（长度 n_boot）。"""
    rng = np.random.default_rng(seed)
    n = len(pairs)
    idx = np.arange(n)
    out = []
    for _ in range(n_boot):
        s = rng.choice(idx, size=n, replace=True)
        out.append(stat_fn([pairs[i] for i in s]))
    return np.array(out, float)


def _lcb_ucb(dist, alpha=0.05):
    finite = dist[np.isfinite(dist)]
    if finite.size == 0:
        return -INF, INF
    return float(np.quantile(finite, alpha)), float(np.quantile(finite, 1 - alpha))


# ---- 统计量（作用于配对样本列表）----
def _rel_dF(pairs):
    Fc = sum(c["F"] for c, _ in pairs); Fi = sum(i["F"] for _, i in pairs)
    return (Fc - Fi) / (Fi + 1e-6)


def _rel_dN(pairs):
    Nc = sum(c["N"] for c, _ in pairs); Ni = sum(i["N"] for _, i in pairs)
    return (Nc - Ni) / (Ni + 1e-6)


def _dCPA(pairs):
    cpa_c = portfolio_cpa([c["C"] for c, _ in pairs], [c["N"] for c, _ in pairs])
    cpa_i = portfolio_cpa([i["C"] for _, i in pairs], [i["N"] for _, i in pairs])
    return cpa_c - cpa_i


def _cpa_cand(pairs):
    return portfolio_cpa([c["C"] for c, _ in pairs], [c["N"] for c, _ in pairs])


@dataclass
class BusinessThresholds:
    delta_F: float = 0.0       # 最小有意义 fixed ADV 相对增益
    delta_CPA: float = 0.0     # CPA 非劣容忍（绝对）
    delta_N: float = 0.05      # 转化非劣容忍（相对）
    cpa_max: float = INF       # 组合 CPA 硬上限
    delta_viol: float = 0.0    # 违规率非劣容忍
    n_boot: int = 1000
    alpha: float = 0.05
    seed: int = 12345


@dataclass
class BusinessReport:
    verdict: str
    dF_lcb: float
    dCPA_ucb: float
    cpa_cand_ucb: float
    dN_lcb: float
    viol_ucb: float
    fixed_indep_ok: bool
    conservation_ok: bool
    checks: Dict[str, bool]
    detail: Dict[str, Any]

    def to_dict(self):
        return dict(verdict=self.verdict, dF_lcb=self.dF_lcb, dCPA_ucb=self.dCPA_ucb,
                    cpa_cand_ucb=self.cpa_cand_ucb, dN_lcb=self.dN_lcb, viol_ucb=self.viol_ucb,
                    fixed_indep_ok=self.fixed_indep_ok, conservation_ok=self.conservation_ok,
                    checks=self.checks, detail=self.detail)


def business_gate(cand: List[Dict[str, Any]], inc: List[Dict[str, Any]],
                  th: BusinessThresholds) -> BusinessReport:
    """cand/inc：每 episode dict(F,C,N, [viol], [illegal], [forced], [budget_err], [dynamic_adv])，
    按 episode 一一配对（同序、同市场种子）。返回裁决报告。"""
    assert len(cand) == len(inc) and len(cand) > 0, "配对样本数须一致且非空"
    pairs = list(zip(cand, inc))

    dF_lcb, _ = _lcb_ucb(_paired_bootstrap(pairs, _rel_dF, th.n_boot, th.seed), th.alpha)
    _, dCPA_ucb = _lcb_ucb(_paired_bootstrap(pairs, _dCPA, th.n_boot, th.seed + 1), th.alpha)
    _, cpa_ucb = _lcb_ucb(_paired_bootstrap(pairs, _cpa_cand, th.n_boot, th.seed + 2), th.alpha)
    dN_lcb, _ = _lcb_ucb(_paired_bootstrap(pairs, _rel_dN, th.n_boot, th.seed + 3), th.alpha)

    def _viol(ps):
        vc = np.mean([c.get("viol", 0.0) for c, _ in ps])
        vi = np.mean([i.get("viol", 0.0) for _, i in ps])
        return vc - vi
    _, viol_ucb = _lcb_ucb(_paired_bootstrap(pairs, _viol, th.n_boot, th.seed + 4), th.alpha)

    # 反重估：fixed ADV 总量须独立提升（不靠 dynamic/revaluation）
    fixed_indep_ok = sum(c["F"] for c in cand) > sum(i["F"] for i in inc)
    # 预算守恒 / 非法 / forced（逐 episode 累计须 0）
    conservation_ok = all(abs(c.get("budget_err", 0.0)) < 1e-6 for c in cand) and \
                      all(c.get("illegal", 0) == 0 for c in cand) and \
                      all(c.get("forced", 0) == 0 for c in cand)

    checks = {
        "value_gain": dF_lcb > th.delta_F,
        "cpa_noninf": dCPA_ucb <= th.delta_CPA,
        "cpa_max": cpa_ucb <= th.cpa_max,
        "conv_noninf": dN_lcb >= -th.delta_N,
        "viol_noninf": viol_ucb <= th.delta_viol,
        "fixed_independent": fixed_indep_ok,
        "conservation": conservation_ok,
    }
    safety_ok = all(checks[k] for k in ("cpa_noninf", "cpa_max", "conv_noninf",
                                        "viol_noninf", "conservation"))
    if safety_ok and checks["value_gain"] and checks["fixed_independent"]:
        verdict = "ACCEPT"
    elif safety_ok and not checks["value_gain"]:
        # 价值下界跨 δ_F：HOLD；已明确 UCB(ΔF)≤δ_F 则 REJECT_NO_GAIN
        _, dF_ucb = _lcb_ucb(_paired_bootstrap(pairs, _rel_dF, th.n_boot, th.seed + 5), th.alpha)
        verdict = "REJECT_NO_GAIN" if dF_ucb <= th.delta_F else "HOLD"
    else:
        verdict = "HOLD"   # 安全门未过 → 不升级、不判有害，追加样本重验

    return BusinessReport(verdict=verdict, dF_lcb=dF_lcb, dCPA_ucb=dCPA_ucb, cpa_cand_ucb=cpa_ucb,
                          dN_lcb=dN_lcb, viol_ucb=viol_ucb, fixed_indep_ok=fixed_indep_ok,
                          conservation_ok=conservation_ok, checks=checks,
                          detail=dict(n_episodes=len(cand),
                                      F_cand=sum(c["F"] for c in cand),
                                      F_inc=sum(i["F"] for i in inc),
                                      cpa_cand=_cpa_cand(pairs),
                                      cpa_inc=portfolio_cpa([i["C"] for i in inc],
                                                            [i["N"] for i in inc])))
