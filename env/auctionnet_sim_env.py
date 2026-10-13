"""AuctionNet auction simulator with fixed bidding strategies and tCPA control.

AuctionNet generates opportunities and settles GSP auctions. Each controlled
advertiser bids through an attached AuctionNet strategy; the per-step action may
change its target CPA, while evaluation code may reallocate unit budget caps.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

# AuctionNet is cloned by the user (see README). Both layouts are accepted:
# `git clone ...` leaves the sources in AuctionNet/, the GitHub zip unpacks to
# AuctionNet/AuctionNet-main/. ADCOEVO_AUCTIONNET overrides either.
from train.paths import auctionnet_dir, _auctionnet_unavailable_message  # noqa: E402

_AN = auctionnet_dir()
for p in (str(_AN), str(_AN / "strategy_train_env")):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from simul_bidding_env.PvGenerator.NeurIPSPvGen import NeurIPSPvGen      # noqa: E402
    from simul_bidding_env.Environment.BiddingEnv import BiddingEnv          # noqa: E402
    from simul_bidding_env.strategy.pid_bidding_strategy import PidBiddingStrategy  # noqa: E402
except ImportError as exc:
    raise ImportError(
        f"cannot import the AuctionNet simulator ({exc}).\n"
        + _auctionnet_unavailable_message(_AN)
    ) from exc

from .types import State, StepResult                                    # noqa: E402
from .auctionnet_profile import AuctionNetProfile, load_profile          # noqa: E402
from .tcpa_levels import is_allowed_tcpa_level                      # noqa: E402

EPS = 1e-9
_DEFAULT_CONFIG = _REPO / "configs" / "auctionnet_sim.yaml"
_SPARSE_SEED_SALT = 0x53504152


def thin_conversions(conversions: np.ndarray, keep_rate: float, seed: int,
                     episode: int, time_step: int) -> np.ndarray:
    """Deterministically thin mature conversions without changing their shape."""
    rate = float(keep_rate)
    if not 0.0 < rate <= 1.0:
        raise ValueError("conversion_keep_rate must be in (0, 1]")
    if rate == 1.0:
        return conversions
    rng = np.random.default_rng(np.random.SeedSequence(
        [int(seed), int(episode), int(time_step), _SPARSE_SEED_SALT]))
    return conversions * (rng.random(conversions.shape) < rate)

class ProfiledNeurIPSPvGen(NeurIPSPvGen):
    """AuctionNet generator with the profile's pValue base set before generation."""
    def __init__(self, profile: AuctionNetProfile, **kwargs):
        self._profile_pvalue_mean_base = float(profile.pvalue_mean_base)
        super().__init__(**kwargs)

    def generate_pvalue_mean(self) -> np.ndarray:
        original = self.pvalue_mean_base
        self.pvalue_mean_base = self._profile_pvalue_mean_base
        try:
            return super().generate_pvalue_mean()
        finally:
            self.pvalue_mean_base = original


def _fresh_tcpa_state(base_cpa: float) -> dict:
    """Initialize tCPA controls and execution records for one unit and period.

    The control target starts at base_cpa with multiplier 1.0 and no adjustments.
    """
    return {"base_cpa": base_cpa, "level": 1.0, "n_adjust": 0, "forced_hold": 0,
            "cpa_trace": [], "control_tcpa": base_cpa,
            "requested": "HOLD", "canonical_requested": "HOLD", "applied": "HOLD",
            "semantic_rewrite": False, "forced_hold_step": False, "illegal": False,
            "quota_consumed": 0, "policy_generated": False}


def determine_quadrant(cost_24h: float, cpr_24h: float, roi_24h: float,
                       tcpr: float, troi: float, head_threshold: float) -> str:
    """Return the head/CPA/ROI observation code used by existing diagnostics."""
    h = int(cost_24h > head_threshold)
    c = int(cpr_24h <= tcpr)
    r = int(roi_24h >= troi)
    return f"Q-{h}{c}{r}"


def _build_native_agents(n_steps: int, ep: int):
    """忠实复刻 AuctionNet `Controller.initialize_agents` 的 48 广告主异质群体（每类8个）。
    偶数类目：[PID, IQL, TD3_BC, OnlineLP(ep=cat), OnlineLP(ep=cat+1), CQL, BC, MbrlMopo]
    奇数类目：[PID, BCQ, MbrlMopo, OnlineLP(ep=cat), OnlineLP(ep=cat+1), TD3_BC, IQL, MbrlComboMicro]
    权重按官方 checkpoint 加载（各策略 __init__ 内 realpath 解析）。返回 48 个策略实例。"""
    import numpy as _np
    S = "simul_bidding_env.strategy"
    def _cls(mod, name):
        return getattr(__import__(f"{S}.{mod}", fromlist=[name]), name)
    Pid = _cls("pid_bidding_strategy", "PidBiddingStrategy")
    Iql = _cls("iql_bidding_strategy", "IqlBiddingStrategy")
    Td3 = _cls("td3_bc_bidding_strategy", "TD3_BCBiddingStrategy")
    Olp = _cls("onlinelp_bidding_strategy", "OnlineLpBiddingStrategy")
    Cql = _cls("cql_bidding_strategy", "CqlBiddingStrategy")
    Bc = _cls("bc_bidding_strategy", "BcBiddingStrategy")
    Bcq = _cls("bcq_bidding_strategy", "BcqBiddingStrategy")
    Mopo = _cls("mbrl_mopo_bidding_strategy", "MbrlMopoBiddingStrategy")
    Combo = _cls("mbrl_combomicro_bidding_strategy", "MbrlComboMicroBiddingStrategy")
    agents = []
    for cat in range(6):                       # 6 类目 × 8 = 48
        lo = min(cat, 6); hi = min(cat + 1, 6)  # onlinelp 仅有 episode 0-6
        if cat % 2 == 0:
            block = [Pid(exp_tempral_ratio=_np.ones(n_steps)), Iql(), Td3(),
                     Olp(episode=lo), Olp(episode=hi), Cql(), Bc(), Mopo()]
        else:
            block = [Pid(exp_tempral_ratio=_np.ones(n_steps)), Bcq(), Mopo(),
                     Olp(episode=lo), Olp(episode=hi), Td3(), Iql(), Combo()]
        agents.extend(block)
    return agents




# Auction helpers adapted from AuctionNet run/run_test.py (Apache-2.0; see NOTICE).
def _get_winner(slot_pit: np.ndarray) -> np.ndarray:
    slot_pit = slot_pit.T
    num_pv, _ = slot_pit.shape
    winner = np.full((num_pv, 3), -1, dtype=int)
    for pos in range(1, 4):
        idx = np.argwhere(slot_pit == pos)
        if idx.size > 0:
            pv_i, ag_i = idx.T
            winner[pv_i, pos - 1] = ag_i
    return winner


def _adjust_over_cost(bids, over_cost_ratio, slot_coefficients, winner_pit):
    import math
    for agent_index in np.where(over_cost_ratio > 0)[0]:
        for i, _ in enumerate(slot_coefficients):
            pv_indices = np.where(winner_pit[:, i] == agent_index)[0]
            rng = np.random.default_rng(seed=1)
            num_to_drop = math.ceil(pv_indices.size * over_cost_ratio[agent_index])
            if num_to_drop > 0:
                dropped = rng.choice(pv_indices, num_to_drop, replace=False)
                bids[dropped, agent_index] = 0


class AuctionNetSimEnv:
    def __init__(self, config: Optional[Dict[str, Any]] = None,
                 config_path: Optional[Path] = None):
        if config is None:
            path = Path(config_path) if config_path else _DEFAULT_CONFIG
            with open(path, "r", encoding="utf-8") as f:
                config = (yaml.safe_load(f) or {}).get("auctionnet_sim_env", {})
        allowed_config = {"start_episode", "n_accounts", "units_per_account", "n_steps",
                          "pv_num", "competitor_mode", "tcpa_max_adjust", "seed",
                          "profile", "conversion_keep_rate", "our_indices"}
        unknown_config = set(config) - allowed_config
        if unknown_config:
            raise ValueError(f"unsupported AuctionNetSimEnv config fields: {sorted(unknown_config)}")
        self.cfg = config
        self.start_episode = int(config.get("start_episode", 0))
        self.n_accounts = int(config.get("n_accounts", 3))
        self.units_per_account = int(config.get("units_per_account", 4))
        self.n_steps = int(config.get("n_steps", 48))
        self.pv_num = int(config.get("pv_num", 500000))
        # Native uses AuctionNet's heterogeneous competitors; pid uses only PID.
        self._competitor_mode = str(config.get("competitor_mode", "native"))
        if self._competitor_mode not in ("pid", "native"):
            raise ValueError("competitor_mode must be pid or native")
        self._TCPA_K = int(config.get("tcpa_max_adjust", 8))
        self.seed = int(config.get("seed", 20270001))
        self.profile = load_profile(str(config.get("profile", "sparse")))
        configured_keep_rate = config.get("conversion_keep_rate")
        self.conversion_keep_rate = (self.profile.conversion_keep_rate if configured_keep_rate is None
                                     else float(configured_keep_rate))
        if not 0.0 < self.conversion_keep_rate <= 1.0:
            raise ValueError("conversion_keep_rate must be in (0, 1]")
        self._budgets = self.profile.budgets
        self._cpas = self.profile.cpa_constraints

        self._pi = 0
        self._t = 0
        self._agents: List[PidBiddingStrategy] = []
        self._env: Optional[BiddingEnv] = None
        self._pvgen: Optional[NeurIPSPvGen] = None
        self._units: Dict[str, Dict[str, Any]] = {}        # uid -> 运行态
        self._accounts: Dict[str, Dict[str, Any]] = {}
        self._uid2idx: Dict[str, int] = {}
        self._our_idx: set = set()
        self._unit_strategies: Dict[str, Any] = {}
        self._unit_hist: Dict[str, Dict[str, list]] = {}
        self._tcpa_state: Dict[str, Dict[str, Any]] = {}
        self._head_threshold = 0.0

    # ---------------- 周期级 ----------------
    def reset(self) -> State:
        self._pi = self.start_episode
        self._unit_strategies = {}
        self._unit_hist = {}
        self._tcpa_state = {}
        self._build_population()
        self._begin_period()
        return self.observe()

    def _build_population(self) -> None:
        n_units = self.n_accounts * self.units_per_account
        # 默认按预算选 top-N 为我方单元；配置 our_indices 可显式指定广告主下标
        # （headroom/单元独立市场实验用：把任意异质单元置为"我方"、其余 47 个作 PID 竞争者）。
        oi = self.cfg.get("our_indices")
        if oi:
            order = np.array([int(x) for x in oi])[:n_units]
        else:
            order = np.argsort(-np.array(self._budgets))[:n_units]    # 按预算选 top-N 为我方单元
        self._our_idx = set(int(i) for i in order)
        self._head_threshold = float(np.median([self._budgets[i] for i in order]))

        self._units, self._accounts, self._uid2idx = {}, {}, {}
        for a in range(self.n_accounts):
            acc_id = f"acct_{a + 1:02d}"
            uids = []
            for j in range(self.units_per_account):
                idx = int(order[a * self.units_per_account + j])
                uid = f"adg_{idx:04d}"
                self._uid2idx[uid] = idx
                self._units[uid] = dict(adgroup_id=uid, idx=idx, account_id=acc_id,
                                        category=idx // 8, budget=float(self._budgets[idx]),
                                        base_budget=float(self._budgets[idx]),
                                        # desired_cpa=固定业务目标 CPAConstraint（d_a，永不被 tCPA 动作改）；
                                        # cpa/tcpr 是评测与观测使用的别名。
                                        desired_cpa=float(self._cpas[idx]),
                                        cpa=float(self._cpas[idx]), tcpr=float(self._cpas[idx]), troi=1.0)
                uids.append(uid)
            self._accounts[acc_id] = dict(account_id=acc_id, unit_ids=uids,
                                          budget=sum(self._budgets[self._uid2idx[u]] for u in uids))

    def _begin_period(self) -> None:
        self._t = 0
        ep = self._pi
        self._pvgen = ProfiledNeurIPSPvGen(self.profile, episode=ep, num_tick=self.n_steps,
                                           num_agent=48, num_agent_category=8,
                                           num_category=6, pv_num=self.pv_num)
        # 周期流量曲线（每步机会数）——择时特征：结构跨 episode 稳定可预报(corr~0.81)
        self._vol_curve = np.array([self._pvgen.pv_values[s].shape[0] for s in range(self.n_steps)],
                                   dtype=float)
        self._env = BiddingEnv()
        self._env.reset(episode=ep)
        if self._competitor_mode == "native":
            # 忠实 AuctionNet：竞争者用官方异质 mix（每类8个：PID/IQL/TD3_BC/OnlineLP×2/CQL/BC/Mbrl…），
            # 非全 PID。我方单元槽位也建原生实例，供 over_cost 预算记账。
            self._agents = _build_native_agents(self.n_steps, ep)
            for i, ag in enumerate(self._agents):
                ag.budget = float(self._budgets[i]); ag.cpa = float(self._cpas[i])
                if hasattr(ag, "reset"):
                    ag.reset()
            self._agent_hist = {i: {"pv": [], "bid": [], "auc": [], "imp": [], "lwc": []}
                                for i in range(48) if i not in self._our_idx}
        else:
            self._agents = []
            for i in range(48):
                ag = PidBiddingStrategy(budget=float(self._budgets[i]), cpa=float(self._cpas[i]),
                                        category=i // 8, exp_tempral_ratio=np.ones(self.n_steps))
                ag.reset()
                self._agents.append(ag)
        for u in self._units.values():
            u["cum_cost"] = 0.0; u["cum_conv"] = 0.0; u["cum_adv"] = 0.0
            # Track raw value, fixed ADV, control-target ADV and their difference.
            u["cum_native_value"] = 0.0; u["cum_fixed_adv"] = 0.0
            u["cum_dynamic_adv"] = 0.0; u["cum_revaluation"] = 0.0
            u["cpr_24h"] = 0.0; u["roi_24h"] = 0.0
            u["pcvr_now"] = 0.0; u["lwc_now"] = 0.0; u["bid_now"] = 0.0
            u["cost_ref"] = 0.0
        for acc in self._accounts.values():
            acc["acc_cost"] = 0.0

    def set_our_strategies(self, factory) -> None:
        """为每个我方单元装一个 AuctionNet 原生出价算法实例。factory(unit_dict)->strategy。
        必须在 reset() 之后调用（此时 _units/_begin_period 已建好）。会立即建历史缓冲。"""
        if not self._units:
            raise RuntimeError("reset() must be called before set_our_strategies()")
        self._unit_strategies = {}
        for uid in [f"adg_{i:04d}" for i in sorted(self._our_idx)]:
            self._unit_strategies[uid] = factory(self._units[uid])
        self._unit_hist = {uid: {"pv": [], "bid": [], "auc": [], "imp": [], "lwc": []}
                           for uid in self._unit_strategies}
        # Start each attached strategy at its native CPA and a fresh action quota.
        self._tcpa_state = {}
        for uid, strat in self._unit_strategies.items():
            base = float(self._units[uid].get("cpa", getattr(strat, "cpa", 0.0)))
            if base > 0:
                strat.cpa = base
            self._tcpa_state[uid] = _fresh_tcpa_state(base)

    # ---------------- 步级（真拍卖） ----------------
    def step(self, actions: Dict[str, Dict[str, Any]]) -> StepResult:
        if self._env is None or self._pvgen is None:
            raise RuntimeError("reset() must be called before step()")
        if set(self._unit_strategies) != set(self._units):
            raise RuntimeError("set_our_strategies() must attach every controlled unit before step()")
        for uid, action in actions.items():
            if uid not in self._units or action is None:
                continue
            unknown_action = set(action) - {"tcpa_level"}
            if unknown_action:
                raise ValueError(f"unsupported AuctionNetSimEnv action fields: {sorted(unknown_action)}")
        t = self._t
        pv = self._pvgen.pv_values[t]            # [npv × 48]
        sig = self._pvgen.pValueSigmas[t]
        npv = pv.shape[0]
        env = self._env

        # 1) Controlled bidders use their attached AuctionNet strategies and
        #    current unit budget caps. Competitors use PID or the native mix.
        bids = np.zeros((npv, 48))
        our_bidder_index = {}
        for i, ag in enumerate(self._agents):
            if i in self._our_idx:
                uid = f"adg_{i:04d}"
                u = self._units[uid]
                act = actions.get(uid) or {}
                strat = self._unit_strategies[uid]
                ts = self._tcpa_state[uid]
                lvl_raw = act.get("tcpa_level", "HOLD")
                requested = "HOLD" if lvl_raw in (None, "HOLD") else str(lvl_raw)
                semantic_rewrite = False; forced_hold_step = False; quota_consumed = 0
                illegal = False
                if requested == "HOLD":
                    applied = "HOLD"
                else:
                    try:
                        new_lvl = float(requested)
                    except (TypeError, ValueError):
                        new_lvl = None; illegal = True
                    if new_lvl is None or not is_allowed_tcpa_level(requested):
                        illegal = True
                        applied = "HOLD"
                    elif new_lvl == ts["level"]:
                        semantic_rewrite = True; applied = "HOLD"
                    elif ts["n_adjust"] >= self._TCPA_K:
                        forced_hold_step = True; applied = "HOLD"
                        ts["forced_hold"] += 1
                    else:
                        ts["level"] = new_lvl; ts["n_adjust"] += 1; quota_consumed = 1
                        if ts["base_cpa"] > 0:
                            strat.cpa = ts["base_cpa"] * new_lvl
                        applied = f"SET({new_lvl})"
                ts["requested"] = requested
                ts["canonical_requested"] = "HOLD" if (requested == "HOLD" or semantic_rewrite or illegal) else applied
                ts["applied"] = applied
                ts["semantic_rewrite"] = semantic_rewrite
                ts["forced_hold_step"] = forced_hold_step
                ts["illegal"] = illegal
                ts["quota_consumed"] = quota_consumed
                ts["control_tcpa"] = ts["base_cpa"] * ts["level"]
                ts["policy_generated"] = ("tcpa_level" in act)
                ts["cpa_trace"].append(round(getattr(strat, "cpa", 0.0), 4))
                rem = max(u["budget"] - u["cum_cost"], env.min_remaining_budget)
                strat.budget = u["budget"]; strat.remaining_budget = rem
                ag.budget = u["budget"]; ag.remaining_budget = rem
                h = self._unit_hist[uid]
                b = strat.bidding(t, pv[:, i], sig[:, i], h["pv"], h["bid"],
                                  h["auc"], h["imp"], h["lwc"])
                bids[:, i] = np.maximum(np.asarray(b, dtype=float), 0.0)
                u["bid_now"] = float(np.mean(bids[:, i]))
                our_bidder_index[uid] = i
            else:
                if ag.remaining_budget < env.min_remaining_budget:
                    continue
                # 竞争者出价：AuctionNet 原生机制，不做任何出清价标定（见 memory: no-clearing-price-calibration）
                if self._competitor_mode == "native":
                    h = self._agent_hist[i]   # 忠实：喂完整历史(RL/LP 竞争者需要状态)
                    bids[:, i] = ag.bidding(t, pv[:, i], sig[:, i], h["pv"], h["bid"],
                                            h["auc"], h["imp"], h["lwc"])
                else:
                    bids[:, i] = ag.bidding(t, pv[:, i], sig[:, i], [], [], [], [], [])
        bids[bids < 0] = 0

        # 2) 真拍卖 + over_cost 卡控（移植 run_test）
        remaining = np.array([ag.remaining_budget for ag in self._agents])
        ratio_max, cost = None, np.zeros(48)
        winner_pit = None
        while ratio_max is None or ratio_max > 0:
            if ratio_max and ratio_max > 0:
                ocr = np.maximum((cost - remaining) / (cost + 1e-4), 0)
                _adjust_over_cost(bids, ocr, env.slot_coefficients, winner_pit)
            xi, slot, cost_pit, exposed, conv, lwc, _mp = env.simulate_ad_bidding(pv, sig, bids)
            conv = thin_conversions(conv, self.conversion_keep_rate, self.seed, self._pi, t)
            real_cost = cost_pit * exposed                     # [48 × npv]
            cost = real_cost.sum(axis=1)                       # [48]
            winner_pit = _get_winner(slot)
            ocr = np.maximum((cost - remaining) / (cost + 1e-4), 0)
            ratio_max = ocr.max()

        conv_per = conv.sum(axis=1)                            # [48]
        # 扣预算：竞争者按自身 remaining 递减；我方单元用 cum_cost 跟踪（每步开头按 Agent 预算重设 remaining）
        for i, ag in enumerate(self._agents):
            if i not in self._our_idx:
                ag.remaining_budget -= float(cost[i])

        # 3) 我方单元 outcome（反事实结算）
        outcomes: Dict[str, Dict[str, Any]] = {}
        agg = dict(cost=0.0, conversions=0.0, adv=0.0)
        for uid, u in self._units.items():
            i = u["idx"]
            c = float(cost[i]); cv = float(conv_per[i])
            value = float((pv[:, i] * exposed[i]).sum())       # native Value = Σ_曝光 pCVR（不依赖 tCPA）
            desired = float(u["desired_cpa"])                  # d_a，固定业务目标
            ts = self._tcpa_state[uid]
            control = float(ts["control_tcpa"])        # τ = d_a·m
            # Control-target ADV equals fixed ADV plus the revaluation difference.
            fixed_adv = value * desired
            dynamic_adv = value * control
            revaluation = dynamic_adv - fixed_adv
            adv = fixed_adv
            u["cum_cost"] += c; u["cum_conv"] += cv; u["cum_adv"] += adv
            u["cum_native_value"] += value; u["cum_fixed_adv"] += fixed_adv
            u["cum_dynamic_adv"] += dynamic_adv; u["cum_revaluation"] += revaluation
            u["cost_ref"] = u["cum_cost"]
            u["pcvr_now"] = float(pv[:, i].mean())
            u["lwc_now"] = float(lwc.mean())
            u["cpr_24h"] = round(u["cum_cost"] / u["cum_conv"], 4) if u["cum_conv"] > 0 else 0.0
            u["roi_24h"] = round(u["cum_adv"] / (u["cum_cost"] + EPS), 4) if u["cum_cost"] > 0 else 0.0
            o = dict(cost=round(c, 4), conversions=round(cv, 4), adv=round(adv, 4),
                     cpr=round(c / (cv + EPS), 4) if cv > 0 else 0.0,
                     roi=round(adv / (c + EPS), 4) if c > 0 else 0.0,
                     budget_completion=round(min(u["cum_cost"] / (u["budget"] + EPS), 1.5), 4),
                     cpr_violation=round(max(c / (cv + EPS) - u["tcpr"], 0), 4) if cv > 0 else 0.0,
                     roi_violation=round(max(u["troi"] - adv / (c + EPS), 0), 4) if c > 0 else 0.0,
                     op_ratio_violation=0.0, invalid_action=0)
            # Keep initial and controlled tCPA separate in settled unit outcomes.
            o["native_value"] = round(value, 8)
            o["fixed_adv"] = round(fixed_adv, 8)
            o["dynamic_adv"] = round(dynamic_adv, 8)
            o["revaluation"] = round(revaluation, 8)
            o["desired_cpa"] = round(desired, 6)
            o["control_tcpa"] = round(control, 6)
            o["tcpa_multiplier"] = round(float(ts["level"]), 6)
            remaining_k = max(self._TCPA_K - ts["n_adjust"], 0)
            deterministic = (remaining_k == 0) and (not ts.get("policy_generated", False))
            o["requested_action"] = ts.get("requested", "HOLD")
            o["canonical_requested_action"] = ts.get("canonical_requested", "HOLD")
            o["applied_action"] = ts.get("applied", "HOLD")
            o["quota_consumed"] = int(ts.get("quota_consumed", 0))
            o["forced_hold"] = bool(ts.get("forced_hold_step", False))
            o["semantic_rewrite"] = bool(ts.get("semantic_rewrite", False))
            o["remaining_k"] = remaining_k
            o["pg_eligible"] = bool(
                ts.get("policy_generated", False)
                and not ts.get("illegal", False)
                and o["canonical_requested_action"] == o["applied_action"]
                and not o["forced_hold"]
                and not o["semantic_rewrite"]
                and not deterministic)
            outcomes[uid] = o
            agg["cost"] += c; agg["conversions"] += cv; agg["adv"] += adv
            acc = self._accounts[u["account_id"]]
            acc["acc_cost"] += c

        # AuctionNet strategy history for the next bid decision.
        real_cost = cost_pit * exposed
        for uid, i in our_bidder_index.items():
            h = self._unit_hist[uid]
            h["pv"].append(np.column_stack([pv[:, i], sig[:, i]]))
            h["bid"].append(bids[:, i].copy())
            h["lwc"].append(np.asarray(lwc).copy())
            h["auc"].append(np.column_stack([exposed[i], exposed[i], real_cost[i]]))
            h["imp"].append(np.column_stack([conv[i], conv[i]]))

        # native 竞争者：累计每个竞争者的历史（同口径），供其 RL/LP 策略下一步构造状态。
        if self._competitor_mode == "native":
            rc = cost_pit * exposed
            for i, h in self._agent_hist.items():
                h["pv"].append(np.column_stack([pv[:, i], sig[:, i]]))
                h["bid"].append(bids[:, i].copy())
                h["lwc"].append(np.asarray(lwc).copy())
                h["auc"].append(np.column_stack([exposed[i], exposed[i], rc[i]]))
                h["imp"].append(np.column_stack([conv[i], conv[i]]))

        self._t += 1
        done = self._t >= self.n_steps
        agg["cpr"] = round(agg["cost"] / (agg["conversions"] + EPS), 4)
        agg["roi"] = round(agg["adv"] / (agg["cost"] + EPS), 4)
        return StepResult(outcome=outcomes, next_state=self.observe(),
                          done=done, info={"aggregated": agg, "time_step": t, "episode": self._pi})

    # ---------------- 观测 ----------------
    def observe(self) -> State:
        t = min(self._t, self.n_steps - 1)
        progress = round(self._t / self.n_steps, 3)
        units_all = list(self._units.values())
        m_cost = sum(u["cum_cost"] for u in units_all)
        m_conv = sum(u["cum_conv"] for u in units_all)
        m_adv = sum(u["cum_adv"] for u in units_all)
        mkt_cpr = round(m_cost / (m_conv + EPS), 4) if m_conv > 0 else 0.0
        mkt_roi = round(m_adv / (m_cost + EPS), 4) if m_cost > 0 else 1.0
        lwc_vals = [u.get("lwc_now", 0.0) for u in units_all if u.get("lwc_now", 0.0) > 0] or [0.0]
        # 择时特征（周期内广告机会时序）：当步相对流量 + 峰位/距峰（Optimizer 反应+短前瞻, Planner pacing 预报共用）
        vc = getattr(self, "_vol_curve", None)
        if vc is not None and len(vc) > 0:
            vmean = float(vc.mean()) or 1.0
            peak = int(vc.argmax())
            step_vol_rel = round(float(vc[t]) / vmean, 3)                 # 当步机会量/周期均值（>1=高流量窗口）
            peak_vol_rel = round(float(vc[peak]) / vmean, 3)
            steps_to_peak = int(peak - t)                                # >0 峰在后（宜预留/克制），<0 已过峰
            vol_early = round(float(vc[:self.n_steps // 3].mean()) / vmean, 3)
            vol_mid = round(float(vc[self.n_steps // 3:2 * self.n_steps // 3].mean()) / vmean, 3)
            vol_late = round(float(vc[2 * self.n_steps // 3:].mean()) / vmean, 3)
        else:
            step_vol_rel = peak_vol_rel = vol_early = vol_mid = vol_late = 1.0; peak = steps_to_peak = 0
        market = {"period_index": self._pi, "time_step": t, "progress": progress,
                  "environment_profile": self.profile.name,
                  "calibration_id": self.profile.calibration_id,
                  "calibration_hash": self.profile.calibration_hash,
                  "market_cpr": mkt_cpr, "market_roi": mkt_roi,
                  "market_24h_cpr": mkt_cpr, "market_24h_roi": mkt_roi,
                  "avg_least_winning_cost": round(sum(lwc_vals) / len(lwc_vals), 6),
                  "pacing_target": progress, "is_promotion": False,
                  "step_volume_rel": step_vol_rel, "vol_peak_step": peak, "steps_to_peak": steps_to_peak,
                  "vol_peak_rel": peak_vol_rel, "vol_early": vol_early, "vol_mid": vol_mid, "vol_late": vol_late}
        accounts = {}
        for aid, acc in self._accounts.items():
            uids = acc["unit_ids"]
            cpr_l = [self._units[u]["cpr_24h"] for u in uids] or [0.0]
            roi_l = [self._units[u]["roi_24h"] for u in uids] or [0.0]
            consumed = acc.get("acc_cost", 0.0)
            head = sum(1 for u in uids if self._units[u]["cost_ref"] > self._head_threshold) / max(len(uids), 1)
            accounts[aid] = {"account_id": aid, "budget": round(acc["budget"], 2),
                             "remaining_budget": round(acc["budget"] - consumed, 2),
                             "budget_consumption_rate": round(min(consumed / (acc["budget"] + EPS), 1.0), 4),
                             "pacing_gap": round(min(consumed / (acc["budget"] + EPS), 1.0) - progress, 4),
                             "account_24h_cpr": round(sum(cpr_l) / len(cpr_l), 4),
                             "account_24h_roi": round(sum(roi_l) / len(roi_l), 4),
                             "num_ads": len(uids), "head_ad_share": round(head, 4)}
        units = {}
        for uid, u in self._units.items():
            cost_24h = u["cum_cost"]
            quad = determine_quadrant(cost_24h, u["cpr_24h"] or u["tcpr"], u["roi_24h"],
                                      u["tcpr"], u["troi"], self._head_threshold)
            units[uid] = {"adgroup_id": uid, "category": u["category"],
                          "cost_24h": round(cost_24h, 2), "cpr_24h": u["cpr_24h"], "roi_24h": u["roi_24h"],
                          "tcpr": round(u["tcpr"], 4), "troi": u["troi"],
                          "is_head_ad": cost_24h > self._head_threshold, "quadrant": quad,
                          "recent_bid": round(u.get("bid_now", 0.0), 6),
                          "pvalue_mean": round(u.get("pcvr_now", 0.0), 6), "pvalue_sigma": 0.0,
                          "remaining_budget_unit": round(u["budget"] - u["cum_cost"], 2)}
            # Include tCPA control state and remaining adjustments in observations.
            ts = self._tcpa_state.get(uid)
            if ts is not None:
                units[uid].update({
                    "desired_cpa": round(float(u["desired_cpa"]), 4),
                    "tcpa_multiplier": round(float(ts["level"]), 4),
                    "control_tcpa": round(float(ts["control_tcpa"]), 4),
                    "remaining_k": max(self._TCPA_K - ts["n_adjust"], 0),
                    "used_k": ts["n_adjust"], "adjustment_limit": self._TCPA_K})
        return {"market": market, "accounts": accounts, "units": units}

    # ---------------- 元信息 ----------------
    @property
    def period_index(self) -> int:
        return self._pi

    @property
    def time_step(self) -> int:
        return self._t

    def account_units(self, account_id: str) -> List[str]:
        return list(self._accounts[account_id]["unit_ids"])
