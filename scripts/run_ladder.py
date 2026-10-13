#!/usr/bin/env python3
"""Evaluate fixed advertising policies with shared role prompts and action handling.

Arms: none (No Operation), budget (Planner), tcpa (Optimizer), both (P+O),
and full (P+O+I). Planner reallocates an account's remaining budget after
observation. Optimizer controls each unit's tCPA at subsequent time steps.
Inspector summarizes settled results and guides the next period.
"""
import argparse
import json
import math
import os
import sys

sys.path.insert(0, ".")
import numpy as np  # noqa: E402
from env.auctionnet_sim_env import AuctionNetSimEnv  # noqa: E402
from env.tcpa_levels import TCPA_LEVELS  # noqa: E402
from simul_bidding_env.strategy.abid_bidding_strategy import AbidBiddingStrategy  # noqa: E402

# 同类目 PORT（3 账户各同类目 4 单元，账户内互为近竞争者、预算接近）
PORT = [10, 12, 13, 14, 17, 18, 19, 23, 33, 34, 36, 37]
TCPA_ENUM = list(TCPA_LEVELS)
UPA = 4          # units per account（同类目单元数）
CPA_TOL = 1.02

# ---------------- LLM 角色 prompt ----------------
PLANNER_SYS = (
    "You are the Planner for one advertising account with four competing ad units. "
    "Reallocate the account's remaining budget among its units while preserving its total budget. "
    "Output one proportion of the remaining budget for each unit; proportions must be "
    "nonnegative and sum to 1. Budget never moves between accounts. "
    "Prioritize marginal budget absorption, not average efficiency alone. Compare realized spending "
    "during the observation window with the elapsed share of the period. Units spending ahead of "
    "pace with high fixed ADV per cost can use additional budget; units behind pace or with low "
    "efficiency should receive less. Do not concentrate budget on a high-efficiency unit that "
    "cannot spend it. Use fixed_adv and realized unit CPA, not value inflated by a higher control tCPA. "
    "Return only a JSON object with reasoning (one sentence) and alpha (an array of four "
    "remaining-budget proportions, in the unit order shown)."
)
OPTIMIZER_SYS = (
    "You are the Optimizer for one advertising account with four ad units. At each remaining "
    "time step, decide whether to change each unit's control tCPA multiplier. A multiplier below "
    "1.0 tends to lower bids and may reduce overpayment; HOLD keeps the current level. The "
    "underlying bidder is Abid. As a heuristic, levels of 0.5 to 0.7 early and 0.8 to 1.3 "
    "later can help, but decide from the observed state rather than a fixed schedule. Units "
    "with larger budgets or higher utilization can overbid and see CPA rise; lower the "
    "affected unit's control tCPA when this happens. "
    "You are asked at every time step. HOLD does not use an adjustment, while changing level "
    "uses one of that unit's at most K adjustments per period. Remaining counts are shown below. "
    "Return only a JSON object with reasoning (one sentence) and decisions (an array of "
    "four values). In the displayed unit order, each value must be one of "
    "{HOLD,0.5,0.6,0.7,0.8,1.0,1.2,1.3}."
)
INSPECTOR_SYS = (
    "You are the Inspector. After each period settles, diagnose delivery performance and give "
    "guidance to the next period's Planner and Optimizer. Market conditions may change, so "
    "state transferable principles rather than naming specific units or numerical targets. "
    "Review account-level fixed_adv, cost, conversions, CPA, budget utilization, and the "
    "record of requested and executed decisions. Consider the stated target cost allowance. "
    "Identify systematic overpayment, budget misallocation, or excess cost. "
    "Return only a JSON object with reasoning (brief diagnosis), has_issue (a boolean), "
    "issue_summary (one general issue, or 'No issue'), planner_guidance (general budget "
    "guidance, or 'Maintain'), and optimizer_guidance (general tCPA and timing guidance, "
    "or 'Maintain'). Keep guidance general and actionable."
)
MAINTAIN_GUIDANCE = "Maintain"
NO_ISSUE_SUMMARY = "No issue"


def _is_maintain(value):
    return value is None or str(value).strip().casefold() in ("", "maintain")


def _is_no_issue(value):
    return value is None or str(value).strip().casefold() in ("", "no issue")


PLANNER_SCHEMA = {"type": "object", "properties": {
    "reasoning": {"type": "string"},
    "alpha": {"type": "array", "items": {"type": "number", "minimum": 0, "maximum": 1},
              "minItems": UPA, "maxItems": UPA}},
    "required": ["alpha"]}
OPTIMIZER_SCHEMA = {"type": "object", "properties": {
    "reasoning": {"type": "string"},
    "decisions": {"type": "array", "items": {"type": "string", "enum": TCPA_ENUM},
                  "minItems": UPA, "maxItems": UPA}},
    "required": ["decisions"]}
INSPECTOR_SCHEMA = {"type": "object", "properties": {
    "reasoning": {"type": "string"},
    "has_issue": {"type": "boolean"},
    "issue_summary": {"type": "string"},
    "planner_guidance": {"type": "string"},
    "optimizer_guidance": {"type": "string"}},
    "required": ["reasoning", "has_issue", "issue_summary", "planner_guidance",
                 "optimizer_guidance"]}


def parse_planner_alpha(root, n_units, initial_budgets):
    """Return a valid simplex action, or allocate by initial budget on failure."""
    if len(initial_budgets) != n_units:
        raise ValueError("Planner initial budgets must match the number of units")
    weights = [float(budget) for budget in initial_budgets]
    if any(not math.isfinite(weight) or weight < 0 for weight in weights) or sum(weights) <= 0:
        raise ValueError("Planner initial budgets must be finite and have positive total")
    total_budget = sum(weights)
    fallback = [weight / total_budget for weight in weights]
    values = root.get("alpha") if isinstance(root, dict) else None
    if not isinstance(values, list) or len(values) != n_units:
        return fallback, False
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or
           not math.isfinite(x) or x < 0 or x > 1 for x in values):
        return fallback, False
    total = sum(values)
    if total <= 0 or abs(total - 1.0) > 0.01:
        return fallback, False
    return [float(x) / total for x in values], True


def parse_optimizer_decisions(root, n_units):
    """Normalize an entire reply and report its legality once per model call."""
    values = root.get("decisions") if isinstance(root, dict) else None
    if not isinstance(values, list) or len(values) != n_units:
        return ["HOLD"] * n_units, False
    legal = all(isinstance(x, str) and x in TCPA_ENUM for x in values)
    return [x if isinstance(x, str) and x in TCPA_ENUM else "HOLD" for x in values], legal


def valid_inspector_reply(root):
    return (isinstance(root, dict) and
            isinstance(root.get("reasoning"), str) and
            isinstance(root.get("has_issue"), bool) and
            all(isinstance(root.get(key), str) for key in
                ("issue_summary", "planner_guidance", "optimizer_guidance")))


def inspector_memory(root, episode):
    """Carry only a complete diagnosis; invalid replies execute a neutral fallback."""
    if not valid_inspector_reply(root):
        return {"planner_guidance": MAINTAIN_GUIDANCE,
                "optimizer_guidance": MAINTAIN_GUIDANCE,
                "issue": NO_ISSUE_SUMMARY, "has_issue": False,
                "updated_after_ep": episode}
    return {"planner_guidance": root["planner_guidance"] or MAINTAIN_GUIDANCE,
            "optimizer_guidance": root["optimizer_guidance"] or MAINTAIN_GUIDANCE,
            "issue": root["issue_summary"] or NO_ISSUE_SUMMARY,
            "has_issue": root["has_issue"], "updated_after_ep": episode}


# ---------------- 环境 ----------------
def make_env(ep, pv, seed, K, budget_scale=1.0, portfolio=None,
             conversion_keep_rate=None, env_profile="sparse"):
    e = AuctionNetSimEnv(config={"n_accounts": 3, "units_per_account": 4, "n_steps": 48,
        "pv_num": pv, "our_indices": portfolio or PORT, "start_episode": ep,
        "competitor_mode": "native", "seed": seed, "tcpa_max_adjust": K,
        "profile": env_profile, "conversion_keep_rate": conversion_keep_rate})
    np.random.seed(seed)
    e.reset()
    scale_budgets(e, budget_scale)
    return e


def scale_budgets(e, budget_scale):
    scale = float(budget_scale)
    if scale <= 0:
        raise ValueError("budget_scale must be positive")
    if scale == 1.0:
        return
    for unit in e._units.values():
        unit["base_budget"] = float(unit["base_budget"]) * scale
        unit["budget"] = float(unit["budget"]) * scale
    for account in e._accounts.values():
        account["budget"] = sum(float(e._units[u]["budget"]) for u in account["unit_ids"])


def _make_strategy(u):
    return AbidBiddingStrategy(budget=float(u["budget"]), cpa=float(u["cpa"]), category=int(u["category"]))


def _accts(e):
    """有序账户 id 列表 + 每账户单元列表（按单元出现顺序稳定）。"""
    order, groups = [], {}
    for u in e._units:
        a = e._units[u]["account_id"]
        if a not in groups:
            groups[a] = []
            order.append(a)
        groups[a].append(u)
    return order, groups


def unit_metrics(e, order, groups, K):
    """Return each unit's visible delivery metrics and remaining adjustments."""
    U = {}
    for a in order:
        for u in groups[a]:
            dd = e._units[u]
            cost = float(dd["cum_cost"]); conv = float(dd["cum_conv"]); fix = float(dd["cum_fixed_adv"])
            budget = float(dd["budget"])
            ts = getattr(e, "_tcpa_state", {}).get(u)
            radj = int(ts["n_adjust"]) if ts is not None else 0
            U[u] = dict(account=a, budget=budget, cost=cost, conv=conv, fix=fix,
                        util=cost / budget if budget > 0 else 0.0,
                        eff=fix / cost if cost > 0 else 0.0,
                        cpa=cost / conv if conv > 0 else float("nan"),
                        desired_cpa=float(dd["cpa"]),
                        remaining_budget=budget - cost,
                        remaining_k=max(K - radj, 0))
    return U


def account_snapshot(e, order, groups):
    """Aggregate settled delivery metrics for each account."""
    snap = {}
    for a in order:
        cost = conv = fix = budget = 0.0
        for u in groups[a]:
            dd = e._units[u]
            cost += float(dd["cum_cost"]); conv += float(dd["cum_conv"]); fix += float(dd["cum_fixed_adv"])
            budget += float(dd["budget"])
        snap[a] = dict(budget=budget, cost=cost, conv=conv, fix=fix,
                       util=cost / budget if budget > 0 else 0.0,
                       eff=fix / cost if cost > 0 else 0.0,
                       cpa=cost / conv if conv > 0 else float("nan"))
    return snap


def apply_rule_budget(e, auids, mults, base):
    """Scale initial unit budgets and normalize to the account's total budget."""
    account_budget = sum(float(base[u]) for u in auids)
    weights = [float(base[u]) * float(mult) for u, mult in zip(auids, mults)]
    total_weight = sum(weights)
    budgets = {u: account_budget * weights[i] / total_weight
               for i, u in enumerate(auids)}
    for u in auids:
        e._units[u]["budget"] = budgets[u]
    return budgets


def apply_budget_proportions(e, auids, alpha, base, max_migration_ratio=None):
    """Execute Planner's remaining-budget proportions: b = c_pre + R * alpha."""
    account_budget = sum(float(base[u]) for u in auids)
    spent = {u: float(e._units[u]["cum_cost"]) for u in auids}
    remaining = max(account_budget - sum(spent.values()), 0.0)
    target = {u: spent[u] + remaining * alpha[i] for i, u in enumerate(auids)}
    if max_migration_ratio is not None:
        moved = 0.5 * sum(abs(target[u] - float(base[u])) for u in auids) / max(account_budget, 1e-9)
        blend = min(1.0, float(max_migration_ratio) / max(moved, 1e-9))
        target = {u: float(base[u]) + blend * (target[u] - float(base[u])) for u in auids}
    for u in auids:
        e._units[u]["budget"] = target[u]
    return target


def new_action_records():
    return {"planner": {}, "optimizer": {}, "optimizer_invalid_replies": {}}


def record_planner_action(records, account, auids, alpha, budgets, valid,
                          requested_alpha=None, spent_before=None):
    executed_alpha = list(alpha)
    if spent_before is not None:
        remaining = sum(float(budgets[u]) - float(spent_before[u]) for u in auids)
        if remaining > 0:
            executed_alpha = [(float(budgets[u]) - float(spent_before[u])) / remaining
                              for u in auids]
    records["planner"][str(account)] = {
        "requested_alpha": requested_alpha if valid else None,
        "executed_alpha": executed_alpha,
        "executed_budgets": {str(u): float(budgets[u]) for u in auids},
        "invalid_reply": not valid,
    }


def record_optimizer_outcome(records, account, uid, requested, state):
    account_record = records["optimizer"].setdefault(str(account), {})
    unit_record = account_record.setdefault(str(uid), {
        "requested_set": 0, "applied_set": 0, "forced_hold": 0,
        "final_multiplier": 1.0,
    })
    unit_record["requested_set"] += int(requested != "HOLD")
    unit_record["applied_set"] += int(state.get("quota_consumed", 0))
    unit_record["forced_hold"] += int(bool(state.get("forced_hold_step", False)))
    unit_record["final_multiplier"] = float(state.get("level", 1.0))


def record_optimizer_reply(records, account, valid):
    if not valid:
        key = str(account)
        counts = records["optimizer_invalid_replies"]
        counts[key] = counts.get(key, 0) + 1


# ---------------- prompt 文本 ----------------
def planner_user(ep, a, auids, U, W, memory=None):
    """因果口径：U = 暖机 W 个 tick 后【已实现】的 unit_metrics（严格历史，无未来信息）。
    信号用 pace-normalized util（已花比例 ÷ 应花进度 W/48）判领跑/落后。"""
    B_acct = sum(U[u]["budget"] for u in auids)
    pace = W / 48.0
    lines = [f"Period ep={ep} | Account {a} (four competing ad units) | "
             f"Account budget={B_acct:.0f} (reallocate only within this account)",
             f"Observed {W}/48 time steps; elapsed-period share={pace:.0%}. "
             "The unit statistics below are realized observations. Reallocate the remaining "
             "budget. Return alpha in the displayed order as four nonnegative proportions "
             "summing to 1. Each final unit budget equals its spend to date plus its "
             "proportion of the account's remaining budget."]
    for j, u in enumerate(auids):
        d = U[u]
        cpa = "NA" if np.isnan(d["cpa"]) else f"{d['cpa']:.1f}"
        pace_util = d["util"] / pace if pace > 0 else 0.0   # >1=领跑(受预算约束) <1=落后(花不完)
        # 边际吸收力标签：相对 pacing 进度领跑=真实需求(该加)、落后=需求不足(该抽)
        if pace_util >= 1.1:
            tag = "ahead of pace; can absorb more" if d["eff"] >= 0.8 else "ahead of pace, but low efficiency"
        elif pace_util <= 0.7:
            tag = "behind pace; limited demand"
        else:
            tag = "near expected spending pace"
        lines.append(f"  Unit {j} (id {u}): budget={d['budget']:.0f}, spend={d['cost']:.0f}, "
                     f"relative pace={pace_util:.2f}x, fixed_adv/cost={d['eff']:.3f}, "
                     f"CPA={cpa}, conversions={d['conv']:.0f}; {tag}")
    lines.append("Favor units ahead of spending pace with high efficiency. Reduce allocations "
                 "to units behind pace or with low efficiency; efficiency alone does not "
                 "show whether a unit can absorb more budget.")
    if memory:
        pg = str(memory.get("planner_guidance") or "").strip()
        iss = str(memory.get("issue") or "").strip()
        if not _is_maintain(pg) or not _is_no_issue(iss):
            lines.append("Previous-period Inspector guidance:")
            if not _is_maintain(pg):
                lines.append(f"  Budget principle: {pg}")
            if not _is_no_issue(iss):
                lines.append(f"  Previous issue: {iss}")
    return "\n".join(lines)


def optimizer_user(ep, t, a, auids, U, cur_level, guidance):
    frac = t / 48.0
    lines = [f"Account {a} | Period ep={ep} | Time step t={t}/48 ({frac:.0%} elapsed)",
             "Current statistics for this account's units (return decisions in this order):"]
    for j, u in enumerate(auids):
        d = U[u]
        cpa = "NA" if np.isnan(d["cpa"]) else f"{d['cpa']:.1f}"
        dcpa = f"{d['desired_cpa']:.1f}"
        tapped = "(adjustment limit reached; this unit must HOLD)" if d["remaining_k"] <= 0 else ""
        lines.append(f"  Unit {j} (id {u}): current tCPA multiplier={cur_level[u]}, "
                     f"spend={d['cost']:.0f}/budget={d['budget']:.0f}, utilization={d['util']:.2f}, "
                     f"realized CPA={cpa}, initial target tCPA={dcpa}, "
                     f"conversions={d['conv']:.0f}, adjustments remaining={d['remaining_k']} "
                     f"{tapped}")
    lines.append("A realized CPA above the initial target indicates overpayment; lowering that "
                 "unit's control tCPA may help. A unit below target with high utilization may "
                 "allow more volume. Choose HOLD (no adjustment used) or a new level (one "
                 "adjustment used) for each unit. Save limited adjustments for useful times.")
    if not _is_maintain(guidance):
        lines.append(f"Inspector guidance on tCPA and timing: {guidance}")
    return "\n".join(lines)


def inspector_user(ep, snap, order, total, memory=None, action_records=None):
    total_cpa = "N/A" if np.isnan(total["cpa"]) else f"{total['cpa']:.1f}"
    headline = (f"Period ep={ep} review | Total fixed_adv={total['fix']:.0f}, "
                f"total CPA={total_cpa}, total conversions={total['conv']:.0f}")
    if "target_cost" in total:
        headline += f", target cost allowance={total['target_cost']:.0f}"
    lines = [headline, "Settled results by account:"]
    for i, a in enumerate(order):
        d = snap[a]
        cpa = "NA" if np.isnan(d["cpa"]) else f"{d['cpa']:.1f}"
        row = (f"  Account {i} ({a}): fixed_adv={d['fix']:.0f}, cost={d['cost']:.0f}, "
               f"conversions={d['conv']:.0f}, CPA={cpa}, utilization={d['util']:.2f}, "
               f"fixed_adv/cost={d['eff']:.3f}")
        lines.append(row)
    if action_records:
        lines.append("Executed decisions this period (requested actions and execution outcomes):")
        for i, a in enumerate(order):
            planner = action_records.get("planner", {}).get(str(a))
            if planner:
                fractions = ", ".join(f"{x:.2f}" for x in planner["executed_alpha"])
                budgets = ", ".join(f"{value:.0f}" for value in planner["executed_budgets"].values())
                fallback = " (fallback after invalid reply)" if planner["invalid_reply"] else ""
                lines.append(f"  Account {i} executed budget proportions=[{fractions}]{fallback}; "
                             f"executed unit budgets=[{budgets}].")
            units = action_records.get("optimizer", {}).get(str(a), {})
            if units:
                invalid_replies = action_records.get("optimizer_invalid_replies", {}).get(str(a), 0)
                changes = "; ".join(
                    f"unit {uid}: requested SET {rec['requested_set']}, "
                    f"executed SET {rec['applied_set']}, forced HOLD {rec['forced_hold']}, "
                    f"final multiplier {rec['final_multiplier']:.1f}"
                    for uid, rec in units.items())
                lines.append(f"  Account {i} tCPA actions: {changes}; "
                             f"invalid model replies: {invalid_replies} "
                             "(malformed unit actions executed as HOLD).")
    if memory and (memory.get("planner_guidance") or memory.get("optimizer_guidance")):
        lines.append("Your previous-period guidance (retain or revise it):")
        pg = str(memory.get("planner_guidance") or "").strip()
        og = str(memory.get("optimizer_guidance") or "").strip()
        if pg:
            lines.append(f"  Budget principle: {pg}")
        if og:
            lines.append(f"  tCPA principle: {og}")
    else:
        lines.append("This is the first period; no previous guidance is available.")
    lines.append("Identify any material issue (systematic overpayment, budget misallocation, "
                 "or excess cost). Give general budget and tCPA/timing principles for the next "
                 "period, without specific unit names or numerical targets. Use 'Maintain' "
                 "when no change is needed.")
    return "\n".join(lines)


# ---------------- LLM 调用 ----------------
def _call(cli, sysmsg, user, schema):
    from agents.llm_backend import extract_json
    resp = cli.complete(sysmsg, user, guided_json=schema)
    if not resp.ok:
        return None, "", resp.tokens
    root = extract_json(resp.text)
    if not isinstance(root, dict):
        return None, resp.text, resp.tokens
    return root, str(root.get("reasoning", "")), resp.tokens


def optimize_account(cli, ep, t, a, auids, U, cur_level, guidance):
    """per-account Optimizer 调用：只管该账户 4 单元，返回 (decisions, reasoning, tokens, illegal)。"""
    user = optimizer_user(ep, t, a, auids, U, cur_level, guidance)
    root, rc, tk = _call(cli, OPTIMIZER_SYS, user, OPTIMIZER_SCHEMA)
    decs, valid = parse_optimizer_decisions(root, len(auids))
    return decs, rc, tk, int(not valid)


# ---------------- 单 episode 模拟 ----------------
def new_simulation_state(order, groups, t_start=0):
    """Mutable counters for one suffix, shared by ``simulate`` and LangGraph."""
    uids = [u for a in order for u in groups[a]]
    return {"uids": uids, "cur_level": {u: "1.0" for u in uids},
            "send_level": {u: "HOLD" for u in uids}, "t": t_start,
            "done": False, "llm_calls": 0, "llm_tokens": 0, "n_illegal": 0,
            "forced_total": 0, "set_total": 0}


def simulate_step(e, ep, arm, cli, order, groups, ops, opt_guidance, K, state,
                  trace=None, action_records=None):
    """Run exactly one causal suffix tick and update ``state`` in place."""
    uids = state["uids"]
    cur_level = state["cur_level"]
    send_level = state["send_level"]
    t = state["t"]
    has_opt = arm in ("tcpa", "both", "full")
    acct_rc, acct_decs = {}, {}
    if has_opt:
        U = unit_metrics(e, order, groups, K)
        for a in order:
            auids = groups[a]
            decs, rc, tk, illegal = optimize_account(cli, ep, t, a, auids, U, cur_level, opt_guidance)
            state["llm_calls"] += 1; state["llm_tokens"] += tk; state["n_illegal"] += illegal
            if action_records is not None:
                record_optimizer_reply(action_records, a, illegal == 0)
            acct_rc[a] = rc; acct_decs[a] = decs
            for i, u in enumerate(auids):
                d = decs[i]
                if d == "HOLD":
                    send_level[u] = "HOLD"       # 保持当前档
                else:
                    send_level[u] = d
    else:
        for u in uids:
            send_level[u] = "HOLD"

    # step 本 tick
    acts = {u: {"tcpa_level": send_level[u]} for u in uids}
    state["done"] = e.step(acts).done

    # 读契约结果：只记真实变档 / 强制降级（供投手复核，避免 HOLD 刷屏）
    if has_opt:
        Upost = unit_metrics(e, order, groups, K)
        for a in order:
            changed = []
            for i, u in enumerate(groups[a]):
                ts = e._tcpa_state[u]
                cur_level[u] = str(float(ts["level"]))
                q = int(ts.get("quota_consumed", 0)); f = bool(ts.get("forced_hold_step", False))
                state["set_total"] += q; state["forced_total"] += int(f)
                if action_records is not None:
                    record_optimizer_outcome(action_records, a, u, acct_decs[a][i], ts)
                if q > 0 or f:
                    changed.append({"unit": u, "requested": acct_decs[a][i], "applied": ts.get("applied", "HOLD"),
                                    "quota_consumed": q, "forced_hold": f,
                                    "remaining_k": Upost[u]["remaining_k"],
                                    "cpa_so_far": None if np.isnan(Upost[u]["cpa"]) else round(Upost[u]["cpa"], 1),
                                    "util": round(Upost[u]["util"], 3)})
            if changed:
                ops.write(json.dumps({"episode": ep, "tick": t, "agent": "optimizer", "account": a,
                    "reasoning": acct_rc[a], "changes": changed}, ensure_ascii=False) + "\n")
    if trace is not None:
        snapshot = collect_metrics(e, order, groups)
        trace.append({"tick": t, "F": snapshot["fix"], "C": snapshot["cost"],
                      "N": snapshot["conv"], "cpa": None if np.isnan(snapshot["cpa"]) else snapshot["cpa"],
                      "budget": sum(float(e._units[u]["budget"]) for u in uids),
                      "levels": dict(cur_level)})
    state["t"] += 1
    return state["done"]


def collect_metrics(e, order, groups):
    per = {}
    tot_cost = tot_conv = tot_fix = tot_nat = tot_dyn = tot_rev = tot_target_cost = 0.0
    for a in order:
        cost = conv = fix = nat = budget = dyn = rev = target_cost = 0.0
        for u in groups[a]:
            dd = e._units[u]
            cost += float(dd["cum_cost"]); conv += float(dd["cum_conv"])
            target_cost += float(dd["desired_cpa"]) * float(dd["cum_conv"])
            fix += float(dd["cum_fixed_adv"]); nat += float(dd["cum_native_value"]); budget += float(dd["budget"])
            dyn += float(dd["cum_dynamic_adv"]); rev += float(dd["cum_revaluation"])
        per[a] = dict(cost=cost, conv=conv, fix=fix, target_cost=target_cost,
                      native_value=nat, budget=budget,
                      dynamic_adv=dyn, revaluation=rev,
                      cpa=cost / conv if conv > 0 else float("nan"))
        tot_cost += cost; tot_conv += conv; tot_fix += fix; tot_nat += nat
        tot_target_cost += target_cost
        tot_dyn += dyn; tot_rev += rev
    return dict(per_acct=per, cost=tot_cost, conv=tot_conv, fix=tot_fix,
                target_cost=tot_target_cost, native_value=tot_nat,
                dynamic_adv=tot_dyn, revaluation=tot_rev,
                cpa=tot_cost / tot_conv if tot_conv > 0 else float("nan"))


def run_warmup(e, uids, W):
    """暖机观察窗：前 W 个 tick 走原生预算 + 全 HOLD（纯观察，产生因果历史，供 t=W 的 Planner 决策）。"""
    for _ in range(W):
        if e.step({u: {"tcpa_level": "HOLD"} for u in uids}).done:
            break


def native_hold_rollout(ep, pv, seed, K, budget_scale=1.0, portfolio=None,
                        conversion_keep_rate=None, env_profile="sparse"):
    """Run No Operation with Abid, initial budgets, and HOLD at every step.

    Reference results are used only for evaluation, never as role-model inputs.
    """
    e = make_env(ep, pv, seed, K, budget_scale=budget_scale, portfolio=portfolio,
                 conversion_keep_rate=conversion_keep_rate, env_profile=env_profile)
    order, groups = _accts(e)
    base = {u: float(e._units[u]["base_budget"]) for u in e._units}
    e.set_our_strategies(_make_strategy)
    uids = list(e._units)
    done = False
    while not done:
        done = e.step({u: {"tcpa_level": "HOLD"} for u in uids}).done
    m = collect_metrics(e, order, groups)
    unit_native = unit_metrics(e, order, groups, K)      # 逐单元零运营参照
    acct_native = account_snapshot(e, order, groups)     # 逐账户零运营参照
    return m, acct_native, unit_native, order, groups, base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["none", "budget", "tcpa", "both", "full"])
    ap.add_argument("--episodes", default="0-9")
    ap.add_argument("--pv", type=int, default=500000)
    ap.add_argument("--seed", type=int, default=20270001)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=12, help="暖机观察窗 tick 数：前 W tick 原生预算+全HOLD 纯观察，t=W 起 Planner 因果再分配/Optimizer 激活")
    ap.add_argument("--env-profile", choices=["general", "sparse", "extra_sparse_feedback_stress"], default="sparse")
    ap.add_argument("--outdir", default="data/ladder_2026-07-28")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default=None,
                    help="served-model-name of the local base model (env ADCOEVO_SERVED_MODEL, default 'model')")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--constrained-json", action=argparse.BooleanOptionalAction, default=True,
                    help="constrain JSON generation in addition to specifying reply format in prompts")
    args = ap.parse_args()

    if "-" in args.episodes:
        lo, hi = args.episodes.split("-"); eps = list(range(int(lo), int(hi) + 1))
    else:
        eps = [int(x) for x in args.episodes.split(",")]

    use_llm = args.arm != "none"
    cli = None
    if use_llm:
        from agents.llm_backend import LLMClient
        from train.paths import served_model_name
        cli = LLMClient(dict(base_url=args.base_url, model=served_model_name(args.model), api_key="EMPTY",
                             temperature=args.temperature, max_tokens=512, timeout=90, enable_thinking=False,
                             constrained_json=args.constrained_json))

    armdir = os.path.join(args.outdir, args.arm)
    os.makedirs(armdir, exist_ok=True)
    ops = open(os.path.join(armdir, "ops.jsonl"), "w")
    epf = open(os.path.join(armdir, "episodes.jsonl"), "w")

    mem = None   # full 臂：Inspector 维护的跨周期宏观建议 {planner_guidance, optimizer_guidance, issue, updated_after_ep}

    for ep in eps:
        # No Operation rollout supplies the evaluation baseline only.
        nat_m, _, _, order, groups, base = native_hold_rollout(
            ep, args.pv, args.seed, args.K, env_profile=args.env_profile)
        cpa_native = nat_m["cpa"]
        B = sum(base.values())

        if args.arm == "none":
            m = nat_m
            llm_calls = llm_tokens = n_illegal = forced_total = set_total = 0
        else:
            from agents.period_graph import build_period_graph

            W = args.warmup
            context = {"memory": mem, "pre_calls": 0, "pre_tokens": 0, "pre_illegal": 0}

            def warmup_node(state):
                ctx = state["context"]
                e = make_env(ep, args.pv, args.seed, args.K, env_profile=args.env_profile)
                order, groups = _accts(e)
                base = {u: float(e._units[u]["base_budget"]) for u in e._units}
                uids_order = [u for a in order for u in groups[a]]
                e.set_our_strategies(_make_strategy)
                run_warmup(e, uids_order, W)
                ctx.update(e=e, order=order, groups=groups, base=base, uids_order=uids_order,
                           U_warm=unit_metrics(e, order, groups, args.K),
                           action_records=new_action_records(),
                           simulation=new_simulation_state(order, groups, W))
                return {"context": ctx}

            def planner_node(state):
                ctx = state["context"]
                if args.arm in ("budget", "both", "full"):
                    pmem = ctx["memory"] if args.arm == "full" else None
                    spent_warm = {u: ctx["U_warm"][u]["cost"] for u in ctx["uids_order"]}
                    newb = {}; alpha_all = {}; reason_all = {}
                    for a in ctx["order"]:
                        auids = ctx["groups"][a]
                        root, rc, tk = _call(cli, PLANNER_SYS,
                                             planner_user(ep, a, auids, ctx["U_warm"], W, memory=pmem),
                                             PLANNER_SCHEMA)
                        alpha, valid = parse_planner_alpha(
                            root, len(auids), [ctx["base"][u] for u in auids])
                        ctx["pre_illegal"] += int(not valid)
                        nb = apply_budget_proportions(ctx["e"], auids, alpha, ctx["base"])
                        newb.update(nb)
                        alpha_all[a] = [round(float(x), 3) for x in alpha]
                        record_planner_action(ctx["action_records"], a, auids, alpha, nb, valid,
                                              root.get("alpha") if isinstance(root, dict) else None,
                                              {u: ctx["U_warm"][u]["cost"] for u in auids})
                        reason_all[a] = rc
                        ctx["pre_calls"] += 1; ctx["pre_tokens"] += tk
                    to_acct = {a: round(sum(newb[u] for u in ctx["groups"][a]), 0)
                               for a in ctx["order"]}
                    ops.write(json.dumps({"episode": ep, "tick": W, "agent": "planner", "warmup_W": W,
                        "spent_in_warmup": {str(u): round(spent_warm[u], 0) for u in ctx["uids_order"]},
                        "action": {"alpha_by_acct": alpha_all,
                                   "to_unit": {str(u): round(newb[u], 0) for u in ctx["uids_order"]},
                                   "to_acct": to_acct},
                        "reasoning": reason_all, "inspector_memory": pmem}, ensure_ascii=False) + "\n")
                return {"context": ctx}

            def optimizer_step_node(state):
                ctx = state["context"]
                guidance = (ctx["memory"].get("optimizer_guidance")
                            if (args.arm == "full" and ctx["memory"]) else "")
                done = simulate_step(ctx["e"], ep, args.arm, cli, ctx["order"], ctx["groups"],
                                     ops, guidance, args.K, ctx["simulation"],
                                     action_records=ctx["action_records"])
                if done:
                    ctx["metrics"] = collect_metrics(ctx["e"], ctx["order"], ctx["groups"])
                return {"context": ctx, "done": done}

            def inspector_node(state):
                ctx = state["context"]
                if args.arm == "full":
                    snap = account_snapshot(ctx["e"], ctx["order"], ctx["groups"])
                    root, rc, tk = _call(cli, INSPECTOR_SYS,
                                         inspector_user(ep, snap, ctx["order"], ctx["metrics"],
                                                        memory=ctx["memory"],
                                                        action_records=ctx["action_records"]),
                                         INSPECTOR_SCHEMA)
                    ctx["inspector_tokens"] = tk
                    ctx["inspector_illegal"] = int(not valid_inspector_reply(root))
                    ctx["memory"] = inspector_memory(root, ep)
                    imem = ctx["memory"]
                    ops.write(json.dumps({"episode": ep, "tick": 48, "agent": "inspector", "phase": "end",
                        "reasoning": rc, "has_issue": imem["has_issue"], "issue_summary": imem["issue"],
                        "planner_guidance": imem["planner_guidance"],
                        "optimizer_guidance": imem["optimizer_guidance"], "carried_memory": imem},
                        ensure_ascii=False) + "\n")
                return {"context": ctx}

            graph = build_period_graph(warmup_node, planner_node, optimizer_step_node, inspector_node)
            context = graph.invoke({"context": context, "done": False},
                                   config={"recursion_limit": 128})["context"]
            order = context["order"]
            m = context["metrics"]
            sim = context["simulation"]
            llm_calls = context["pre_calls"] + sim["llm_calls"] + int(args.arm == "full")
            llm_tokens = context["pre_tokens"] + sim["llm_tokens"] + context.get("inspector_tokens", 0)
            n_illegal = context["pre_illegal"] + sim["n_illegal"] + context.get("inspector_illegal", 0)
            forced_total, set_total = sim["forced_total"], sim["set_total"]
            mem = context["memory"]

        feasible = (not np.isnan(m["cpa"])) and (not np.isnan(cpa_native)) and m["cpa"] <= cpa_native * CPA_TOL
        rec = {"arm": args.arm, "episode": ep, "seed": args.seed, "pv": args.pv, "K": args.K,
               "environment_profile": args.env_profile,
               "fixed_adv": m["fix"], "dynamic_adv": m["dynamic_adv"], "revaluation": m["revaluation"],
               "cost": m["cost"], "conv": m["conv"], "native_value": m["native_value"],
               "cpa": None if np.isnan(m["cpa"]) else m["cpa"],
               "cpa_native": None if np.isnan(cpa_native) else cpa_native,
               "fixed_native": nat_m["fix"], "cpa_feasible": bool(feasible),
               "per_acct": {a: {"fixed_adv": m["per_acct"][a]["fix"], "cost": m["per_acct"][a]["cost"],
                                "conv": m["per_acct"][a]["conv"], "budget": m["per_acct"][a]["budget"],
                                "native_value": m["per_acct"][a]["native_value"],
                                "dynamic_adv": m["per_acct"][a]["dynamic_adv"],
                                "revaluation": m["per_acct"][a]["revaluation"],
                                "cpa": None if np.isnan(m["per_acct"][a]["cpa"]) else m["per_acct"][a]["cpa"]}
                            for a in order},
               "llm_calls": llm_calls, "llm_tokens": llm_tokens,
               "n_illegal": n_illegal, "forced_hold": forced_total, "set_total": set_total}
        epf.write(json.dumps(rec, ensure_ascii=False) + "\n"); epf.flush()
        dpct = 100 * (m["fix"] / nat_m["fix"] - 1) if nat_m["fix"] > 0 else 0.0
        print(f"[{args.arm}] ep{ep}: fix={m['fix']:.0f} ({dpct:+.1f}% vs 零运营) "
              f"cpa={m['cpa']:.1f}(native {cpa_native:.1f}) feas={'Y' if feasible else '-'} "
              f"SET={set_total} forced={forced_total} illegal={n_illegal} calls={llm_calls}", flush=True)

    ops.close(); epf.close()
    print(f"[{args.arm}] done → {armdir}", flush=True)


if __name__ == "__main__":
    main()
