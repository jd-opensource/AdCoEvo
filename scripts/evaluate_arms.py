#!/usr/bin/env python3
"""Held-out five-arm business evaluation using run_ladder.

对每个 held-out episode 跑指定臂（none=A1 零运营基线 / budget / tcpa / both / full），
产逐 episode 记录（fixed_adv=F, cost=C, conv=N, cpa, feasible, illegal, forced），
再用 business_gate 把候选臂 vs incumbent 臂（默认 both vs none）配对裁决。

因果口径：暖机 W=12（原生预算+全 HOLD 纯观察），Planner 在 t=W 账户级再分配一次，
Optimizer t=W..47 每 tick 自判变档。tcpa/both/full 臂需 cli（vLLM serve 目标 LoRA）。
none 臂无 LLM，可离线跑作基线/冒烟。

用法：
  # A1 基线（无 LLM）
  python scripts/evaluate_arms.py --arms none --episodes 0-9 --pv 500000 --out eval/a1
  # A4 vs A1 裁决（需先起 vLLM）
  python scripts/evaluate_arms.py --arms none,both --cand-arm both --inc-arm none \
        --episodes 0-9 --pv 500000 --base-url http://localhost:8000/v1 --model <served-model-name> --out eval/a4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")

import numpy as np

import run_ladder as RL
from env.auctionnet_profile import load_profile
from train.business_gate import BusinessThresholds, business_gate
from train.evaluation_metrics import summarize_delivery


def run_episode_arm(arm, ep, cli, pv, seed, K, W, mem=None, max_migration_ratio=None,
                    budget_scale=1.0, portfolio=None, telemetry=None,
                    conversion_keep_rate=None, env_profile="sparse"):
    """在单个 episode 上跑一个臂，返回 (rec, mem_next)。rec 含 F/C/N/cpa/feasible/illegal/forced。
    逻辑与 run_ladder.main 单 episode 分支等价（复用同一批函数，禁另写拍卖逻辑）。
    cli 可为单个 LLMClient，也可为 {planner, optimizer, inspector}，用于 LoRA 角色路由。"""
    if isinstance(cli, dict):
        planner_cli = cli.get("planner")
        optimizer_cli = cli.get("optimizer")
        inspector_cli = cli.get("inspector") or planner_cli
    else:
        planner_cli = optimizer_cli = inspector_cli = cli
    episode_started = time.perf_counter()
    nat_m, _, _, order, groups, base = RL.native_hold_rollout(
        ep, pv, seed, K, budget_scale=budget_scale, portfolio=portfolio,
        conversion_keep_rate=conversion_keep_rate, env_profile=env_profile)
    cpa_native = nat_m["cpa"]

    if arm == "none":
        m = nat_m
        n_illegal = forced_total = set_total = 0
        pre_calls = pre_tokens = llm_calls = llm_tokens = 0
    else:
        from agents.period_graph import build_period_graph

        context = {"memory": mem, "pre_calls": 0, "pre_tokens": 0, "pre_illegal": 0,
                   "trace": [] if telemetry is not None else None}

        def warmup_node(state):
            ctx = state["context"]
            e = RL.make_env(ep, pv, seed, K, budget_scale=budget_scale, portfolio=portfolio,
                            conversion_keep_rate=conversion_keep_rate, env_profile=env_profile)
            order, groups = RL._accts(e)
            base = {u: float(e._units[u]["base_budget"]) for u in e._units}
            uids_order = [u for a in order for u in groups[a]]
            e.set_our_strategies(RL._make_strategy)
            RL.run_warmup(e, uids_order, W)
            ctx.update(e=e, order=order, groups=groups, base=base,
                       U_warm=RL.unit_metrics(e, order, groups, K),
                       action_records=RL.new_action_records(),
                       simulation=RL.new_simulation_state(order, groups, W))
            return {"context": ctx}

        def planner_node(state):
            ctx = state["context"]
            if arm in ("budget", "both", "full"):
                pmem = ctx["memory"] if arm == "full" else None
                for a in ctx["order"]:
                    auids = ctx["groups"][a]
                    root, rc, tk = RL._call(planner_cli, RL.PLANNER_SYS,
                                            RL.planner_user(ep, a, auids, ctx["U_warm"], W, memory=pmem),
                                            RL.PLANNER_SCHEMA)
                    alpha, valid = RL.parse_planner_alpha(
                        root, len(auids), [ctx["base"][u] for u in auids])
                    ctx["pre_illegal"] += int(not valid)
                    budgets = RL.apply_budget_proportions(
                        ctx["e"], auids, alpha, ctx["base"], max_migration_ratio=max_migration_ratio)
                    RL.record_planner_action(ctx["action_records"], a, auids, alpha, budgets, valid,
                                             root.get("alpha") if isinstance(root, dict) else None,
                                             {u: ctx["U_warm"][u]["cost"] for u in auids})
                    ctx["pre_calls"] += 1; ctx["pre_tokens"] += tk
            return {"context": ctx}

        def optimizer_step_node(state):
            ctx = state["context"]
            guidance = (ctx["memory"].get("optimizer_guidance")
                        if (arm == "full" and ctx["memory"]) else "")
            done = RL.simulate_step(
                ctx["e"], ep, arm, optimizer_cli, ctx["order"], ctx["groups"],
                _NullOps(), guidance, K, ctx["simulation"], trace=ctx["trace"],
                action_records=ctx["action_records"])
            if done:
                ctx["metrics"] = RL.collect_metrics(ctx["e"], ctx["order"], ctx["groups"])
            return {"context": ctx, "done": done}

        def inspector_node(state):
            ctx = state["context"]
            if arm == "full":
                snap = RL.account_snapshot(ctx["e"], ctx["order"], ctx["groups"])
                root, rc, tk = RL._call(inspector_cli, RL.INSPECTOR_SYS,
                                        RL.inspector_user(ep, snap, ctx["order"], ctx["metrics"],
                                                          memory=ctx["memory"],
                                                          action_records=ctx["action_records"]),
                                        RL.INSPECTOR_SCHEMA)
                ctx["inspector_tokens"] = tk
                ctx["inspector_illegal"] = int(not RL.valid_inspector_reply(root))
                ctx["memory"] = RL.inspector_memory(root, ep)
            return {"context": ctx}

        graph = build_period_graph(warmup_node, planner_node, optimizer_step_node, inspector_node)
        context = graph.invoke({"context": context, "done": False},
                               config={"recursion_limit": 128})["context"]
        order, groups, base = context["order"], context["groups"], context["base"]
        m = context["metrics"]
        sim = context["simulation"]
        n_illegal = context["pre_illegal"] + sim["n_illegal"] + context.get("inspector_illegal", 0)
        forced_total, set_total = sim["forced_total"], sim["set_total"]
        llm_calls = context["pre_calls"] + sim["llm_calls"] + int(arm == "full")
        llm_tokens = context["pre_tokens"] + sim["llm_tokens"] + context.get("inspector_tokens", 0)
        mem = context["memory"]
        if telemetry is not None:
            telemetry.extend({"episode": ep, **item} for item in context["trace"])

    feasible = (not np.isnan(m["cpa"])) and (not np.isnan(cpa_native)) and m["cpa"] <= cpa_native * RL.CPA_TOL
    # Budget allocation preserves the total budget of each account.
    budget_err = abs(sum(base.values()) - sum(nat_m["per_acct"][a]["budget"] for a in order)) if arm == "none" else 0.0
    profile = load_profile(env_profile)
    effective_keep_rate = (profile.conversion_keep_rate if conversion_keep_rate is None
                           else float(conversion_keep_rate))
    stress_profile = ("extra_sparse_feedback_stress_57_of_1512"
                      if effective_keep_rate < 1.0 else None)
    rec = dict(arm=arm, episode=ep, seed=seed, pv=pv, K=K, budget_scale=float(budget_scale),
               environment_profile=env_profile, calibration_id=profile.calibration_id,
               calibration_hash=profile.calibration_hash,
               conversion_keep_rate=effective_keep_rate, sparse_profile=stress_profile,
               F=float(m["fix"]), C=float(m["cost"]), N=float(m["conv"]),
               native_F=float(nat_m["fix"]), native_C=float(nat_m["cost"]),
               native_N=float(nat_m["conv"]),
               dynamic_adv=float(m["dynamic_adv"]), revaluation=float(m["revaluation"]),
               cpa=None if np.isnan(m["cpa"]) else float(m["cpa"]),
               cpa_native=None if np.isnan(cpa_native) else float(cpa_native),
               cpa_feasible=bool(feasible),
               viol=0.0 if feasible else 1.0, illegal=int(n_illegal),
               forced=int(forced_total), set_total=int(set_total),
               budget_err=float(budget_err),
               llm_calls=int(llm_calls), llm_tokens=int(llm_tokens),
               runtime_s=time.perf_counter() - episode_started)
    return rec, mem


class _NullOps:
    """simulate 需要一个 ops 句柄写逐操作日志；eval 不落该日志。"""
    def write(self, *_a, **_k):
        pass


def _parse_eps(s):
    if "-" in s:
        lo, hi = s.split("-"); return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in s.split(",")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="none", help="逗号分隔：none,budget,tcpa,both,full")
    ap.add_argument("--episodes", default="0-9")
    ap.add_argument("--pv", type=int, default=500000)
    ap.add_argument("--seed", type=int, default=20270001)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=12)
    ap.add_argument("--env-profile", choices=["general", "sparse", "extra_sparse_feedback_stress"], default="sparse")
    ap.add_argument("--conversion-keep-rate", type=float, default=None)
    ap.add_argument("--out", default="eval/arms")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default=None,
                    help="served-model-name of the local base model (env ADCOEVO_SERVED_MODEL, default 'model')")
    ap.add_argument("--planner-model", default=None, help="Planner 使用的 served model；默认回退到 --model")
    ap.add_argument("--optimizer-model", default=None, help="Optimizer 使用的 served model；默认回退到 --model")
    ap.add_argument("--inspector-model", default=None, help="Inspector 使用的 served model；默认回退到 --planner-model/--model")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--constrained-json", action=argparse.BooleanOptionalAction, default=True,
                    help="constrain JSON generation in addition to specifying reply format in prompts")
    # 裁决：候选臂 vs incumbent 臂
    ap.add_argument("--cand-arm", default=None)
    ap.add_argument("--inc-arm", default=None)
    ap.add_argument("--delta_F", type=float, default=0.0)
    ap.add_argument("--delta_CPA", type=float, default=0.0)
    ap.add_argument("--delta_N", type=float, default=0.05)
    ap.add_argument("--cpa_max", type=float, default=float("inf"))
    ap.add_argument("--n_boot", type=int, default=1000)
    args = ap.parse_args()
    if args.conversion_keep_rate is not None and not 0.0 < args.conversion_keep_rate <= 1.0:
        raise SystemExit("--conversion-keep-rate must be in (0, 1]")

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    eps = _parse_eps(args.episodes)
    need_llm = any(a != "none" for a in arms)
    cli = None
    if need_llm:
        from agents.llm_backend import LLMClient
        from train.paths import served_model_name
        args.model = served_model_name(args.model)
        planner_model = args.planner_model or args.model
        optimizer_model = args.optimizer_model or args.model
        inspector_model = args.inspector_model or planner_model
        def make_cli(model):
            return LLMClient(dict(base_url=args.base_url, model=model, api_key="EMPTY",
                                  temperature=args.temperature, max_tokens=512, timeout=90,
                                  enable_thinking=False, constrained_json=args.constrained_json))
        if planner_model == optimizer_model == inspector_model:
            cli = make_cli(planner_model)
        else:
            cli = {"planner": make_cli(planner_model),
                   "optimizer": make_cli(optimizer_model),
                   "inspector": make_cli(inspector_model)}

    os.makedirs(args.out, exist_ok=True)
    recs = {a: [] for a in arms}
    mem = {a: None for a in arms}
    with open(os.path.join(args.out, "arms.jsonl"), "w", encoding="utf-8") as f:
        for ep in eps:
            for a in arms:
                rec, mem[a] = run_episode_arm(
                    a, ep, cli, args.pv, args.seed, args.K, args.warmup, mem[a],
                    conversion_keep_rate=args.conversion_keep_rate, env_profile=args.env_profile)
                recs[a].append(rec)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                print(f"[eval:{a}] ep{ep} F={rec['F']:.0f} C={rec['C']:.0f} N={rec['N']:.0f} "
                      f"cpa={rec['cpa']} feasible={rec['cpa_feasible']}")

    summary = {}
    for a in arms:
        summary[a] = summarize_delivery(recs[a])
        summary[a]["feasible_rate"] = float(np.mean([r["cpa_feasible"] for r in recs[a]]))
    print(f"[eval] summary={json.dumps(summary, ensure_ascii=False)}")

    verdict = None
    if args.cand_arm and args.inc_arm:
        assert args.cand_arm in recs and args.inc_arm in recs, "cand/inc 臂须在 --arms 内"
        th = BusinessThresholds(delta_F=args.delta_F, delta_CPA=args.delta_CPA,
                                delta_N=args.delta_N, cpa_max=args.cpa_max, n_boot=args.n_boot)
        rep = business_gate(recs[args.cand_arm], recs[args.inc_arm], th)
        verdict = rep.to_dict()
        print(f"[eval] 裁决 {args.cand_arm} vs {args.inc_arm}: {rep.verdict}  "
              f"dF_lcb={rep.dF_lcb:.3f} dCPA_ucb={rep.dCPA_ucb:.3f} "
              f"cpa_cand_ucb={rep.cpa_cand_ucb:.3f} dN_lcb={rep.dN_lcb:.3f}")
        print(f"    checks={rep.checks}")

    with open(os.path.join(args.out, "eval_report.json"), "w", encoding="utf-8") as f:
        json.dump(dict(arms=arms, episodes=eps, summary=summary,
                       environment_profile=args.env_profile,
                       conversion_keep_rate=args.conversion_keep_rate,
                       cand_arm=args.cand_arm, inc_arm=args.inc_arm, verdict=verdict),
                  f, ensure_ascii=False, indent=2)
    print(f"    → {args.out}/arms.jsonl  {args.out}/eval_report.json")


if __name__ == "__main__":
    main()
