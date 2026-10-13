#!/usr/bin/env python3
"""Evaluate the Rule policy against No Operation on matched AuctionNet episodes."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path[:0] = [".", "scripts"]
import run_ladder as RL


def planner_mults(metrics, unit_ids):
    multipliers = []
    for unit_id in unit_ids:
        row = metrics[unit_id]
        cpa = row["cpa"]
        if not math.isfinite(cpa):
            multipliers.append(1.0)
        elif cpa <= row["desired_cpa"]:
            multipliers.append(1.1)
        else:
            multipliers.append(0.9)
    return multipliers


def optimizer_decision(row):
    cpa = row["cpa"]
    if row["remaining_k"] <= 0 or not math.isfinite(cpa):
        return "HOLD"
    target_cpa = row["desired_cpa"]
    if cpa > 1.10 * target_cpa:
        return "0.7"
    if cpa < 0.90 * target_cpa:
        return "1.3"
    return "HOLD"


def parse_episodes(value):
    if "-" in value:
        lo, hi = (int(item) for item in value.split("-"))
        return list(range(lo, hi + 1))
    return [int(item) for item in value.split(",")]


def run_episode(episode, pv, seed, profile, budget_scale, K=8, warmup=12):
    native, _, _, _, _, base = RL.native_hold_rollout(
        episode, pv, seed, K, budget_scale=budget_scale, env_profile=profile)
    env = RL.make_env(episode, pv, seed, K, budget_scale=budget_scale,
                      env_profile=profile)
    order, groups = RL._accts(env)
    env.set_our_strategies(RL._make_strategy)
    unit_ids = [unit for account in order for unit in groups[account]]
    RL.run_warmup(env, unit_ids, warmup)

    warmup_metrics = RL.unit_metrics(env, order, groups, K)
    for account in order:
        account_units = groups[account]
        RL.apply_rule_budget(
            env, account_units, planner_mults(warmup_metrics, account_units), base)

    forced = 0
    set_total = 0
    done = False
    while not done:
        metrics = RL.unit_metrics(env, order, groups, K)
        actions = {
            unit: {"tcpa_level": optimizer_decision(metrics[unit])}
            for unit in unit_ids
        }
        done = env.step(actions).done
        for unit in unit_ids:
            state = env._tcpa_state[unit]
            forced += int(bool(state.get("forced_hold_step", False)))
            set_total += int(state.get("quota_consumed", 0))

    result = RL.collect_metrics(env, order, groups)
    feasible = (not np.isnan(result["cpa"])) and result["cpa"] <= native["cpa"] * RL.CPA_TOL
    return {
        "arm": "rule", "episode": episode, "seed": seed, "pv": pv, "K": K,
        "warmup": warmup, "budget_scale": float(budget_scale),
        "environment_profile": profile,
        "F": float(result["fix"]), "C": float(result["cost"]),
        "N": float(result["conv"]),
        "native_F": float(native["fix"]), "native_C": float(native["cost"]),
        "native_N": float(native["conv"]),
        "cpa": None if np.isnan(result["cpa"]) else float(result["cpa"]),
        "cpa_native": None if np.isnan(native["cpa"]) else float(native["cpa"]),
        "cpa_feasible": bool(feasible), "illegal": 0, "forced": forced,
        "set_total": set_total, "llm_calls": 0, "llm_tokens": 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", default="0-49")
    parser.add_argument("--pv", type=int, default=500000)
    parser.add_argument("--seed", type=int, default=20270001)
    parser.add_argument("--env-profile", choices=["general", "sparse"], default="sparse")
    parser.add_argument("--budget-scale", type=float, default=1.0)
    parser.add_argument("--out", default="runs/rule_demo")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    episodes = parse_episodes(args.episodes)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    arms_path = out / "arms.jsonl"
    state_path = out / "resume_state.json"
    protocol = {"episodes": episodes, "pv": args.pv, "seed": args.seed,
                "environment_profile": args.env_profile,
                "budget_scale": float(args.budget_scale), "K": 8, "warmup": 12}
    rows = []
    if args.resume and arms_path.exists():
        rows = [json.loads(line) for line in arms_path.open(encoding="utf-8") if line.strip()]
        completed_sequence = [int(row["episode"]) for row in rows]
        if completed_sequence != episodes[:len(completed_sequence)]:
            raise SystemExit("cannot resume: completed episodes are not a contiguous prefix")
        if not state_path.exists():
            raise SystemExit(f"cannot resume: {state_path} is missing")
        saved = json.loads(state_path.read_text(encoding="utf-8"))
        if saved.get("protocol") != protocol or saved.get("completed_episodes") != completed_sequence:
            raise SystemExit("cannot resume: protocol or completed sequence mismatch")
    elif args.resume:
        rows = []
    elif arms_path.exists() or state_path.exists():
        raise SystemExit(f"refusing to overwrite existing output in {out}; use --resume or a fresh directory")

    completed = {int(row["episode"]) for row in rows}
    mode = "a" if args.resume else "w"
    with arms_path.open(mode, encoding="utf-8") as handle:
        for episode in episodes:
            if episode in completed:
                continue
            row = run_episode(episode, args.pv, args.seed, args.env_profile,
                              args.budget_scale)
            rows.append(row)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            state_path.write_text(json.dumps({
                "protocol": protocol,
                "completed_episodes": [int(item["episode"]) for item in rows],
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[rule] ep{episode} F={row['F']:.1f} CPA={row['cpa']} "
                  f"N={row['N']:.0f} feasible={row['cpa_feasible']}", flush=True)

    F = sum(row["F"] for row in rows)
    C = sum(row["C"] for row in rows)
    N = sum(row["N"] for row in rows)
    summary = {
        "method": "Rule", "environment_profile": args.env_profile,
        "episodes": episodes, "seed": args.seed, "pv": args.pv,
        "n": len(rows), "F": F, "ADV_per_period": F / len(rows),
        "C": C, "N": N, "CPA": C / N if N else None,
        "N_per_period": N / len(rows),
        "feasible_rate": sum(row["cpa_feasible"] for row in rows) / len(rows),
        "illegal": sum(row["illegal"] for row in rows),
        "forced": sum(row["forced"] for row in rows),
        "set_total": sum(row["set_total"] for row in rows),
    }
    from train.evaluation_metrics import summarize_delivery
    summary.update(summarize_delivery(rows))
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
