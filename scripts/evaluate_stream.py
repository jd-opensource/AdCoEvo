#!/usr/bin/env python3
"""Evaluate fixed role policies across a continuous market sequence.

Inspector guidance carries between periods. Completed results and feedback state are
saved together for resuming an interrupted run.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
import time
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")

import numpy as np

from agents.llm_backend import LLMClient
from scripts.evaluate_arms import _parse_eps, run_episode_arm
from train.coevolution.ablation import FeedbackRouter, MODES
from train.evaluation_metrics import summarize_delivery


class TelemetryClient:
    def __init__(self, client, role, records):
        self.client = client
        self.role = role
        self.records = records

    def complete(self, *args, **kwargs):
        response = self.client.complete(*args, **kwargs)
        self.records.append({"role": self.role, "latency_s": response.latency_s,
                             "tokens": response.tokens, "ok": response.ok,
                             "error": response.error})
        return response


def make_cli(base_url, model, temperature, api_key, role=None, telemetry=None,
             constrained_json=True):
    client = LLMClient(dict(base_url=base_url, model=model, api_key=api_key,
                            temperature=temperature, top_p=(1.0 if temperature == 0 else None),
                            max_tokens=512, timeout=90, enable_thinking=False,
                            constrained_json=constrained_json))
    return TelemetryClient(client, role, telemetry) if telemetry is not None else client


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="full")
    ap.add_argument("--episodes", default="0-49")
    ap.add_argument("--pv", type=int, default=500000)
    ap.add_argument("--seed", type=int, default=20270001)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=12)
    ap.add_argument("--max-migration-ratio", type=float, default=None)
    ap.add_argument("--budget-scale", type=float, default=1.0)
    ap.add_argument("--env-profile", choices=["general", "sparse", "extra_sparse_feedback_stress"], default="sparse")
    ap.add_argument("--conversion-keep-rate", type=float, default=None)
    ap.add_argument("--portfolio", default=None, help="12 comma-separated AuctionNet unit indices")
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-url", default="http://localhost:8001/v1")
    ap.add_argument("--model", default=None,
                    help="served-model-name of the local base model; any model works "
                         "(env ADCOEVO_SERVED_MODEL, default 'model')")
    ap.add_argument("--api-key-env", default=None)
    ap.add_argument("--planner-model", default=None)
    ap.add_argument("--optimizer-model", default=None)
    ap.add_argument("--inspector-model", default=None)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--constrained-json", action=argparse.BooleanOptionalAction, default=True,
                    help="use --no-constrained-json for prompt-only general-purpose model evaluation")
    ap.add_argument("--ablation-mode", choices=MODES, default="full")
    ap.add_argument("--telemetry", action="store_true")
    ap.add_argument("--pacing-trace", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--stop-on-error", action="store_true",
                    help="stop at the first failed period so a later --resume keeps feedback causal")
    args = ap.parse_args()
    if args.conversion_keep_rate is not None and not 0.0 < args.conversion_keep_rate <= 1.0:
        raise SystemExit("--conversion-keep-rate must be in (0, 1]")
    portfolio = None
    if args.portfolio:
        portfolio = [int(x) for x in args.portfolio.split(",")]
        if len(portfolio) != 12 or len(set(portfolio)) != 12:
            raise SystemExit("--portfolio requires 12 unique unit indices")

    api_key=os.environ.get(args.api_key_env, "EMPTY") if args.api_key_env else "EMPTY"
    if args.model is None:
        from train.paths import served_model_name
        args.model = served_model_name()
    telemetry = [] if args.telemetry else None
    pacing_trace = [] if args.pacing_trace else None
    planner_model = args.planner_model or args.model
    optimizer_model = args.optimizer_model or args.model
    inspector_model = args.inspector_model or planner_model
    if planner_model == optimizer_model == inspector_model and telemetry is None:
        cli = make_cli(args.base_url, planner_model, args.temperature, api_key,
                       constrained_json=args.constrained_json)
    else:
        cli = {"planner": make_cli(args.base_url, planner_model, args.temperature, api_key, "planner", telemetry,
                                    constrained_json=args.constrained_json),
               "optimizer": make_cli(args.base_url, optimizer_model, args.temperature, api_key, "optimizer", telemetry,
                                      constrained_json=args.constrained_json),
               "inspector": make_cli(args.base_url, inspector_model, args.temperature, api_key, "inspector", telemetry,
                                      constrained_json=args.constrained_json)}

    eps = _parse_eps(args.episodes)
    os.makedirs(args.out, exist_ok=True)
    out_path = Path(args.out)
    arms_path = out_path / "arms.jsonl"
    failures_path = out_path / "failures.jsonl"
    state_path = out_path / "resume_state.json"
    recs = []
    fails = []
    feedback = FeedbackRouter(args.ablation_mode, args.seed)
    completed = set()
    if args.resume and arms_path.exists():
        recs = [json.loads(line) for line in arms_path.open(encoding="utf-8") if line.strip()]
        completed = {int(row["episode"]) for row in recs}
        if completed:
            if not state_path.exists():
                raise SystemExit(f"cannot resume: {state_path} is missing")
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if state.get("completed_episodes") != [ep for ep in eps if ep in completed]:
                raise SystemExit("cannot resume: completed episode sequence does not match resume_state")
            expected_env = {"pv": args.pv, "seed": args.seed, "K": args.K,
                            "warmup": args.warmup, "budget_scale": args.budget_scale,
                            "conversion_keep_rate": args.conversion_keep_rate,
                            "environment_profile": args.env_profile, "portfolio": portfolio,
                            "temperature": args.temperature, "constrained_json": args.constrained_json}
            if state["environment"] != expected_env:
                raise SystemExit("cannot resume: environment configuration does not match resume_state")
            feedback.history = [(int(ep), mem) for ep, mem in state.get("feedback_history", [])]
            feedback.mapping = state.get("feedback_mapping", [])
    mode = "a" if args.resume else "w"
    run_started = time.perf_counter()
    with arms_path.open(mode, encoding="utf-8") as af, failures_path.open(mode, encoding="utf-8") as ff:
        for ep in eps:
            if ep in completed:
                continue
            try:
                mem, feedback_source_episode = feedback.consume(ep)
                rec, produced_mem = run_episode_arm(
                    args.arm, ep, cli, args.pv, args.seed, args.K, args.warmup, mem,
                    max_migration_ratio=args.max_migration_ratio, budget_scale=args.budget_scale,
                    portfolio=portfolio, telemetry=pacing_trace,
                    conversion_keep_rate=args.conversion_keep_rate, env_profile=args.env_profile)
                feedback.produce(ep, produced_mem or {})
                rec["ablation_mode"] = args.ablation_mode
                rec["feedback_source_episode"] = feedback_source_episode
                recs.append(rec)
                af.write(json.dumps(rec, ensure_ascii=False) + "\n"); af.flush()
                state_path.write_text(json.dumps({
                    "completed_episodes": [x for x in eps if x in {int(row['episode']) for row in recs}],
                    "feedback_history": [[source_ep, item] for source_ep, item in feedback.history],
                    "feedback_mapping": feedback.mapping,
                    "last_memory": produced_mem,
                    "environment": {"pv": args.pv, "seed": args.seed, "K": args.K,
                                    "warmup": args.warmup, "budget_scale": args.budget_scale,
                                    "conversion_keep_rate": args.conversion_keep_rate,
                                    "environment_profile": args.env_profile, "portfolio": portfolio,
                                    "temperature": args.temperature, "constrained_json": args.constrained_json},
                }, ensure_ascii=False, indent=2), encoding="utf-8")
                memory_path = out_path / f"memory_after_{ep}.json"
                memory_path.write_text(json.dumps(produced_mem, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"[robust:{args.arm}] ep{ep} OK F={rec['F']:.0f} C={rec['C']:.0f} N={rec['N']:.0f} cpa={rec['cpa']} feasible={rec['cpa_feasible']}", flush=True)
            except Exception as e:
                fail = {"episode": ep, "error_type": type(e).__name__, "error": str(e),
                        "traceback": traceback.format_exc()[-4000:]}
                fails.append(fail)
                ff.write(json.dumps(fail, ensure_ascii=False) + "\n"); ff.flush()
                print(f"[robust:{args.arm}] ep{ep} FAIL {type(e).__name__}: {e}", flush=True)
                if args.stop_on_error:
                    break

    recs.sort(key=lambda row: int(row["episode"]))
    if recs:
        F = sum(r["F"] for r in recs); C = sum(r["C"] for r in recs); N = sum(r["N"] for r in recs)
        summary = {args.arm: dict(n=len(recs), requested=len(eps), failed=len(fails),
                                  F=F, C=C, N=N, cpa_port=(C / N if N else None),
                                  feasible_rate=float(np.mean([r["cpa_feasible"] for r in recs])),
                                  illegal=sum(int(r.get("illegal", 0)) for r in recs),
                                  forced=sum(int(r.get("forced", 0)) for r in recs))}
    else:
        summary = {args.arm: dict(n=0, requested=len(eps), failed=len(fails), F=0, C=0, N=0,
                                  cpa_port=None, feasible_rate=0.0, illegal=0, forced=0)}
    summary[args.arm].update(summarize_delivery(recs))
    role_telemetry = {}
    if telemetry is not None:
        for role in ("planner", "optimizer", "inspector"):
            items = [x for x in telemetry if x["role"] == role]
            latency = [x["latency_s"] for x in items]
            role_telemetry[role] = {"calls": len(items), "tokens": sum(x["tokens"] for x in items),
                                    "errors": sum(not x["ok"] for x in items),
                                    "latency_p50_s": float(np.percentile(latency, 50)) if latency else None,
                                    "latency_p95_s": float(np.percentile(latency, 95)) if latency else None}
        with open(os.path.join(args.out, "call_telemetry.jsonl"), "w", encoding="utf-8") as tf:
            for item in telemetry:
                tf.write(json.dumps(item, ensure_ascii=False) + "\n")
    if pacing_trace is not None:
        with open(os.path.join(args.out, "pacing_trace.jsonl"), "w", encoding="utf-8") as pf:
            for item in pacing_trace:
                pf.write(json.dumps(item, ensure_ascii=False) + "\n")
    report = {"arm": args.arm, "episodes": eps, "summary": summary, "failures": fails,
              "ablation_mode": args.ablation_mode, "feedback_mapping": feedback.mapping,
              "pv": args.pv, "budget_scale": args.budget_scale, "portfolio": portfolio,
              "environment_profile": args.env_profile,
              "conversion_keep_rate": args.conversion_keep_rate,
              "sparse_profile": ("extra_sparse_feedback_stress_57_of_1512"
                                 if args.conversion_keep_rate is not None
                                 and args.conversion_keep_rate < 1.0 else None),
              "runtime_s": time.perf_counter() - run_started, "role_telemetry": role_telemetry,
              "temperature": args.temperature, "constrained_json": args.constrained_json,
              "memory_final": feedback.history[-1][1] if feedback.history else None}
    with open(os.path.join(args.out, "eval_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[robust] summary={json.dumps(summary, ensure_ascii=False)}")
    print(f"    → {args.out}/arms.jsonl  {args.out}/failures.jsonl  {args.out}/eval_report.json")


if __name__ == "__main__":
    main()
