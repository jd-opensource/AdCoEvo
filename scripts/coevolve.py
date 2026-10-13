#!/usr/bin/env python3
"""AdCoEvo one-click co-evolution entry point.

Select the market profile and the post-training method, then run a continuous
co-evolution sequence:

    Planner / Optimizer / Inspector
      -> factual rollout on AuctionNet
      -> scheduled role update
      -> new version published for the next period

Examples:

    # what is available
    python scripts/coevolve.py --list-methods

    # inspect the resolved configuration without starting anything
    python scripts/coevolve.py --env-profile general --method adcoevo --print-config

    # run one 50-period market sequence from externally supplied adapters
    python scripts/coevolve.py \
        --env-profile sparse --method adcoevo \
        --episodes 0-49 --pv 500000 --seed 20270001 \
        --run-dir runs/demo_adcoevo --gpu 0 --port 8001 \
        --inspector-stats /path/to/inspector_reward_stats.json \
        --planner-init /path/to/planner_adapter \
        --optimizer-init /path/to/optimizer_adapter \
        --inspector-init /path/to/inspector_adapter
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DEFAULT_THRESHOLDS = "configs/srppo_business_pass_v1.json"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def adapter_fingerprint(path: str) -> str:
    target = Path(path)
    if target.is_dir():
        weight = target / "adapter_model.safetensors"
        if weight.exists():
            return sha256_of(weight)
    elif target.is_file():
        return sha256_of(target)
    return "missing"


def parse_episodes(value: str) -> list[int]:
    if "-" in value:
        lo, hi = value.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(item) for item in value.split(",")]


def launch_flags(args, spec) -> list[str]:
    """Method-specific flags consumed by scripts/online_rl/launch.py."""
    flags: list[str] = []
    for key, value in spec.launch_args.items():
        if key == "prefix_value":
            flags.append("--prefix-value")
            continue
        override = args.learner_lr if key == "learner_lr" else (
            args.beta_kl if key == "beta_kl" else None)
        flags += [f"--{key.replace('_', '-')}", str(override if override is not None else value)]
    if spec.surrogate:
        flags += ["--surrogate", spec.surrogate]
    return flags


def learner_flags(args, spec) -> list[str]:
    """Explicit learner routing for SPO / SR-PPO style learners."""
    if spec.builtin_learner:
        return []
    flags = ["--learner-script", spec.learner]
    values = dict(spec.learner_args)
    if args.learner_lr is not None:
        values["actor_lr" if spec.name == "srppo" else "lr"] = str(args.learner_lr)
    if args.beta_kl is not None:
        values["beta_kl"] = str(args.beta_kl)
    if spec.needs_thresholds:
        flags += ["--learner-arg", f"thresholds={args.thresholds}"]
    for key in sorted(values):
        flags += ["--learner-arg", f"{key}={values[key]}"]
    return flags


def gate_script(base_model: str, args, run_dir: Path, versions: dict) -> list[str]:
    import shutil
    vllm = shutil.which("vllm") or str(Path(sys.executable).parent / "vllm")
    policies = run_dir / "policies"
    lora = []
    for role in ("planner", "optimizer", "inspector"):
        path = policies / role / f"v{int(versions[role]):06d}"
        lora += [f"{role}={path.resolve()}"]
    return [vllm, "serve", base_model, "--served-model-name", args.served_model_name,
            "--port", str(args.port + 1), "--dtype", "bfloat16", "--max-model-len", "8192",
            "--gpu-memory-utilization", "0.35", "--enable-lora", "--max-lora-rank", "16",
            "--lora-modules", *lora, "--tensor-parallel-size", "1",
            "--attention-backend", "TRITON_ATTN"]


def main() -> None:
    from env.auctionnet_profile import available_profiles, load_profile
    from train.coevolution.method_catalog import describe, method_names, resolve
    from train.online_rl.budget_regime import ROLE_SCHEDULES, scheduled_update_counts

    parser = argparse.ArgumentParser(
        description="Run AdCoEvo co-evolution across consecutive delivery periods.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list-methods", action="store_true", help="list available post-training methods")
    parser.add_argument("--print-config", action="store_true",
                        help="print the resolved configuration and exit without launching")
    parser.add_argument("--env-profile", choices=list(available_profiles()), default="sparse",
                        help="AuctionNet market profile")
    parser.add_argument("--method", default="adcoevo", help=f"one of: {', '.join(method_names())}")
    parser.add_argument("--episodes", default="0-49",
                        help="consecutive delivery periods; one market sequence is 50")
    parser.add_argument("--role-schedule", default="round_robin",
                        choices=list(ROLE_SCHEDULES),
                        help="update Planner/Optimizer/Inspector in turn, only Optimizer, "
                             "or alternate Planner/Optimizer; all roles execute delivery")
    parser.add_argument("--pv", type=int, default=500000)
    parser.add_argument("--budget-scale", type=float, default=1.0,
                        help="multiply the controlled accounts' starting budgets")
    parser.add_argument("--window", type=int, default=12,
                        help="warm-up ticks W: first W ticks run native budget + all HOLD, "
                             "agent decisions start at tick W")
    parser.add_argument("--quota", type=int, default=8,
                        help="maximum tCPA changes per unit and period; additional SET actions become HOLD")
    parser.add_argument("--seed", type=int, default=20270001)
    parser.add_argument("--served-model-name", default=None,
                        help="identifier the local vLLM server exposes; any base model works "
                             "(env ADCOEVO_SERVED_MODEL, default 'model')")
    parser.add_argument("--run-dir", default="runs/coevolve_demo")
    parser.add_argument("--resume", action="store_true",
                        help="resume a run from its existing run-dir and saved collector state")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--planner-init", default=None)
    parser.add_argument("--optimizer-init", default=None)
    parser.add_argument("--inspector-init", default=None)
    parser.add_argument("--inspector-stats", default=None,
                        help="JSON with frozen training-set mu_F, sigma_F, mu_N and sigma_N")
    parser.add_argument("--thresholds", default=DEFAULT_THRESHOLDS,
                        help="business-pass thresholds required by the SR-PPO method")
    parser.add_argument("--learner-lr", type=float, default=None)
    parser.add_argument("--beta-kl", type=float, default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--rollout-gpu", type=int, default=None)
    parser.add_argument("--learner-gpu", type=int, default=None)
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--server-startup-timeout", type=float, default=1200)
    parser.add_argument("--gate", action="store_true",
                        help="after co-evolution, freeze the final versions and evaluate them")
    parser.add_argument("--gate-episodes", default=None,
                        help="disjoint periods for optional frozen evaluation (default: six after training)")
    args = parser.parse_args()

    if args.list_methods:
        for row in describe():
            print(f"{row['method']:>6}  [{row['collection']} / {row['advantage']}]")
            print(f"        learner  : {row['learner']}")
            print(f"        surrogate: {row['surrogate'] or 'n/a'}")
            print(f"        {row['summary']}")
        return

    try:
        spec = resolve(args.method)
    except KeyError as exc:
        raise SystemExit(str(exc))
    if spec.name == "adcoevo" and args.beta_kl is not None:
        raise SystemExit("--beta-kl is not used by adcoevo; KL only controls monitoring and rollback")
    profile = load_profile(args.env_profile)
    episodes = parse_episodes(args.episodes)
    if not episodes or episodes != sorted(set(episodes)):
        raise SystemExit("--episodes must be an increasing, nonempty sequence without duplicates")
    if args.budget_scale <= 0:
        raise SystemExit("--budget-scale must be positive")
    if args.gate_episodes is None:
        args.gate_episodes = f"{episodes[-1] + 1}-{episodes[-1] + 6}"
    if args.gate and set(episodes).intersection(parse_episodes(args.gate_episodes)):
        raise SystemExit("--gate-episodes must not overlap co-evolution episodes")
    from train.paths import base_model as resolve_base_model, served_model_name
    base_model = (os.environ.get("ADCOEVO_BASE_MODEL", "<not configured>")
                  if args.print_config else str(resolve_base_model()))
    args.served_model_name = served_model_name(args.served_model_name)
    rollout_gpu = args.gpu if args.rollout_gpu is None else args.rollout_gpu
    learner_gpu = args.gpu if args.learner_gpu is None else args.learner_gpu

    if spec.needs_thresholds and not Path(args.thresholds).exists():
        raise SystemExit(f"--thresholds file not found: {args.thresholds}")

    launch_cmd = [
        sys.executable, "scripts/online_rl/launch.py",
        "--run-dir", args.run_dir,
        "--episodes", args.episodes,
        "--pv", str(args.pv),
        "--budget-scale", str(args.budget_scale),
        "--seed", str(args.seed),
        "--K", str(args.quota),
        "--window", str(args.window),
        "--env-profile", args.env_profile,
        "--temperature", str(args.temperature),
        "--served-model-name", args.served_model_name,
        "--collector-out", str(Path(args.run_dir) / "eval"),
        "--role-schedule", args.role_schedule,
        "--schedule-origin", str(episodes[0]),
        "--wait-for-role-update",
        "--rollout-gpu", str(rollout_gpu),
        "--optimizer-gpu", str(learner_gpu),
        "--planner-gpu", str(learner_gpu),
        "--inspector-gpu", str(learner_gpu),
        "--port", str(args.port),
        "--server-startup-timeout", str(args.server_startup_timeout),
    ]
    launch_cmd += launch_flags(args, spec)
    launch_cmd += learner_flags(args, spec)
    if args.inspector_stats:
        launch_cmd += ["--inspector-stats", args.inspector_stats]
    if args.resume:
        launch_cmd.append("--resume")
    for flag, value in (("--planner-init", args.planner_init),
                        ("--optimizer-init", args.optimizer_init),
                        ("--inspector-init", args.inspector_init)):
        if value:
            launch_cmd += [flag, value]

    resolved = {
        "environment_profile": args.env_profile,
        "calibration_id": profile.calibration_id,
        "pvalue_mean_base": profile.pvalue_mean_base,
        "total_budget": sum(profile.budgets),
        "cpa_constraint_range": [min(profile.cpa_constraints), max(profile.cpa_constraints)],
        "method": spec.name,
        "collection": spec.collection,
        "advantage": spec.advantage,
        "surrogate": spec.surrogate,
        "learner": spec.learner,
        "schedule": list(ROLE_SCHEDULES[args.role_schedule]),
        "role_schedule": args.role_schedule,
        "scheduled_updates": scheduled_update_counts(episodes, episodes[0],
                                                       args.role_schedule, "full"),
        "episodes": episodes,
        "pv": args.pv,
        "budget_scale": args.budget_scale,
        "warmup_window_w": args.window,
        "set_quota_k": args.quota,
        "seed": args.seed,
        "temperature": args.temperature,
        "rollout_gpu": rollout_gpu,
        "learner_gpu": learner_gpu,
        "run_dir": args.run_dir,
        "resume": args.resume,
        "gate_episodes": args.gate_episodes if args.gate else None,
        "base_model": base_model,
        "served_model_name": args.served_model_name,
        "inspector_stats": args.inspector_stats,
        "inspector_stats_sha256": (sha256_of(Path(args.inspector_stats))
                                   if args.inspector_stats and Path(args.inspector_stats).is_file()
                                   else None),
        "launch_command": launch_cmd,
    }
    if args.print_config:
        print(json.dumps(resolved, ensure_ascii=False, indent=2))
        return

    if not args.inspector_stats or not Path(args.inspector_stats).is_file():
        raise SystemExit("--inspector-stats must name a JSON file of frozen training-set statistics")

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    resolved["initial_adapter_sha256"] = {
        role: adapter_fingerprint(value)
        for role, value in (("planner", args.planner_init), ("optimizer", args.optimizer_init),
                            ("inspector", args.inspector_init)) if value
    }
    (run_dir / "coevolve_manifest.json").write_text(
        json.dumps(resolved, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[coevolve] profile={args.env_profile} method={spec.name} run={run_dir}", flush=True)

    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(Path.cwd()))
    rc = subprocess.call(launch_cmd, env=env)
    if rc != 0:
        raise SystemExit(f"co-evolution launcher failed with exit code {rc}")

    from train.evaluation_metrics import summarize_delivery
    rows = [json.loads(line) for line in (run_dir / "eval" / "arms.jsonl").read_text().splitlines()
            if line.strip()]
    (run_dir / "eval" / "summary.json").write_text(
        json.dumps(summarize_delivery(rows), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    versions = json.loads((run_dir / "policies" / "manifest.json").read_text())["versions"]
    print(f"[coevolve] final versions: {versions}", flush=True)

    if args.gate:
        serve = subprocess.Popen(gate_script(base_model, args, run_dir, versions))
        try:
            import time
            import urllib.request
            url = f"http://127.0.0.1:{args.port + 1}/v1/models"
            for _ in range(600):
                try:
                    urllib.request.urlopen(url, timeout=2)
                    break
                except Exception:
                    time.sleep(2)
            else:
                raise SystemExit("gate: vLLM startup timeout")
            gate_out = run_dir / "gate"
            gate_out.mkdir(parents=True, exist_ok=True)
            subprocess.check_call([
                sys.executable, "scripts/evaluate.py",
                "--episodes", args.gate_episodes, "--env-profile", args.env_profile,
                "--temperature", str(args.temperature), "--pv", str(args.pv), "--seed", str(args.seed),
                "--budget-scale", str(args.budget_scale),
                "--window", str(args.window), "--quota", str(args.quota),
                "--base-url", f"http://127.0.0.1:{args.port + 1}/v1",
                "--model", args.served_model_name,
                "--constrained-json",
                "--planner-model", "planner", "--optimizer-model", "optimizer",
                "--inspector-model", "inspector", "--out", str(gate_out)], env=env)
            print(f"[coevolve] gate results -> {gate_out}", flush=True)
        finally:
            serve.terminate()
            try:
                serve.wait(timeout=15)
            except subprocess.TimeoutExpired:
                serve.kill()


if __name__ == "__main__":
    main()
