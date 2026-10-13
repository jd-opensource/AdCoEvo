#!/usr/bin/env python3
"""Run the three AdCoEvo roles with fixed policies and no learner processes.

The rollout is the existing ``full`` evaluation arm: Planner acts after the
warm-up window, Optimizer acts on each remaining tick, and Inspector's guidance
is carried to the next period. This entry can serve three local LoRA adapters
or use an already-running OpenAI-compatible endpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ROLES = ("planner", "optimizer", "inspector")


def parse_episodes(value: str) -> list[int]:
    try:
        if "-" in value:
            parts = value.split("-")
            if len(parts) != 2:
                raise ValueError
            lo, hi = (int(part) for part in parts)
            episodes = list(range(lo, hi + 1))
        else:
            episodes = [int(part) for part in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("episodes must be N-M or comma-separated integers") from exc
    if not episodes or episodes != sorted(set(episodes)) or episodes[0] < 0:
        raise argparse.ArgumentTypeError("episodes must be increasing, unique, and nonnegative")
    return episodes


def adapter_fingerprint(path: Path) -> str:
    """Record the adapter weights used by a frozen run for resume checks."""
    weight = path / "adapter_model.safetensors"
    if not weight.is_file():
        weight = path / "adapter_model.bin"
    if not weight.is_file():
        raise SystemExit(f"adapter weights not found in {path}")
    digest = hashlib.sha256()
    with weight.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Evaluate fixed Planner, Optimizer, and Inspector policies.")
    ap.add_argument("--episodes", default="0-49", help="consecutive market periods, for example 71-120")
    ap.add_argument("--env-profile", choices=("sparse", "general", "extra_sparse_feedback_stress"),
                    default="sparse")
    ap.add_argument("--pv", type=int, default=500000)
    ap.add_argument("--budget-scale", type=float, default=1.0,
                    help="multiply the controlled accounts' starting budgets")
    ap.add_argument("--seed", type=int, default=20270001)
    ap.add_argument("--window", "--warmup", dest="window", type=int, default=12,
                    help="warm-up ticks W before agent decisions")
    ap.add_argument("--quota", "--K", dest="quota", type=int, default=8,
                    help="maximum tCPA changes per unit and period")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--out", "--run-dir", dest="out", default="runs/frozen_demo")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--print-config", action="store_true", help="show the resolved run without starting it")
    ap.add_argument("--base-url", default=None,
                    help="use an already-running endpoint; omit to serve local adapters with vLLM")
    ap.add_argument("--model", "--served-model-name", dest="model", default=None,
                    help="base model identifier exposed by the endpoint")
    for role in ROLES:
        ap.add_argument(f"--{role}-model", default=None,
                        help=f"endpoint model identifier for {role}")
        ap.add_argument(f"--{role}-init", default=None,
                        help=f"local {role} LoRA adapter directory")
    ap.add_argument("--base-model", default=None, help="local base model directory (or ADCOEVO_BASE_MODEL)")
    ap.add_argument("--api-key-env", default=None, help="environment variable holding endpoint API key")
    ap.add_argument("--constrained-json", action=argparse.BooleanOptionalAction, default=None,
                    help="constrain JSON generation; defaults off for external endpoints and on for local adapters")
    ap.add_argument("--gpu", type=int, default=0, help="GPU for the local vLLM server")
    ap.add_argument("--port", type=int, default=8001, help="port for the local vLLM server")
    ap.add_argument("--server-startup-timeout", type=float, default=1200)
    return ap


def resolve_config(args: argparse.Namespace) -> dict:
    from train.paths import base_model, served_model_name

    episodes = parse_episodes(args.episodes)
    if args.pv <= 0 or args.budget_scale <= 0 or not 0 <= args.window < 48 or args.quota < 0:
        raise SystemExit("--pv and --budget-scale must be positive, --window must be 0..47, and --quota must be nonnegative")
    if args.temperature < 0:
        raise SystemExit("--temperature must be nonnegative")
    if args.server_startup_timeout <= 0 or not 1 <= args.port <= 65535:
        raise SystemExit("--server-startup-timeout and --port must be positive")
    model = served_model_name(args.model)
    external = args.base_url is not None
    constrained_json = not external if args.constrained_json is None else args.constrained_json
    adapter_paths = {}
    adapter_hashes = {}
    if external:
        if args.base_model or any(getattr(args, f"{role}_init") for role in ROLES):
            raise SystemExit("--base-url uses an existing endpoint; omit local model and adapter paths")
        role_models = {role: getattr(args, f"{role}_model") or model for role in ROLES}
        base = None
        base_url = args.base_url.rstrip("/")
    else:
        if args.api_key_env or any(getattr(args, f"{role}_model") for role in ROLES):
            raise SystemExit("--api-key-env and --*-model require --base-url")
        base = str(base_model(args.base_model))
        for role in ROLES:
            raw = getattr(args, f"{role}_init")
            if not raw:
                raise SystemExit(f"--{role}-init is required when serving local adapters")
            path = Path(raw).expanduser().resolve()
            if not path.is_dir():
                raise SystemExit(f"--{role}-init must name an adapter directory: {path}")
            adapter_paths[role] = str(path)
            adapter_hashes[role] = adapter_fingerprint(path)
        role_models = {role: role for role in ROLES}
        base_url = f"http://127.0.0.1:{args.port}/v1"
    if not base_url.endswith("/v1"):
        raise SystemExit("--base-url must point to the OpenAI-compatible /v1 endpoint")

    output = Path(args.out).expanduser().resolve()
    rollout_cmd = [
        sys.executable, str(ROOT / "scripts" / "evaluate_stream.py"),
        "--arm", "full", "--episodes", args.episodes,
        "--env-profile", args.env_profile, "--pv", str(args.pv),
        "--budget-scale", str(args.budget_scale),
        "--seed", str(args.seed), "--warmup", str(args.window), "--K", str(args.quota),
        "--temperature", str(args.temperature), "--out", str(output),
        "--base-url", base_url, "--model", model,
        "--planner-model", role_models["planner"],
        "--optimizer-model", role_models["optimizer"],
        "--inspector-model", role_models["inspector"], "--stop-on-error",
    ]
    rollout_cmd.append("--constrained-json" if constrained_json else "--no-constrained-json")
    if args.api_key_env:
        rollout_cmd += ["--api-key-env", args.api_key_env]
    if args.resume:
        rollout_cmd.append("--resume")
    return {
        "protocol": "frozen_full_roles_v1",
        "updates": 0,
        "episodes": episodes,
        "environment_profile": args.env_profile,
        "pv": args.pv,
        "budget_scale": args.budget_scale,
        "seed": args.seed,
        "warmup_window_w": args.window,
        "set_quota_k": args.quota,
        "temperature": args.temperature,
        "constrained_json": constrained_json,
        "base_model": base,
        "served_model_name": model,
        "role_models": role_models,
        "adapter_paths": adapter_paths,
        "adapter_sha256": adapter_hashes,
        "base_url": base_url,
        "out": str(output),
        "rollout_command": rollout_cmd,
    }


def _resume_identity(config: dict) -> dict:
    keys = ("protocol", "updates", "episodes", "environment_profile", "pv", "budget_scale", "seed",
            "warmup_window_w", "set_quota_k", "temperature", "constrained_json", "base_model",
            "served_model_name", "role_models", "adapter_sha256")
    return {key: config[key] for key in keys}


def prepare_output(config: dict, resume: bool) -> None:
    output = Path(config["out"])
    manifest = output / "frozen_manifest.json"
    if resume:
        if not manifest.is_file():
            raise SystemExit(f"cannot resume: {manifest} is missing")
        previous = json.loads(manifest.read_text(encoding="utf-8"))
        if _resume_identity(previous) != _resume_identity(config):
            raise SystemExit("cannot resume: frozen run settings or adapter weights changed")
    else:
        if (output / "arms.jsonl").exists() or manifest.exists():
            raise SystemExit(f"output already contains a run; use --resume or a fresh --out: {output}")
        output.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def serve_command(config: dict, port: int) -> list[str]:
    executable = shutil.which("vllm") or str(Path(sys.executable).parent / "vllm")
    adapters = [f"{role}={config['adapter_paths'][role]}" for role in ROLES]
    return [executable, "serve", config["base_model"],
            "--served-model-name", config["served_model_name"],
            "--port", str(port), "--dtype", "bfloat16", "--max-model-len", "8192",
            "--gpu-memory-utilization", "0.35", "--enable-lora", "--max-lora-rank", "16",
            "--lora-modules", *adapters, "--tensor-parallel-size", "1",
            "--attention-backend", "TRITON_ATTN"]


def wait_for_server(process: subprocess.Popen, base_url: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise SystemExit(f"vLLM exited with code {process.returncode}; see logs/serve.log")
        try:
            with urllib.request.urlopen(base_url + "/models", timeout=2) as response:
                if response.status == 200:
                    return
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(2)
    raise SystemExit("vLLM startup timeout; see logs/serve.log")


def ensure_port_free(port: int) -> None:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            raise SystemExit(f"port {port} is already in use; choose another --port")
    except OSError:
        pass


def verify_result(config: dict) -> None:
    report_path = Path(config["out"]) / "eval_report.json"
    arms_path = Path(config["out"]) / "arms.jsonl"
    if not report_path.is_file():
        raise SystemExit(f"frozen evaluator did not produce {report_path}")
    if not arms_path.is_file():
        raise SystemExit(f"frozen evaluator did not produce {arms_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    result = report["summary"]["full"]
    rows = [json.loads(line) for line in arms_path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    completed = [int(row["episode"]) for row in rows]
    if (result["n"] != len(config["episodes"]) or completed != config["episodes"]
            or any(row.get("arm") != "full" for row in rows)):
        raise SystemExit(f"frozen evaluation incomplete: {result['n']} of "
                         f"{len(config['episodes'])} periods; see failures.jsonl")


def main() -> None:
    ap = parser()
    args = ap.parse_args()
    try:
        config = resolve_config(args)
    except argparse.ArgumentTypeError as exc:
        ap.error(str(exc))
    if args.print_config:
        print(json.dumps(config, ensure_ascii=False, indent=2))
        return
    prepare_output(config, args.resume)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    server = None
    try:
        if args.base_url is None:
            logs = Path(config["out"]) / "logs"
            logs.mkdir(exist_ok=True)
            ensure_port_free(args.port)
            env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
            with (logs / "serve.log").open("w", encoding="utf-8") as log:
                server = subprocess.Popen(serve_command(config, args.port), cwd=ROOT,
                                          env=env, stdout=log, stderr=subprocess.STDOUT)
                wait_for_server(server, config["base_url"], args.server_startup_timeout)
                rc = subprocess.call(config["rollout_command"], cwd=ROOT, env=env)
        else:
            rc = subprocess.call(config["rollout_command"], cwd=ROOT, env=env)
        if rc != 0:
            raise SystemExit(f"frozen evaluator exited with code {rc}")
        verify_result(config)
        print(f"[evaluate] fixed-policy results: {config['out']}/arms.jsonl", flush=True)
    finally:
        if server is not None:
            server.terminate()
            try:
                server.wait(timeout=15)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()


if __name__ == "__main__":
    main()
