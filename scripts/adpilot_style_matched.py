#!/usr/bin/env python3
"""Evaluate a shared AdPilot-style actor with short and long reflection memory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from agents.llm_backend import LLMClient, extract_json
from scripts.evaluate import parse_episodes
from scripts.evaluate_arms import run_episode_arm
from train.evaluation_metrics import summarize_delivery


class ReflectingClient:
    def __init__(self, base_url, model, temperature=0.7):
        self.cli = LLMClient({
            "base_url": base_url,
            "model": model,
            "api_key": "EMPTY",
            "temperature": temperature,
            "max_tokens": 512,
            "timeout": 90,
            "enable_thinking": False,
        })
        self.short = []
        self.long = ""

    def state_dict(self):
        return {"short": self.short, "long": self.long}

    def load_state_dict(self, state):
        self.short = list(state["short"])
        self.long = str(state["long"])

    def complete(self, system, user, guided_json=None, max_tokens=None):
        memory = "\n".join(self.short[-2:])
        prompt = (
            f"LONG_MEMORY={self.long}\nSHORT_MEMORY={memory}\nSTATE={user}\n"
            "Reflect internally on delivery trends, CPA deviation, and unproductive "
            "adjustments. Then return valid JSON in the format required by the system instruction."
        )
        response = self.cli.complete(system, prompt, max_tokens=max_tokens,
                                     guided_json=guided_json)
        if response.ok:
            action = extract_json(response.text)
            self.short.append(f"decision={json.dumps(action, ensure_ascii=False)[:240]}")
            if len(self.short) % 32 == 0:
                self.long = "; ".join(self.short[-4:])[-800:]
        return response


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:9901/v1")
    parser.add_argument("--model", default="adpilot_actor")
    parser.add_argument("--out", default="runs/adpilot_style_demo")
    parser.add_argument("--episodes", default="0-49", type=parse_episodes)
    parser.add_argument("--seed", type=int, default=20270001)
    parser.add_argument("--budget-scale", type=float, default=1.0)
    parser.add_argument("--env-profile", choices=["general", "sparse"], default="sparse")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    client = ReflectingClient(args.base_url, args.model, args.temperature)
    memory = None
    rows = []
    episodes = args.episodes
    arms_path = output / "arms.jsonl"
    state_path = output / "resume_state.json"
    protocol = {
        "episodes": episodes,
        "seed": args.seed,
        "pv": 500000,
        "K": 8,
        "warmup": 12,
        "budget_scale": float(args.budget_scale),
        "environment_profile": args.env_profile,
        "model": args.model,
        "temperature": args.temperature,
    }
    if args.resume and arms_path.exists():
        rows = [json.loads(line) for line in arms_path.read_text(encoding="utf-8").splitlines()
                if line.strip()]
        if rows:
            if not state_path.exists():
                raise SystemExit(f"cannot resume: {state_path} is missing")
            state = json.loads(state_path.read_text(encoding="utf-8"))
            completed = [int(row["episode"]) for row in rows]
            if completed != episodes[:len(completed)] or state["completed_episodes"] != completed:
                raise SystemExit("cannot resume: completed episode sequence does not match")
            if state["protocol"] != protocol:
                raise SystemExit("cannot resume: run configuration does not match")
            memory = state["memory"]
            client.load_state_dict(state["reflector"])

    completed = {row["episode"] for row in rows}
    mode = "a" if args.resume else "w"
    with arms_path.open(mode, encoding="utf-8") as handle:
        for episode in episodes:
            if episode in completed:
                continue
            record, memory = run_episode_arm(
                "full", episode, client, 500000, args.seed, 8, 12, memory,
                budget_scale=args.budget_scale, env_profile=args.env_profile,
            )
            record["arm"] = "adpilot_style"
            rows.append(record)
            completed.add(episode)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            state = {
                "protocol": protocol,
                "completed_episodes": [item for item in episodes if item in completed],
                "memory": memory,
                "reflector": client.state_dict(),
            }
            state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n",
                                  encoding="utf-8")
            (output / f"memory_after_{episode}.json").write_text(
                json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
            )

    rows.sort(key=lambda row: row["episode"])
    count = len(rows)
    summary = summarize_delivery(rows)
    summary.update({
        "method": "AdPilot-style",
        "N_per_period": summary["N"] / count,
        "feasible_rate": sum(row["cpa_feasible"] for row in rows) / count,
        "forced_per_period": summary["forced"] / count,
        "llm_calls_per_period": summary["agent_requests"] / count,
        "llm_tokens_per_period": sum(row["llm_tokens"] for row in rows) / count,
    })
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )


if __name__ == "__main__":
    main()
