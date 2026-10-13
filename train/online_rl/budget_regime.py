"""Helpers for fixed-budget online AdCoEvo adaptation records."""

from __future__ import annotations

from typing import Any, Dict

import math

from train.coevolution.ablation import effective_target


ROLE_SCHEDULES = {
    "round_robin": ("planner", "optimizer", "inspector"),
    "optimizer_only": ("optimizer",),
    "po_alternating": ("planner", "optimizer"),
}


def scheduled_role(episode: int, schedule_origin: int | None,
                   schedule: tuple[str, ...] | None = None) -> str | None:
    index = episode - (0 if schedule_origin is None else schedule_origin)
    active_schedule = schedule or ROLE_SCHEDULES["round_robin"]
    return active_schedule[index % len(active_schedule)] if index >= 0 else None


def scheduled_update_counts(episodes: list[int], schedule_origin: int,
                            role_schedule: str, ablation_mode: str) -> Dict[str, int]:
    """Count the learner records the collector will enqueue for a run.

    Inspector's first record can only be settled by a subsequent period.
    Settled Inspector records are buffered and train only on scheduled rounds.
    """
    if not episodes:
        raise ValueError("at least one episode is required")
    schedule = ROLE_SCHEDULES[role_schedule]
    counts = {"planner": 0, "optimizer": 0, "inspector": 0}
    for position, episode in enumerate(episodes):
        role = effective_target(scheduled_role(episode, schedule_origin, schedule),
                                ablation_mode)
        if role in ("planner", "optimizer"):
            counts[role] += 1
        if position > 0 and role == "inspector":
            counts["inspector"] += 1
    return counts


def episode_record(traj: Any, scheduled: str | None, update_role: str | None,
                   versions_before: Dict[str, int], versions_after: Dict[str, int],
                   feedback_source_episode: int | None, budget_scale: float,
                   illegal: int, forced: int, set_total: int,
                   agent_requests: int = 0, set_requests: int = 0,
                   cpa_tolerance: float = 1.02) -> Dict[str, Any]:
    final = traj.final_metrics
    native = traj.native_metrics
    cpa = float(final["cpa"]) if final.get("cpa") is not None else None
    cpa_native = float(native["cpa"]) if native.get("cpa") is not None else None
    cpa = cpa if cpa is not None and math.isfinite(cpa) else None
    cpa_native = cpa_native if cpa_native is not None and math.isfinite(cpa_native) else None
    feasible = cpa is not None and cpa_native is not None and cpa <= cpa_native * cpa_tolerance
    return {
        "arm": "online_adcoevo", "episode": int(traj.episode), "seed": int(traj.seed),
        "pv": int(traj.pv), "K": int(traj.K), "budget_scale": float(budget_scale),
        "F": float(final["F"]), "C": float(final["C"]), "N": float(final["N"]), "cpa": cpa,
        "cpa_native": cpa_native, "native_F": float(native["F"]),
        "native_C": float(native["C"]), "native_N": float(native["N"]),
        "cpa_feasible": bool(feasible), "viol": 0.0 if feasible else 1.0,
        "illegal": int(illegal), "forced": int(forced), "set_total": int(set_total),
        "agent_requests": int(agent_requests), "set_requests": int(set_requests),
        "scheduled_role": scheduled, "update_role": update_role,
        "policy_versions_before": dict(versions_before),
        "policy_versions_after": dict(versions_after),
        "feedback_source_episode": feedback_source_episode,
    }
