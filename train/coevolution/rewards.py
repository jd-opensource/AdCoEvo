"""Settled delivery feedback and delayed credit for Inspector guidance."""

from __future__ import annotations

import math
from typing import Dict


WEIGHTS = {"adv": 1.0, "conversions": 0.2, "cpa_margin": 0.5,
           "forced": 0.1}


def clip(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


def validate_stats(stats: Dict[str, float]) -> None:
    for key in ("mu_F", "sigma_F", "mu_N", "sigma_N"):
        if key not in stats or not math.isfinite(float(stats[key])):
            raise ValueError(f"Inspector reward stats require finite {key}")
    if float(stats["sigma_F"]) < 0 or float(stats["sigma_N"]) < 0:
        raise ValueError("Inspector reward standard deviations must be nonnegative")


def channels(metrics: Dict, stats: Dict[str, float], forced: int = 0,
             set_requests: int = 0, epsilon: float = 1e-8) -> Dict[str, float]:
    """Normalize settled P/O business results and the Optimizer forced-HOLD rate."""
    validate_stats(stats)
    f = float(metrics["F"])
    n = float(metrics["N"])
    c = float(metrics["C"])
    m = float(metrics["M"])
    return {
        "adv": clip((f - float(stats["mu_F"])) /
                    (float(stats["sigma_F"]) + epsilon), -3.0, 3.0),
        "conversions": clip((n - float(stats["mu_N"])) /
                            (float(stats["sigma_N"]) + epsilon), -3.0, 3.0),
        "cpa_margin": clip((m - c) / max(m, epsilon), -3.0, 3.0),
        "forced": -clip(float(forced) / max(set_requests, 1), 0.0, 3.0),
    }


def scalar(ch: Dict[str, float]) -> float:
    return sum(WEIGHTS[key] * ch[key] for key in WEIGHTS)


def recovery(current: Dict[str, float], previous: Dict[str, float]) -> Dict[str, float]:
    """Measure the delivery change following the preceding period's diagnosis."""
    return {key: float(current[key] - previous[key]) for key in WEIGHTS}
