"""Pure helpers for prefix-value learner reward attribution and learner diagnostics."""

from __future__ import annotations

import math
from typing import Dict, Iterable


def metric_delta(after: Dict, before: Dict) -> Dict[str, float]:
    return {"F": float(after["fix"] - before["fix"]),
            "C": float(after["cost"] - before["cost"]),
            "N": float(after["conv"] - before["conv"]),
            "M": float(after["target_cost"] - before["target_cost"])}


def scalar_reward(delta: Dict[str, float], cost_margin_weight: float = 1.0) -> float:
    """Q = (F - eta * (C - rho * N)) / s, using M = rho * N and s = 10000."""
    return (float(delta["F"]) - cost_margin_weight *
            (float(delta["C"]) - float(delta["M"]))) / 10000.0


def attributed_reward(account_delta: Dict[str, float], portfolio_delta: Dict[str, float],
                      portfolio_weight: float = 0.2,
                      cost_margin_weight: float = 1.0) -> Dict[str, float]:
    account = scalar_reward(account_delta, cost_margin_weight)
    portfolio = scalar_reward(portfolio_delta, cost_margin_weight)
    return {"account_reward": account, "portfolio_reward": portfolio,
            "portfolio_weight": float(portfolio_weight),
            "combined_reward": account + float(portfolio_weight) * portfolio}


def normalize_advantages(values: Iterable[float], clip: float = 3.0) -> list[float]:
    vals = [float(x) for x in values]
    if not vals:
        return []
    mean = sum(vals) / len(vals)
    std = math.sqrt(sum((x - mean) ** 2 for x in vals) / len(vals))
    if std < 1e-9:
        return [0.0 for _ in vals]
    return [max(-clip, min(clip, (x - mean) / std)) for x in vals]


def role_advantages(role: str, gae_values: Iterable[float], rewards: Iterable[float],
                    inspector_baseline: float = 0.0) -> list[float]:
    gae = [float(x) for x in gae_values]; reward = [float(x) for x in rewards]
    if role == "optimizer":
        return normalize_advantages(gae)
    if role == "planner":
        return normalize_advantages(reward)
    if role == "inspector":
        return [max(-3.0, min(3.0, x - inspector_baseline)) for x in reward]
    raise ValueError(role)


def adaptive_epoch_count(advantages: Iterable[float], max_epochs: int = 4) -> int:
    vals = [abs(float(x)) for x in advantages]
    strength = sum(vals) / len(vals) if vals else 0.0
    if strength < 0.25:
        return 1
    if strength < 0.75:
        return min(2, max_epochs)
    return max_epochs


def explained_variance(predictions: Iterable[float], targets: Iterable[float]) -> float:
    pred = [float(x) for x in predictions]; target = [float(x) for x in targets]
    if len(pred) != len(target) or not target:
        return float("nan")
    mean = sum(target) / len(target)
    var = sum((x - mean) ** 2 for x in target) / len(target)
    if var < 1e-12:
        return float("nan")
    residual = sum(((t - p) - sum(t0 - p0 for t0, p0 in zip(target, pred)) / len(target)) ** 2
                   for t, p in zip(target, pred)) / len(target)
    return 1.0 - residual / var


def correlation(xs: Iterable[float], ys: Iterable[float]) -> float:
    x = [float(v) for v in xs]; y = [float(v) for v in ys]
    if len(x) != len(y) or len(x) < 2:
        return float("nan")
    mx = sum(x) / len(x); my = sum(y) / len(y)
    vx = sum((v - mx) ** 2 for v in x); vy = sum((v - my) ** 2 for v in y)
    if vx < 1e-12 or vy < 1e-12:
        return float("nan")
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / math.sqrt(vx * vy)
