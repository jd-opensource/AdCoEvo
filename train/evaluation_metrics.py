"""Aggregate settled delivery outcomes using the paper's evaluation metrics."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any


def _nonnegative(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _count(value: Any, name: str) -> int:
    result = _nonnegative(value, name)
    if not result.is_integer():
        raise ValueError(f"{name} must be an integer count")
    return int(result)


def summarize_delivery(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize periods from one method and evaluation configuration.

    CPA and IR use pooled costs/conversions and illegal replies/replies,
    respectively. The matched No Operation outcomes define the Score baseline.
    A method without model calls has IR=0; undefined business ratios are None.
    """
    rows = list(records)
    outcome_keys = ("F", "C", "N", "native_F", "native_C", "native_N")
    outcomes = {key: [] for key in outcome_keys}
    counts = {key: 0 for key in ("illegal", "agent_requests", "forced", "set_total")}
    for row in rows:
        for key in outcome_keys:
            outcomes[key].append(_nonnegative(row[key], key))
        requests = _count(row["agent_requests"] if "agent_requests" in row
                          else row["llm_calls"], "agent_requests")
        illegal = _count(row["illegal"], "illegal")
        if illegal > requests:
            raise ValueError("illegal replies cannot exceed total agent replies")
        counts["illegal"] += illegal
        counts["agent_requests"] += requests
        for key in ("forced", "set_total"):
            counts[key] += _count(row.get(key, 0), key)

    totals = {key: math.fsum(values) for key, values in outcomes.items()}
    n = len(rows)
    adv = totals["F"] / n if n else None
    native_adv = totals["native_F"] / n if n else None
    cpa = totals["C"] / totals["N"] if totals["N"] > 0 else None
    native_cpa = (totals["native_C"] / totals["native_N"]
                  if totals["native_N"] > 0 else None)
    ir = (counts["illegal"] / counts["agent_requests"]
          if counts["agent_requests"] else 0.0 if n else None)
    score = (adv / native_adv * native_cpa / cpa - ir
             if native_adv is not None and native_adv > 0
             and cpa is not None and cpa > 0 and native_cpa is not None else None)
    return {
        "n": n, "F": totals["F"], "C": totals["C"], "N": totals["N"],
        "ADV_per_period": adv, "CPA": cpa, "IR": ir,
        "IR_percent": 100 * ir if ir is not None else None, "Score": score,
        **counts, "native_ADV_per_period": native_adv, "native_CPA": native_cpa,
    }
