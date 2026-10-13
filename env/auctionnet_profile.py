"""AuctionNet's native Sparse profile and General profile calibrated from public data."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Tuple

import yaml

_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "auctionnet_profiles.yaml"


@dataclass(frozen=True)
class AuctionNetProfile:
    name: str
    calibration_id: str
    calibration_hash: str
    description: str
    pvalue_mean_base: float
    conversion_keep_rate: float
    budgets: Tuple[float, ...]
    cpa_constraints: Tuple[float, ...]


def available_profiles(config_path: Path | None = None) -> Tuple[str, ...]:
    data = _load(config_path)
    return tuple(data["profiles"])


def load_profile(name: str | None = None, config_path: Path | None = None) -> AuctionNetProfile:
    data = _load(config_path)
    profile_name = name or data.get("default_profile", "sparse")
    if profile_name not in data["profiles"]:
        choices = ", ".join(data["profiles"])
        raise ValueError(f"unknown AuctionNet profile {profile_name!r}; choose from {choices}")
    raw = data["profiles"][profile_name]
    budgets = tuple(float(value) for value in raw["budgets"])
    cpas = tuple(float(value) for value in raw["cpa_constraints"])
    if len(budgets) != 48 or len(cpas) != 48:
        raise ValueError(f"profile {profile_name!r} requires 48 budgets and CPA constraints")
    if any(value <= 0 for value in budgets + cpas):
        raise ValueError(f"profile {profile_name!r} contains non-positive parameters")
    pvalue_mean_base = float(raw["pvalue_mean_base"])
    keep_rate = float(raw.get("conversion_keep_rate", 1.0))
    if not 0 < pvalue_mean_base <= 1:
        raise ValueError(f"profile {profile_name!r} has invalid pvalue_mean_base")
    if not 0 < keep_rate <= 1:
        raise ValueError(f"profile {profile_name!r} has invalid conversion_keep_rate")
    calibration_blob = yaml.safe_dump(raw, sort_keys=True).encode("utf-8")
    return AuctionNetProfile(
        name=profile_name,
        calibration_id=str(raw["calibration_id"]),
        calibration_hash=hashlib.sha256(calibration_blob).hexdigest(),
        description=str(raw.get("description", "")),
        pvalue_mean_base=pvalue_mean_base,
        conversion_keep_rate=keep_rate,
        budgets=budgets,
        cpa_constraints=cpas,
    )


def _load(config_path: Path | None) -> dict:
    path = Path(config_path) if config_path else _CONFIG
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data.get("profiles"), dict) or not data["profiles"]:
        raise ValueError(f"no AuctionNet profiles configured in {path}")
    return data
