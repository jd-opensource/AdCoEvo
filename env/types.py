"""Small result container for a settled AuctionNet simulator tick."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

State = Dict[str, Any]
Outcome = Dict[str, Any]


@dataclass
class StepResult:
    outcome: Dict[str, Outcome]
    next_state: State
    done: bool = False
    info: Dict[str, Any] = field(default_factory=dict)

    def aggregated(self) -> Outcome:
        return self.info.get("aggregated", {})
