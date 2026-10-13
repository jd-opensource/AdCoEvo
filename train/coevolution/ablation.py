"""Collaboration-ablation controls for continuous co-evolution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

MODES = ("full", "frozen_partners", "no_inspector_feedback", "shuffled_feedback")
EMPTY_FEEDBACK = {
    "planner_guidance": "",
    "optimizer_guidance": "",
    "issue": "",
    "has_issue": False,
    "updated_after_ep": None,
}


def effective_target(scheduled_role: Optional[str], mode: str) -> Optional[str]:
    if mode not in MODES:
        raise ValueError(f"unknown ablation mode: {mode}")
    if mode == "frozen_partners" and scheduled_role is not None:
        return "optimizer"
    return scheduled_role


@dataclass
class FeedbackRouter:
    mode: str
    seed: int
    history: list[Tuple[int, Dict[str, Any]]] = field(default_factory=list)
    mapping: list[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"unknown ablation mode: {self.mode}")

    def consume(self, episode: int) -> Tuple[Optional[Dict[str, Any]], Optional[int]]:
        source = None
        feedback: Optional[Dict[str, Any]] = None
        if self.mode in ("full", "frozen_partners") and self.history:
            source, feedback = self.history[-1]
        elif self.mode == "shuffled_feedback":
            # Exclude the most recent correction so shuffled feedback cannot collapse to Full.
            # Selection is deterministic and only considers strictly earlier, already-produced feedback.
            candidates = self.history[:-1]
            if candidates:
                source, feedback = candidates[(self.seed + episode) % len(candidates)]
            else:
                feedback = dict(EMPTY_FEEDBACK)
        self.mapping.append({"consumer_episode": episode, "source_episode": source,
                             "mode": self.mode, "empty": feedback is not None and source is None})
        return feedback, source

    def produce(self, episode: int, feedback: Dict[str, Any]) -> None:
        self.history.append((episode, dict(feedback)))
