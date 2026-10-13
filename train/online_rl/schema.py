"""Data contracts for complete asynchronous single-rollout AdCoEvo."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

SCHEMA_VERSION = "adcoevo-online-v1"
ROLES = ("planner", "optimizer", "inspector")


@dataclass
class ActionTransition:
    role: str
    decision_index: int
    episode: int
    tick: int
    account: str
    prompt_token_ids: List[int]
    completion_token_ids: List[int]
    behavior_token_logprobs: List[float]
    parsed_action: Dict[str, Any]
    action_valid: bool
    reward: float = 0.0
    duration: int = 1
    terminal: bool = False
    observation_before: Dict[str, Any] = field(default_factory=dict)
    outcome_after: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.role not in ROLES:
            raise ValueError(f"invalid role: {self.role}")
        if not self.prompt_token_ids:
            raise ValueError("empty prompt token ids")
        # A malformed or failed model reply still belongs in the settled
        # trajectory for illegal-response accounting, but cannot train Actor.
        if self.action_valid and not self.completion_token_ids:
            raise ValueError("valid action has empty completion token ids")
        if self.action_valid and len(self.completion_token_ids) != len(self.behavior_token_logprobs):
            raise ValueError("valid action has completion/logprob length mismatch")
        if self.duration < 0:
            raise ValueError("duration must be >=0")


@dataclass
class OnlineTrajectory:
    trajectory_id: str
    schema_version: str
    behavior_policy_version: int
    role_policy_versions: Dict[str, int]
    episode: int
    seed: int
    pv: int
    K: int
    transitions: List[ActionTransition]
    final_metrics: Dict[str, float]
    native_metrics: Dict[str, float]
    reward_config: Dict[str, float]
    collector_id: str
    created_at: float

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"schema mismatch: {self.schema_version}")
        if not self.transitions:
            raise ValueError("empty trajectory")
        for t in self.transitions:
            t.validate()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "OnlineTrajectory":
        x = dict(d)
        x["transitions"] = [ActionTransition(**t) for t in x["transitions"]]
        out = OnlineTrajectory(**x)
        out.validate()
        return out


def trajectory_id(episode: int, collector_id: str, seq: int, versions: Dict[str, int]) -> str:
    blob = json.dumps({"ep": episode, "collector": collector_id, "seq": seq, "versions": versions},
                      sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:24]
