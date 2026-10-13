"""User-facing co-evolution method catalog.

Maps short method names to the complete runtime recipe: collection mode, advantage
estimator, policy surrogate, learner script, role schedule and default
hyper-parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

@dataclass(frozen=True)
class MethodSpec:
    name: str
    summary: str
    collection: str
    advantage: str
    learner: str
    surrogate: str | None = None
    needs_thresholds: bool = False
    # Flags consumed by scripts/online_rl/launch.py itself.
    launch_args: Dict[str, str] = field(default_factory=dict)
    # Flags passed straight to the learner when it is routed explicitly.
    learner_args: Dict[str, str] = field(default_factory=dict)
    # True when the learner is chosen by launch.py from --prefix-value / --surrogate.
    builtin_learner: bool = True


METHODS: Dict[str, MethodSpec] = {
    "adcoevo": MethodSpec(
        name="adcoevo",
        summary="Versioned asynchronous actor with token-level bilateral ratio mask, a "
                "prefix value baseline, KL/mask epoch gating and atomic policy releases.",
        collection="async_single_rollout",
        advantage="gae",
        learner="scripts/online_rl/learner_prefix.py",
        surrogate="token_ratio_mask",
        launch_args={"prefix_value": "", "learner_lr": "1e-6", "value_lr": "5e-4",
                     "actor_epochs": "4", "target_kl": "0.01", "min_mask_frac": "0.4",
                     "critic_warmup_updates": "1", "portfolio_reward_weight": "0.2"},
    ),
    "ppo": MethodSpec(
        name="ppo",
        summary="Synchronous single rollout with GAE and a clipped token-level PPO surrogate.",
        collection="sync_single_rollout",
        advantage="gae",
        learner="scripts/online_rl/learner.py",
        surrogate="ppo_clip",
        launch_args={"learner_lr": "3e-7", "value_lr": "5e-4", "beta_kl": "0.1"},
    ),
    "cispo": MethodSpec(
        name="cispo",
        summary="Asynchronous single rollout with clipped detached importance weights.",
        collection="async_single_rollout",
        advantage="gae",
        learner="scripts/online_rl/learner.py",
        surrogate="cispo_clipped_is",
        launch_args={"learner_lr": "3e-7", "value_lr": "5e-4", "beta_kl": "0.1"},
    ),
    "gspo": MethodSpec(
        name="gspo",
        summary="Asynchronous single rollout with a sequence-level importance ratio and clipping.",
        collection="async_single_rollout",
        advantage="gae",
        learner="scripts/online_rl/learner.py",
        surrogate="gspo_sequence_clip",
        launch_args={"learner_lr": "3e-7", "value_lr": "5e-4", "beta_kl": "0.1"},
    ),
    "spo": MethodSpec(
        name="spo",
        summary="Continuous single-stream policy optimization with a persistent "
                "read-before-write EMA baseline instead of a learned critic.",
        collection="rolling_window",
        advantage="ema_baseline",
        learner="scripts/online_rl/learner_sspo.py",
        builtin_learner=False,
        learner_args={"lr": "1e-6", "beta_kl": "1e-3", "min_batch": "1"},
    ),
    "srppo": MethodSpec(
        name="srppo",
        summary="k=1 single-rollout PPO with a Monte-Carlo prefix credit head that "
                "redistributes terminal business credit over tokens.",
        collection="rolling_window",
        advantage="prefix_credit",
        learner="scripts/online_rl/learner_srppo.py",
        needs_thresholds=True,
        builtin_learner=False,
        learner_args={"actor_lr": "1e-6", "critic_lr": "1e-5", "beta_kl": "1e-3",
                      "min_batch": "1"},
    ),
}

def method_names() -> Tuple[str, ...]:
    return tuple(METHODS)


def resolve(name: str) -> MethodSpec:
    if name not in METHODS:
        choices = ", ".join(METHODS)
        raise KeyError(f"unknown co-evolution method {name!r}; available: {choices}")
    return METHODS[name]


def describe() -> list[dict]:
    return [{"method": spec.name, "collection": spec.collection,
             "advantage": spec.advantage, "surrogate": spec.surrogate,
             "learner": spec.learner, "summary": spec.summary}
            for spec in METHODS.values()]
