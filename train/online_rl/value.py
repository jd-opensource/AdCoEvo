"""Learned prefix value model and skip-observation GAE for AdCoEvo."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn

from train.online_rl.schema import ActionTransition


class PrefixValueModel(nn.Module):
    """Small learned V(prefix) on frozen LM hidden-state summaries.

    The LM is never part of this module: callers provide final-token hidden vectors for each
    action prefix, allowing frequent critic updates without duplicating an 8B trainable model.
    """

    def __init__(self, hidden_size: int, value_hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, value_hidden),
                                 nn.Tanh(), nn.Linear(value_hidden, 1))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.net(hidden).squeeze(-1)


@dataclass
class GAEOutput:
    advantages: List[float]
    returns: List[float]
    deltas: List[float]


def skip_observation_gae(rewards: Sequence[float], values: Sequence[float], durations: Sequence[int],
                         terminals: Sequence[bool], gamma: float = 1.0,
                         gae_lambda: float = 0.95) -> GAEOutput:
    """GAE over model action nodes only; environment observation steps are skipped via durations.

    `values` has len(rewards)+1, where the final value is bootstrap V(next action prefix).
    """
    n = len(rewards)
    if len(values) != n + 1 or len(durations) != n or len(terminals) != n:
        raise ValueError("invalid GAE lengths")
    adv = [0.0] * n; delta = [0.0] * n
    running = 0.0
    for i in range(n - 1, -1, -1):
        discount = gamma ** max(int(durations[i]), 0)
        nonterminal = 0.0 if terminals[i] else 1.0
        delta[i] = float(rewards[i]) + discount * float(values[i + 1]) * nonterminal - float(values[i])
        running = delta[i] + discount * gae_lambda * nonterminal * running
        adv[i] = running
    returns = [adv[i] + float(values[i]) for i in range(n)]
    return GAEOutput(advantages=adv, returns=returns, deltas=delta)


def account_isolated_gae(nodes: Sequence[ActionTransition], values: Sequence[float],
                         gamma: float = 1.0, gae_lambda: float = 0.95) -> GAEOutput:
    """Compute role-specific GAE without linking different accounts or episodes.

    Planner and Inspector have one-step rewards. Optimizer nodes form a separate
    time series for each (episode, account); tick gaps replace the collector's
    account-order-dependent durations. Results follow the input node order.
    """
    if len(nodes) != len(values):
        raise ValueError("node/value length mismatch")
    n = len(nodes)
    advantages = [0.0] * n
    returns = [0.0] * n
    deltas = [0.0] * n
    optimizer_groups = defaultdict(list)
    for i, node in enumerate(nodes):
        if node.role == "optimizer":
            optimizer_groups[(node.episode, node.account)].append(i)
        elif node.role in ("planner", "inspector"):
            deltas[i] = float(node.reward) - float(values[i])
            advantages[i] = deltas[i]
            returns[i] = float(node.reward)
        else:
            raise ValueError(f"invalid role: {node.role}")

    for indices in optimizer_groups.values():
        indices.sort(key=lambda i: nodes[i].tick)
        if any(nodes[i].tick == nodes[j].tick for i, j in zip(indices, indices[1:])):
            raise ValueError("duplicate optimizer tick for one episode and account")
        durations = [int(nodes[j].tick) - int(nodes[i].tick)
                     for i, j in zip(indices, indices[1:])]
        durations.append(0)
        terminals = [bool(getattr(nodes[i], "terminal", False)) for i in indices]
        terminals[-1] = True  # The last retained action has no observed successor.
        group = skip_observation_gae(
            [nodes[i].reward for i in indices],
            [values[i] for i in indices] + [0.0],
            durations,
            terminals,
            gamma,
            gae_lambda,
        )
        for position, i in enumerate(indices):
            advantages[i] = group.advantages[position]
            returns[i] = group.returns[position]
            deltas[i] = group.deltas[position]
    return GAEOutput(advantages=advantages, returns=returns, deltas=deltas)


def bilateral_ratio_mask(new_logprobs: torch.Tensor, behavior_logprobs: torch.Tensor,
                          low: float = 0.8, high: float = 1.2) -> Tuple[torch.Tensor, torch.Tensor]:
    if len(new_logprobs) != len(behavior_logprobs):
        raise ValueError("ratio mask token mismatch")
    ratio = torch.exp(new_logprobs - behavior_logprobs)
    mask = ((ratio >= low) & (ratio <= high)).to(new_logprobs.dtype).detach()
    return ratio, mask
