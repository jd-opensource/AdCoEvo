"""Role-independent completion log-probability helper."""

from __future__ import annotations

from typing import Sequence, Tuple

import torch


def completion_logprobs_from_ids(model, prompt_ids: Sequence[int], completion_ids: Sequence[int],
                                 device) -> Tuple[torch.Tensor, int]:
    """Compute completion log probabilities from the recorded token IDs.

    Return (logp[C], C) without retokenizing the recorded prompt or completion.
    Both ID sequences must contain nonnegative integers and be nonempty.
    """
    if not prompt_ids or not completion_ids:
        raise ValueError("completion_logprobs_from_ids: prompt/completion token IDs 不能为空")
    if any((not isinstance(t, int)) or t < 0 for t in list(prompt_ids) + list(completion_ids)):
        raise ValueError("completion_logprobs_from_ids: token ID 必须为非负整数")
    ids = torch.tensor([list(prompt_ids) + list(completion_ids)], device=device)
    out = model(input_ids=ids)
    logp = torch.log_softmax(out.logits[:, :-1, :], dim=-1)
    tok_logp = logp.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    C = len(completion_ids)
    return tok_logp[0, -C:], C
