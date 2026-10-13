"""Policy surrogates for single-rollout learners."""

from __future__ import annotations

import torch


def policy_loss(kind: str, new_lp: torch.Tensor, old_lp: torch.Tensor, advantage: float,
                low: float = 0.8, high: float = 1.2, tau: float = 8.0):
    ratio=torch.exp(new_lp-old_lp); adv=torch.as_tensor(float(advantage),device=new_lp.device)
    if kind in ("token_ratio_mask","cispo_clipped_is"):
        mask=((ratio>=low)&(ratio<=high)).float().detach()
        w=torch.clamp(ratio.detach(),low,high)
        loss=-(mask*w*adv*new_lp).sum()/mask.sum().clamp_min(1.0)
    elif kind=="ppo_clip":
        s1=ratio*adv; s2=torch.clamp(ratio,low,high)*adv; loss=-torch.minimum(s1,s2).mean(); mask=torch.ones_like(ratio)
    elif kind=="gspo_sequence_clip":
        r=torch.exp((new_lp-old_lp).mean()); loss=-torch.minimum(r*adv,torch.clamp(r,low,high)*adv); mask=torch.ones_like(ratio)
    elif kind=="sapo_soft_gate":
        gate=torch.sigmoid(tau*(ratio-low))*torch.sigmoid(tau*(high-ratio)); loss=-(gate*ratio*adv).mean(); mask=gate.detach()
    elif kind=="dapo_asymmetric_clip":
        r=torch.exp((new_lp-old_lp).mean()); upper=max(high,1.28); loss=-torch.minimum(r*adv,torch.clamp(r,low,upper)*adv); mask=torch.ones_like(ratio)
    else: raise KeyError(kind)
    return loss,ratio,mask


def backward_mean_valid_responses(losses_and_masks, parameters):
    """Backpropagate the mean loss over responses with at least one retained token.

    Each response is backpropagated immediately so its computation graph can be
    released before the next response is evaluated. Scaling the accumulated
    gradients afterward excludes all-zero masks from the response denominator.
    """
    valid_count = 0
    loss_total = 0.0
    for loss, mask in losses_and_masks:
        if not bool(mask.any().item()):
            continue
        loss.backward()
        valid_count += 1
        loss_total += float(loss.detach())
    if valid_count:
        for parameter in parameters:
            if parameter.grad is not None:
                parameter.grad.div_(valid_count)
    return (loss_total / valid_count if valid_count else 0.0), valid_count


def behavior_kl_estimate(new_lp: torch.Tensor, old_lp: torch.Tensor) -> torch.Tensor:
    """Nonnegative KL diagnostic on sampled completion tokens."""
    log_ratio = new_lp.float() - old_lp.float()
    return (torch.expm1(log_ratio) - log_ratio).mean()
