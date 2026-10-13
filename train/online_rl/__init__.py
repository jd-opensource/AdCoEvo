"""Complete asynchronous single-rollout AdCoEvo runtime."""

from train.online_rl.schema import ActionTransition, OnlineTrajectory
__all__ = ["ActionTransition", "OnlineTrajectory", "PrefixValueModel",
           "bilateral_ratio_mask", "skip_observation_gae", "account_isolated_gae"]


def __getattr__(name):
    if name in ("PrefixValueModel", "bilateral_ratio_mask", "skip_observation_gae", "account_isolated_gae"):
        from importlib import import_module
        value = import_module("train.online_rl.value")
        return getattr(value, name)
    raise AttributeError(name)
