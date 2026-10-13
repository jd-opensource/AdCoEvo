"""AuctionNet replay environment, imported on demand."""

__all__ = ["AuctionNetSimEnv"]


def __getattr__(name: str):
    if name != "AuctionNetSimEnv":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    return importlib.import_module(".auctionnet_sim_env", __name__).AuctionNetSimEnv


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(globals()))
