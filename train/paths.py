"""Resolve repository paths, external AuctionNet sources and language-model assets."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# AuctionNet is cloned by the user, not vendored in this repository.
#
# `git clone https://github.com/alimama-tech/AuctionNet` produces AuctionNet/ with the
# sources directly inside; downloading the GitHub zip produces AuctionNet/AuctionNet-main/.
# Accept both, plus the source-layout check, so either way works without configuration.
_ENV_AUCTIONNET = "ADCOEVO_AUCTIONNET"
_AUCTIONNET_CANDIDATES = (ROOT / "AuctionNet" / "AuctionNet-main", ROOT / "AuctionNet")
# Note: `git clone` yields AuctionNet/, the GitHub zip yields AuctionNet/AuctionNet-main/.
_AUCTIONNET_MARKER = "simul_bidding_env"


def _looks_like_auctionnet(path: Path) -> bool:
    return (path / _AUCTIONNET_MARKER).is_dir()


def pick_auctionnet_dir(candidates) -> Path:
    """First candidate that actually contains the simulator; else the first candidate."""
    for candidate in candidates:
        if _looks_like_auctionnet(candidate):
            return candidate
    return Path(candidates[0])


def auctionnet_dir(root: Path | None = None) -> Path:
    """Resolve the AuctionNet source root: env override first, then the usual layouts."""
    override = os.environ.get(_ENV_AUCTIONNET)
    if override:
        path = Path(override).expanduser()
        if not path.exists():
            raise SystemExit(
                f"{_ENV_AUCTIONNET} points at {path}, which does not exist. Point it at the "
                f"directory that contains {_AUCTIONNET_MARKER}/.")
        return path
    base = Path(root) if root is not None else ROOT
    return pick_auctionnet_dir((base / "AuctionNet" / "AuctionNet-main", base / "AuctionNet"))


def _auctionnet_unavailable_message(path: Path) -> str:
    searched = "\n".join(f"    {c}" for c in _AUCTIONNET_CANDIDATES)
    return (
        f"AuctionNet source not found at {path}.\n"
        f"AuctionNet is an external dependency and is not bundled here. Clone it:\n"
        f"    git clone https://github.com/alimama-tech/AuctionNet\n"
        f"then either leave it at AuctionNet/ under the repository root, or point "
        f"{_ENV_AUCTIONNET} at the directory containing {_AUCTIONNET_MARKER}/.\n"
        f"Searched:\n{searched}")


_ENV_BASE_MODEL = "ADCOEVO_BASE_MODEL"
_ENV_TOKENIZER = "ADCOEVO_TOKENIZER"
_ENV_SERVED_MODEL = "ADCOEVO_SERVED_MODEL"


def served_model_name(explicit: str | None = None) -> str:
    """Name the local vLLM server exposes.

    It is only an identifier: the collector and the evaluator must use the same string
    as the one passed to ``vllm serve --served-model-name``. Any base model works.
    """
    return explicit or os.environ.get(_ENV_SERVED_MODEL) or "model"


def base_model(explicit: str | None = None) -> Path:
    """Resolve the base language model directory."""
    value = explicit or os.environ.get(_ENV_BASE_MODEL)
    if not value:
        raise SystemExit(
            f"base model path is not configured; pass it explicitly or set {_ENV_BASE_MODEL} "
            "to your own local model directory")
    path = Path(value).expanduser().resolve()
    if not path.exists():
        raise SystemExit(f"base model path does not exist: {path}")
    return path


def tokenizer(explicit: str | None = None) -> Path:
    """Resolve the tokenizer directory, defaulting to the base model."""
    value = explicit or os.environ.get(_ENV_TOKENIZER) or os.environ.get(_ENV_BASE_MODEL)
    if not value:
        raise SystemExit(
            f"tokenizer path is not configured; pass it explicitly or set {_ENV_TOKENIZER} "
            f"or {_ENV_BASE_MODEL}")
    path = Path(value).expanduser().resolve()
    if not path.exists():
        raise SystemExit(f"tokenizer path does not exist: {path}")
    return path


def auctionnet_paths() -> list[str]:
    """sys.path entries required to import the AuctionNet simulator."""
    root = auctionnet_dir()
    return [str(root), str(root / "strategy_train_env")]


def require_auctionnet() -> None:
    path = auctionnet_dir()
    if not _looks_like_auctionnet(path):
        raise SystemExit(_auctionnet_unavailable_message(path))
