# Contributing to AdCoEvo

Contributions to code, documentation, and reproducibility instructions are welcome. Please open an issue to describe a bug or discuss a substantial change before implementing it.

## Development setup

Use Python 3.10. Clone the full AuctionNet simulator into the AdCoEvo repository root, or point `ADCOEVO_AUCTIONNET` to an existing AuctionNet checkout:

```bash
git clone https://github.com/jd-opensource/AdCoEvo.git adcoevo
cd adcoevo
git clone https://github.com/alimama-tech/AuctionNet
conda create -n adcoevo python=3.10 -y
conda activate adcoevo
pip install -r requirements.txt
```

Install PyTorch and vLLM for your CUDA version when using a GPU. To run the continuous delivery and update loop, also provide a compatible causal language model, initial LoRA adapters for Planner, Optimizer, and Inspector, and frozen Inspector reward statistics, as described in [README.md](README.md).

## Before submitting a pull request

1. Make a focused change on a separate branch. Put implementation changes in the relevant `agents/`, `env/`, `train/`, or `scripts/` module.
2. Verify the affected workflow and run `git diff --check`. Describe your verification and any limitations in the pull request.
3. Update documentation when command-line options, environment profiles, output formats, or assumptions change. Keep [README.md](README.md) and [README.zh-CN.md](README.zh-CN.md) consistent.
4. Open a pull request that explains the change, its motivation, and how you verified it. For a bug fix, include reproduction steps.

Do not commit API keys, private business data, model weights, training trajectories, or third-party datasets. The full AuctionNet simulator is obtained separately. If you adapt third-party code, identify its source and license and update [NOTICE](NOTICE). Contributions to this repository are made under its [Apache License 2.0](LICENSE).

中文说明见 [CONTRIBUTING.zh-CN.md](CONTRIBUTING.zh-CN.md)。
