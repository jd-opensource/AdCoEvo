# AdCoEvo

AdCoEvo is a co-evolving multi-agent framework for advertising delivery. Planner allocates budgets among ad units within an account, Optimizer adjusts unit-level control tCPA, and Inspector diagnoses settled outcomes to guide planning and delivery adjustments in the next period.

LangGraph orchestrates collaboration within each period. The framework updates model parameters from settled feedback across consecutive periods, using AuctionNet as the simulation environment and its Abid bidder for per-impression bidding.

[中文说明](README.zh-CN.md)

## Installation

```bash
git clone https://github.com/jd-opensource/AdCoEvo.git adcoevo
cd adcoevo

git clone https://github.com/alimama-tech/AuctionNet

conda create -n adcoevo python=3.10 -y
conda activate adcoevo
pip install -r requirements.txt
```

Use PyTorch and vLLM versions compatible with your GPU and CUDA environment. AuctionNet is resolved from `AuctionNet/` under the repository root. Set `ADCOEVO_AUCTIONNET` to use another location.

## Environment and action interfaces

The default setup controls three accounts with four ad units each. Each period contains 48 ticks and 500,000 traffic opportunities.

| Profile | pValue generation base | Initial tCPA range | Total budget of 48 units |
|---|---:|---:|---:|
| `sparse` | 0.0005 | 60–130 | 150,000 |
| `general` | 0.005 | 6–12 | 150,000 |

Sparse retains AuctionNet's native simulator parameters. General configures the pValue generation base, 48 unit budgets, and initial tCPA values from public General data, retaining the native auction mechanisms. See the [environment guide](docs/environment_profiles.md) for settings and provenance.

| Flag | Meaning | Default |
|---|---|---:|
| `--window W` | Observe the first W ticks with initial budgets and tCPA; agent decisions follow | 12 |
| `--quota K` | Maximum tCPA changes per unit and period | 8 |
| `--budget-scale` | Multiplier for controlled accounts' initial budgets | 1.0 |
| `--temperature` | Model sampling temperature | 0.7 |

Planner outputs remaining-budget proportions as `alpha`. Optimizer outputs `HOLD` or a tCPA multiplier from `0.5, 0.6, 0.7, 0.8, 1.0, 1.2, 1.3`. Setting the current multiplier is treated as HOLD; SET actions beyond the adjustment limit become forced HOLD. Inspector receives the system's settled outcomes and action records and produces diagnosis and guidance.

The Rule baseline selects unit-budget multipliers from observed CPA and normalizes the resulting budgets to the account total. Its mapping is implemented separately from Planner's remaining-budget proportions.
During delivery, Rule sets the tCPA multiplier to `0.7` when CPA exceeds 1.10 times the target, to `1.3` when CPA falls below 0.90 times the target, and otherwise uses `HOLD`. It also uses `HOLD` when no mature conversions are available or the adjustment limit is reached.

## Co-evolution

Provide a local causal language model, three compatible role LoRA adapters, and frozen Inspector reward statistics. This repository includes continuous delivery and policy updating, without model weights or offline training pipelines.

```bash
export ADCOEVO_BASE_MODEL=/path/to/base-model
# Set this only when the tokenizer is stored elsewhere.
export ADCOEVO_TOKENIZER=/path/to/tokenizer
```

Compute the mean and standard deviation of ADV and conversions from historical settled trajectories matching the selected environment. Save them as JSON and keep them fixed during the run. The following illustrates the field format; replace the numbers with your actual statistics:

```json
{"mu_F": 0.0, "sigma_F": 1.0, "mu_N": 0.0, "sigma_N": 1.0}
```

```bash
python scripts/coevolve.py \
  --env-profile sparse --method adcoevo \
  --episodes 0-49 --window 12 --quota 8 --temperature 0.7 \
  --run-dir runs/coevolve_demo --gpu 0 --port 8001 \
  --inspector-stats /path/to/inspector_reward_stats.json \
  --planner-init /path/to/planner-lora \
  --optimizer-init /path/to/optimizer-lora \
  --inspector-init /path/to/inspector-lora
```

Each period proceeds through observation, budget planning, tick-level delivery adjustments, and diagnosis. Settled feedback enters the corresponding role buffers. Training rotates in Planner → Optimizer → Inspector order, and newly published policies take effect in the next period. For the default `adcoevo` method, a role's first scheduled update trains its value model; subsequent updates also train policy parameters.

`--role-schedule optimizer_only` updates only Optimizer each period; `po_alternating` alternates Planner and Optimizer updates. Both retain all three roles in delivery and diagnosis.

```bash
python scripts/coevolve.py --list-methods
python scripts/coevolve.py --env-profile general --method adcoevo --print-config
```

| Method | Update mechanism |
|---|---|
| `adcoevo` | PrefixValueModel, role-specific advantages, token importance-ratio filtering, and KL rollback |
| `ppo` | Token-level clipped PPO |
| `cispo` | Clipped, stop-gradient importance weights |
| `gspo` | Sequence-level importance ratios and clipping |
| `spo` | Persistent EMA historical baseline |
| `srppo` | Monte-Carlo prefix credit |

Each method collects one executed trajectory per period. The `gspo` entry uses sequence-level importance ratios and clipping with role-specific advantages during continuous updates. Methods using PrefixValueModel compute reward-minus-value advantages for Planner and Inspector, and GAE separately for each account and period for Optimizer. The default `adcoevo` actor optimizes only its policy loss; KL is used for monitoring and rollback. See the [method guide](docs/methods.md) for details.

## Fixed policies and general-purpose models

The fixed-policy entry point runs the same collaborative workflow, carries diagnosis guidance across periods, and leaves model parameters unchanged:

```bash
python scripts/evaluate.py \
  --env-profile sparse --episodes 0-49 \
  --window 12 --quota 8 --temperature 0.7 \
  --out runs/frozen_demo --gpu 0 --port 8001 \
  --planner-init /path/to/planner-lora \
  --optimizer-init /path/to/optimizer-lora \
  --inspector-init /path/to/inspector-lora
```

An existing OpenAI-compatible API can be used without local models or adapters:

```bash
python scripts/evaluate.py \
  --base-url https://your-endpoint/v1 --model your-model \
  --api-key-env YOUR_API_KEY \
  --env-profile sparse --episodes 0-49 \
  --temperature 0.7 --out runs/api_demo
```

`YOUR_API_KEY` names the environment variable holding your key. Use `--planner-model`, `--optimizer-model`, and `--inspector-model` to select separate role models.

External APIs use prompt-specified JSON by default; local adapters use constrained JSON generation. Select either mode explicitly with `--constrained-json` or `--no-constrained-json`. Both use the same response parser and action validation. Fixed-policy runs do not require Inspector reward statistics.

## Results and resuming

Continuous runs save period records to `eval/arms.jsonl` and aggregates to `eval/summary.json`. Fixed-policy runs save `arms.jsonl` and `eval_report.json`. Combine market windows from the same method, environment, and budget as follows:

```bash
python scripts/summarize_results.py \
  --input runs/window_a/eval/arms.jsonl runs/window_b/eval/arms.jsonl \
  --out runs/overall_summary.json
```

ADV is averaged across periods. CPA is total cost divided by total mature conversions. Invalid output rate `IR` is the number of invalid replies divided by all model replies. The composite score is:

```text
Score = (ADV / ADV_0) * (CPA_0 / CPA) - IR
```

Reference metrics come from a separate No Operation rollout using the same market, environment, and budget and are used only for evaluation. IR is a fraction in the formula; `IR_percent` is its percentage representation. Undefined CPA or Score values are reported as `null`.

Keep the full run directory after interruption and rerun the same command with `--resume`. AuctionNet's main market randomness is determined by episode identifiers. Compare methods on identical episode sequences and use different windows to evaluate different markets. See the [run and aggregation guide](docs/reproduction.md).

## License and contributions

AdCoEvo is released under the [Apache License 2.0](LICENSE). Obtain the full AuctionNet simulator and original datasets separately. Attribution and licenses for adapted code are listed in [NOTICE](NOTICE). See the [contribution guide](CONTRIBUTING.md) for development setup and contribution instructions.
