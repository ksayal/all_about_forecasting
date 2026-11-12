# all_about_forecasting

A central repository for experimenting with different time series approaches on
different datasets (univariate, multivariate and covariate series) to see which
model performs best, including recent foundation models such as Chronos-2,
TimeLLM and TimeGPT, and how to fine-tune them for the best results.

The code lives in the `fchub` package: one interface for classical and
foundation-model forecasters, plus a small benchmark that reproduces the
*shape* of the fev-bench / GIFT-Eval leaderboard (win rate, skill score,
runtime, failures) on data you can run on a laptop.

```python
from fchub import load_model
from fchub.data import load_dataset

df = load_dataset("air_passengers")          # long format: item_id, timestamp, target
model = load_model("Chronos-2")              # or "TiRex", "AutoETS", ...
fc = model.predict(df, horizon=12, freq="ME")
# -> item_id, timestamp, mean, 0.1, 0.2, ..., 0.9
```

## Models

| Model | Module | Wraps | License | Env |
|---|---|---|---|---|
| Chronos-2 | `models/chronos2` | `chronos-forecasting` 2.3.2, `amazon/chronos-2` | Apache-2.0 | `chronos2` |
| TimesFM-2.5 | `models/timesfm25` | `timesfm[torch]` 3.0.2, `google/timesfm-2.5-200m-pytorch` | Apache-2.0 | `timesfm25` |
| TiRex | `models/tirex` | `tirex-ts` 1.4.2, `NX-AI/TiRex` | NXAI Community License ⚠️ | `tirex` |
| Toto-1.0 | `models/toto` | `toto-ts` 0.2.0, `Datadog/Toto-Open-Base-1.0` | Apache-2.0 | `toto` |
| TabPFN-TS | `models/tabpfn_ts` | `tabpfn-time-series` 1.4.0 | Code Apache-2.0, weights Prior Labs license ⚠️ | `tabpfn_ts` |
| COSMIC | `models/cosmic` | **placeholder**: no public code/weights ([paper](https://arxiv.org/abs/2506.03128)) | n/a | n/a |
| Moirai-2.0 | `models/moirai2` | `uni2ts` 2.0.0, `Salesforce/moirai-2.0-R-small` | Code Apache-2.0, **weights CC-BY-NC-4.0** ⚠️ | `moirai2` |
| Chronos-Bolt | `models/chronos_bolt` | `chronos-forecasting` 2.3.2, `amazon/chronos-bolt-base` | Apache-2.0 | `chronos_bolt` |
| Sundial | `models/sundial` | `transformers` 4.40.1, `thuml/sundial-base-128m` | Apache-2.0 (verify) | `sundial` |
| StatEnsemble, AutoETS, AutoARIMA, AutoTheta, Naive, SeasonalNaive | `models/<name>` | `statsforecast` 2.1.1 | Apache-2.0 | `stats` |

⚠️ = check the license before any commercial use. Details and links: [docs/MODELS.md](docs/MODELS.md).

## Quick start

Requires [uv](https://docs.astral.sh/uv/) (`brew install uv` or `pip install uv`).

```bash
# 1. statistical baselines (small, no GPU, no downloads)
scripts/make_env.sh stats
.venvs/stats/bin/python -m pytest -q
.venvs/stats/bin/python -m fchub.benchmark --models SeasonalNaive,Naive,AutoETS,AutoTheta

# 2. a foundation model (downloads the package + weights from Hugging Face on first use)
scripts/make_env.sh tirex
.venvs/tirex/bin/python -m fchub.benchmark --models TiRex

# 3. leaderboard over everything in results/
.venvs/stats/bin/python -m fchub.leaderboard
```

Or run everything that has an env: `scripts/run_benchmark.sh`.

## Repository layout

```
src/fchub/
  base.py          Forecaster interface + data contract (read this first)
  registry.py      leaderboard name -> model class (lazy imports)
  data.py          sample datasets, rolling evaluation windows, HF loader
  metrics.py       MASE / SQL / WQL, skill score, win rate
  benchmark.py     run models over tasks -> results/<model>.csv
  leaderboard.py   results/*.csv -> leaderboard table
  models/
    <model>/       one folder per model: model.py, requirements.txt, README.md
    _shared/       code shared by the statsforecast baselines
data/              small bundled real datasets (CSV)
scripts/           make_env.sh, run_benchmark.sh
docs/              STEP_BY_STEP.md (how and why it's built), MODELS.md (sources, licenses)
```

## Design in one paragraph

Each model folder is a **thin wrapper around the official package and
pretrained weights**, not a copy of the model's source. The weights are the
model; the official package is the only code guaranteed to match them, and it
keeps getting bug fixes. Because the packages pin conflicting versions of torch,
numpy and transformers, every foundation model gets its **own virtualenv**, and
the benchmark writes one results file per model so the runs never need to share
an environment. See [docs/STEP_BY_STEP.md](docs/STEP_BY_STEP.md) for the full
reasoning.

## Adding a model

1. `src/fchub/models/<name>/model.py`: subclass `Forecaster`, set `info`,
   implement `_predict_arrays(contexts, horizon, quantile_levels, freq)` returning
   `ArrayForecast(mean (n,h), quantiles (n,h,q))`.
2. Add `requirements.txt` (pinned), `README.md`, `__init__.py`.
3. Register it in `src/fchub/registry.py` and in `scripts/run_benchmark.sh`.
4. `scripts/make_env.sh <name>` and run `python -m fchub.benchmark --models <Name>`.
