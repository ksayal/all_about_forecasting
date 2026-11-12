# How this repository was built, step by step

This guide explains every decision, in the order the code was written. Read
it alongside the files it points to.

---

## Step 0: Should each model contain its full source code?

The original idea was "each model is its own module containing its full code,
self-contained". We kept the **one module per model** part and dropped the
**copy the full source** part. Here's why.

**What "the model" actually is.** A foundation model like Chronos-2 is two
things: (a) the architecture code, and (b) hundreds of MB of pretrained weights
on Hugging Face. The weights hold the behaviour. Copying the source without
the weights gives you an untrained network. Copying the source *and* the weights
gives you a frozen fork that you now have to maintain.

**Problems with copying the source:**

1. *It goes stale.* These repos ship bug fixes, faster kernels and new
   checkpoints every few weeks (TimesFM went 1.0 → 2.0 → 2.5 → 3 in about 18 months).
2. *Silent mismatches.* Weights are only guaranteed to load correctly with the
   code version they were released with. A hand-copied file that drifts by
   one layer-norm detail gives wrong forecasts with no error.
3. *Licenses travel with code.* TiRex (NXAI Community License) and Moirai-2.0
   weights (CC-BY-NC-4.0) restrict commercial use. Copying their code into your
   repo puts those terms inside your repo.
4. *Dependency clashes don't go away.* Copied code still needs the same torch,
   numpy and transformers versions, and those conflict between models (Step 5).

**What we do instead:** each folder under `src/fchub/models/` is a thin wrapper
(~50-80 lines) that imports the **official package at a pinned version** and
loads the **official weights**. The folder still reads as self-contained: its
own `model.py`, `requirements.txt` and `README.md` describing the architecture,
license and quirks. You'd only copy source when no package exists. That applies
to COSMIC, but it has no public weights either, so it's a documented placeholder.

This is also how the GIFT-Eval and fev-bench leaderboards run these models.

---

## Step 1: Finding the official sources

For each leaderboard entry we looked for the package published by the
authors, checked PyPI for the latest version and license, and checked the Hugging
Face model card for the weights license. Results are in [MODELS.md](MODELS.md).
Two findings shaped the design:

- **COSMIC** (AWS) is only a paper so far: no code, no weights.
- **Dependency pins conflict** (`uni2ts` needs torch < 2.5, `toto-ts` pins
  torch == 2.7.0, Sundial wants transformers 4.40), which leads to Step 5.

---

## Step 2: One data format and one interface (`src/fchub/base.py`)

Every model has a different API: Chronos takes a list of tensors, TimesFM a
list of numpy arrays after a `compile()` call, Toto a `MaskedTimeseries`
object, TabPFN-TS a DataFrame with timestamps, statsforecast a `unique_id/ds/y`
frame. To compare them we need one front door.

**Input**: a long-format DataFrame `item_id, timestamp, target`. This is the
layout used by fev, AutoGluon, Chronos `predict_df` and TabPFN-TS, so it's the
closest thing the field has to a standard.

**Output**: `item_id, timestamp, mean, "0.1", ..., "0.9"`, a point forecast
plus quantiles. Quantiles matter because the leaderboard scores *probabilistic*
accuracy, and in practice you plan with ranges, not single numbers.

**The `Forecaster` class**:

```python
class Forecaster:
    info: ModelInfo                       # name, source, license, weights
    def fit(train_df, freq)               # no-op for zero-shot models; kept for fine-tuning later
    def predict(context_df, horizon, freq, quantile_levels) -> DataFrame
    def _predict_arrays(contexts, horizon, quantile_levels, freq) -> ArrayForecast
```

`predict` handles the DataFrame bookkeeping (split into series, build future
timestamps, assemble output). Most models only implement `_predict_arrays`,
which gets a plain list of 1-D numpy arrays. That keeps each wrapper focused on
the one thing that differs: how to call the model.

Small helpers live here too:
- `select_quantiles`: models are trained on fixed quantile grids (usually the
  deciles), so this picks or interpolates the levels you asked for.
- `quantiles_from_samples`: sampling models (Toto, Sundial) return paths;
  this turns them into quantiles.
- `pick_device`: CUDA → Apple MPS → CPU.

---

## Step 3: Statistical baselines (`models/naive`, `seasonal_naive`, `autoets`, `autoarima`, `autotheta`, `stat_ensemble`)

These come first because they're cheap, need no downloads, and **anchor the
leaderboard**: Skill Score is defined relative to SeasonalNaive.

All six use [Nixtla statsforecast](https://github.com/Nixtla/statsforecast), the
fastest maintained implementation (numba-compiled ports of R's `forecast`
package). The shared code is in `models/_shared/statsforecast_base.py`:

- **Season length** comes from the frequency (`fchub/freq.py`: hourly → 24,
  daily → 7, weekly → 52, monthly → 12).
- **Intervals → quantiles**: statsforecast returns prediction intervals
  (`lo-80`, `hi-80`). Quantile 0.1 is the lower edge of the 80% interval, 0.9 the
  upper edge, 0.5 the point forecast.
- **Ensemble** averages AutoETS + AutoARIMA + AutoTheta, then re-sorts
  quantiles because averaging can make them cross.

Each model folder is ~25 lines: it only says *which* statsforecast model to build.

What to expect: AutoARIMA on long weekly series (season 52) is very slow. That's
the 186 s median runtime and 10 failures on the leaderboard, and you'll see it here too.

---

## Step 4: Foundation model wrappers

Each wrapper does three things: load the pretrained weights in `__init__`,
convert our arrays to the model's input format, and convert its output back to
`(mean, quantiles)`. What's specific to each:

| Model | Input conversion | Output conversion |
|---|---|---|
| Chronos-2 | list of arrays → `predict_quantiles` | per-series tensors `(1,h,q)` → stack |
| Chronos-Bolt | batches of tensors | `(b,h,q)` tensor; mean returned directly |
| TimesFM-2.5 | must `compile(ForecastConfig)` first; we compile lazily and only recompile if the horizon grows | output index 0 = mean, 1..9 = deciles |
| TiRex | list of arrays, `output_type="numpy"` | deciles; median as mean |
| Toto-1.0 | builds a `MaskedTimeseries` per series (values, padding mask, time interval) | 256 sampled paths → quantiles; median as point |
| Moirai-2.0 | `prediction_length` is a constructor arg, so we build a light `Moirai2Forecast` per horizon around the shared pretrained module | `(b,Q,h)` → transpose |
| Sundial | per-series `generate(num_samples=100)` via transformers | samples → quantiles; mean as point |
| TabPFN-TS | needs real timestamps for calendar features, so it overrides `predict` and passes the DataFrame | quantile columns already named |
| COSMIC | raises `NotImplementedError` with an explanation | n/a |

Defaults that differ from the libraries, and why:
- `torch.compile` is **off** (TimesFM, TabPFN, Toto): it speeds up long CUDA
  runs but is slow or broken on CPU and Apple MPS.
- Moirai defaults to CPU unless CUDA exists: its pinned torch < 2.5 has patchy MPS support.
- Sundial uses `trust_remote_code=True` (required by the model). Pin a reviewed
  `revision` before trusting it.

---

## Step 5: One virtualenv per model (`scripts/make_env.sh`)

The official packages can't all be installed together:

| Package | Pins |
|---|---|
| uni2ts (Moirai) | torch < 2.5, numpy ~= 1.26, gluonts ~= 0.14 |
| toto-ts | torch == 2.7.0, numpy == 1.26.4, transformers == 4.52.1 |
| Sundial | transformers == 4.40.1 |
| chronos-forecasting | transformers >= 4.41 |

So the core package (`fchub`) depends only on numpy and pandas, and each model
lists its own pinned requirements. `scripts/make_env.sh <model>` builds
`.venvs/<model>` with [uv](https://docs.astral.sh/uv/) (fast, so ten envs is fine).
Python 3.11 is the newest version every pin supports.

`registry.py` imports model modules **lazily**, so `import fchub` works in any
env, and asking for a model whose env isn't installed tells you which script to run.

---

## Step 6: Evaluation tasks (`src/fchub/data.py`)

A **task** = dataset + horizon + number of rolling windows. For each window we
hide the last `horizon` points, forecast them from everything before, and
score. Several windows per task reduce luck.

Bundled real datasets (CSV in `data/`, no network needed): AirPassengers
(monthly), Mauna Loa CO2 (weekly), El Niño SST (monthly), Sunspots (yearly),
plus a generated 20-series hourly set with daily and weekly seasonality. They're
deliberately small and varied so a laptop can run every model. For real
benchmarking, `load_hf_chronos_dataset("m4_hourly")` pulls the public benchmark
datasets from Hugging Face.

---

## Step 7: Metrics and the leaderboard (`metrics.py`, `leaderboard.py`)

Per task:
- **MASE**: point-forecast MAE divided by the in-sample seasonal-naive MAE.
  Scale-free, so series with different units can be averaged.
- **SQL** (scaled quantile loss): pinball loss averaged over quantiles, same
  scaling. The probabilistic version of MASE, and the default ranking metric.
- **WQL**: pinball loss normalised by the series magnitude.

Across tasks (the leaderboard columns):
- **Skill Score** = 1 − geometric mean of (model error ÷ SeasonalNaive error).
  SeasonalNaive = 0, positive = better. Geometric mean stops one easy task from dominating.
- **Avg. Win Rate**: how often the model beats each other model on a task
  (ties count half), averaged over opponents.
- **Median runtime** and **#Failures**. A failed window is scored with a
  seasonal-naive forecast (fev's convention), so crashing is penalised rather than hidden.

Leakage (%) from the original table isn't reproduced: it measures overlap
between benchmark data and each model's *training* data, which needs the
training corpus lists, not code.

---

## Step 8: Running it

```bash
scripts/make_env.sh stats && .venvs/stats/bin/python -m pytest -q
scripts/make_env.sh tirex
scripts/run_benchmark.sh SeasonalNaive AutoETS TiRex
cat results/LEADERBOARD.md
```

Each model writes `results/<model>.csv`, so you can add models one at a time
and rebuild the table whenever you like.

---

## Where to go next

- Add covariate tasks (Chronos-2, TimesFM-2.5, TabPFN-TS and Toto support them).
- Plug in the full fev-bench or GIFT-Eval task lists via `load_hf_chronos_dataset`.
- Add LLM-based forecasters (e.g. prompting a general LLM with the series) as
  another `Forecaster`. The interface doesn't care what's inside.
- Fine-tuning: implement `fit()` for Chronos-2 / Toto, which both ship fine-tuning code.
