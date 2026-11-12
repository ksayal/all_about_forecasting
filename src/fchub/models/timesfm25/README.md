# TimesFM-2.5

*Google Research, Sep 2025*

| | |
|---|---|
| Official source | https://github.com/google-research/timesfm |
| Package (pinned) | `timesfm[torch]==3.0.2` |
| Weights | `google/timesfm-2.5-200m-pytorch` |
| License | Apache-2.0 (code and weights) |
| Env | `scripts/make_env.sh timesfm25` |

## How it works
200M decoder-only transformer over patches of 32 points, outputting patches of 128. Supports up to 16k context and an optional 30M continuous quantile head for horizons up to 1k. Has to be `compile()`d with a max context/horizon before forecasting; the wrapper does this lazily.

## Notes
The 3.x package also contains TimesFM 3 (`timesfm3`); this module targets 2.5 to match the leaderboard. Covariates via `forecast_with_covariates` (needs the `xreg` extra, JAX).

## Use
```python
from fchub import load_model
model = load_model("TimesFM-2.5")
fc = model.predict(context_df, horizon=24, freq="h")
```
