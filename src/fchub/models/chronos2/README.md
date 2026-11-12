# Chronos-2

*Amazon, Oct 2025*

| | |
|---|---|
| Official source | https://github.com/amazon-science/chronos-forecasting |
| Package (pinned) | `chronos-forecasting==2.3.2` |
| Weights | `amazon/chronos-2 (120M)` |
| License | Apache-2.0 (code and weights) |
| Env | `scripts/make_env.sh chronos2` |

## How it works
Encoder-only transformer with *group attention*, so one model handles univariate, multivariate and covariate-informed forecasting zero-shot. Inputs are scaled and patched; a quantile head outputs 21 quantiles (0.01 to 0.99) for all horizon steps in one forward pass.

## Notes
Covariates: the official API accepts `past_covariates` / `future_covariates` per series; this wrapper uses the univariate path so results are comparable with the other models. Runs on CPU or Apple MPS.

## Use
```python
from fchub import load_model
model = load_model("Chronos-2")
fc = model.predict(context_df, horizon=24, freq="h")
```
