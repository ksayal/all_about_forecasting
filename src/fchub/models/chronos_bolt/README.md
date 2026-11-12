# Chronos-Bolt

*Amazon, Nov 2024*

| | |
|---|---|
| Official source | https://github.com/amazon-science/chronos-forecasting |
| Package (pinned) | `chronos-forecasting==2.3.2` |
| Weights | `amazon/chronos-bolt-{tiny,mini,small,base}` |
| License | Apache-2.0 (code and weights) |
| Env | `scripts/make_env.sh chronos_bolt` |

## How it works
T5 encoder-decoder. Unlike the original Chronos (which tokenised values and sampled autoregressively), Bolt splits the context into patches and the decoder emits multi-step quantile forecasts directly, which is why it's the fastest model on the leaderboard.

## Notes
Pick a size with `load_model("Chronos-Bolt", model_id="amazon/chronos-bolt-small")`. Quantiles outside 0.1..0.9 are extrapolated by the library.

## Use
```python
from fchub import load_model
model = load_model("Chronos-Bolt")
fc = model.predict(context_df, horizon=24, freq="h")
```
