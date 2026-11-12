# Toto-1.0

*Datadog, May 2025*

| | |
|---|---|
| Official source | https://github.com/DataDog/toto |
| Package (pinned) | `toto-ts==0.2.0` |
| Weights | `Datadog/Toto-Open-Base-1.0 (151M)` |
| License | Apache-2.0 (code and weights) |
| Env | `scripts/make_env.sh toto` |

## How it works
Decoder-only transformer trained mostly on observability metrics (plus public data). It alternates time-wise and variate-wise attention and predicts a Student-t mixture, so forecasts are produced by sampling paths autoregressively. Quantiles come from the samples.

## Notes
`num_samples` trades accuracy for speed (default 256). `toto-ts` pins torch==2.7.0 and numpy 1.26, so it needs its own env.

## Use
```python
from fchub import load_model
model = load_model("Toto-1.0")
fc = model.predict(context_df, horizon=24, freq="h")
```
