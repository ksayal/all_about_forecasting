# COSMIC

*AWS AI Labs, Jun 2025*

| | |
|---|---|
| Official source | https://arxiv.org/abs/2506.03128 (paper only) |
| Package (pinned) | `none` |
| Weights | `none public` |
| License | n/a |
| Env | `scripts/make_env.sh cosmic` |

## How it works
Encoder-decoder (efficient T5) trained with *informative covariate augmentation*: synthetic covariates are mixed into real series during pretraining so the model learns to use covariates zero-shot. Sizes 20M / 40M / 200M.

## Notes
**Placeholder.** No code or weights were public when this repo was built, so the module refuses to run instead of faking it. If AWS releases it (likely via `chronos-forecasting` or AutoGluon), implement it like `chronos_bolt`.

## Use
```python
from fchub import load_model
model = load_model("COSMIC")
fc = model.predict(context_df, horizon=24, freq="h")
```
