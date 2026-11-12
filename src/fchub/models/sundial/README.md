# Sundial

*Tsinghua THUML, ICML 2025*

| | |
|---|---|
| Official source | https://github.com/thuml/Sundial |
| Package (pinned) | `transformers==4.40.1 (+ torch)` |
| Weights | `thuml/sundial-base-128m` |
| License | Apache-2.0 per the model card (verify) |
| Env | `scripts/make_env.sh sundial` |

## How it works
Decoder-only transformer with a *TimeFlow* (flow-matching) head: instead of fixed quantiles it generates sample paths directly from continuous values, without tokenising. Trained on TimeBench (~1 trillion points).

## Notes
Loaded with `trust_remote_code=True`, which runs Python downloaded from the model repo. Review it once and pin `revision=<commit>`.

## Use
```python
from fchub import load_model
model = load_model("Sundial")
fc = model.predict(context_df, horizon=24, freq="h")
```
