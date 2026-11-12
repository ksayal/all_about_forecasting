# TabPFN-TS

*Prior Labs, 2025*

| | |
|---|---|
| Official source | https://github.com/PriorLabs/tabpfn-time-series |
| Package (pinned) | `tabpfn-time-series==1.4.0` |
| Weights | `TabPFN regressor weights (downloaded by the `tabpfn` package)` |
| License | Code Apache-2.0; TabPFN weights under Prior Labs' license (newer versions non-commercial; check) |
| Env | `scripts/make_env.sh tabpfn_ts` |

## How it works
Treats forecasting as tabular regression: every time step becomes a row with features (running index, calendar features, auto-detected seasonal sin/cos), and the TabPFN tabular foundation model predicts the target in-context. No time-series pretraining at all, yet very competitive.

## Notes
`mode="local"` runs on your machine (slow on CPU; that's the 305 s median runtime). `mode="client"` uses Prior Labs' hosted API and needs a token; don't send confidential data.

## Use
```python
from fchub import load_model
model = load_model("TabPFN-TS")
fc = model.predict(context_df, horizon=24, freq="h")
```
