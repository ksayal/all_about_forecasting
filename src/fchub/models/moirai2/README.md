# Moirai-2.0

*Salesforce AI Research, Aug 2025*

| | |
|---|---|
| Official source | https://github.com/SalesforceAIResearch/uni2ts |
| Package (pinned) | `uni2ts==2.0.0` |
| Weights | `Salesforce/moirai-2.0-R-small (11M)` |
| License | Code Apache-2.0; **weights CC-BY-NC-4.0 (non-commercial)** |
| Env | `scripts/make_env.sh moirai2` |

## How it works
Decoder-only successor to Moirai 1 (masked encoder) and Moirai-MoE. Uses a quantile loss, multi-token prediction and a single patch size, which makes it faster and more accurate than the larger earlier versions.

## Notes
`uni2ts` pins torch<2.5 and numpy 1.26, so Python 3.11 is used. The leaderboard's 28% leakage figure means some benchmark datasets were in its training data, so treat its scores on those with care.

## Use
```python
from fchub import load_model
model = load_model("Moirai-2.0")
fc = model.predict(context_df, horizon=24, freq="h")
```
