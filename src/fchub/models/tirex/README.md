# TiRex

*NX-AI, May 2025*

| | |
|---|---|
| Official source | https://github.com/NX-AI/tirex |
| Package (pinned) | `tirex-ts==1.4.2` |
| Weights | `NX-AI/TiRex (35M)` |
| License | NXAI Community License: free for research and for commercial use below a revenue threshold; read it before production use |
| Env | `scripts/make_env.sh tirex` |

## How it works
Built on xLSTM (sLSTM blocks) rather than a transformer. The recurrent state lets a small 35M model track long patterns, and *contiguous patch masking* in training makes it strong on long horizons. Outputs the deciles.

## Notes
On NVIDIA GPUs install `tirex-ts[cuda]` for custom kernels; elsewhere the pure-torch backend is used automatically.

## Use
```python
from fchub import load_model
model = load_model("TiRex")
fc = model.predict(context_df, horizon=24, freq="h")
```
