# Naive

Repeats the last observed value. The simplest possible baseline; anything worse than this is broken.

- **Implementation:** [Nixtla statsforecast](https://github.com/Nixtla/statsforecast) (`pip install statsforecast==2.1.1`), Apache-2.0
- **Weights:** none, fitted on each series at predict time
- **Season length:** inferred from the frequency (`fchub.freq.season_length`), override with `season_length=`

```python
from fchub.registry import load_model
model = load_model("Naive")
fc = model.predict(context_df, horizon=24, freq="h")
```
