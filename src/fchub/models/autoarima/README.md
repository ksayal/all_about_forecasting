# AutoARIMA

Automatic (seasonal) ARIMA order selection, a port of R's forecast::auto.arima. Slow on long series.

- **Implementation:** [Nixtla statsforecast](https://github.com/Nixtla/statsforecast) (`pip install statsforecast==2.1.1`), Apache-2.0
- **Weights:** none, fitted on each series at predict time
- **Season length:** inferred from the frequency (`fchub.freq.season_length`), override with `season_length=`

```python
from fchub.registry import load_model
model = load_model("AutoARIMA")
fc = model.predict(context_df, horizon=24, freq="h")
```
