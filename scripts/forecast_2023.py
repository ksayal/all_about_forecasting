"""Forecast 2023 EDB_English daily call volume with Chronos-2 and score it with MAPE.

Trains on 2022, forecasts every day of 2023 that has actuals (1/1/23 to 8/30/23).
Run from the repo root inside the Chronos-2 environment:

    scripts/make_env.sh chronos2
    .venvs/chronos2/bin/python scripts/forecast_2023.py
"""

from pathlib import Path

import pandas as pd

from fchub import load_model

df = pd.read_csv("data/EDB_English_daily_long.csv", parse_dates=["timestamp"])
train = df[df.timestamp < "2023-01-01"]
test = df[df.timestamp >= "2023-01-01"]

# predict() returns columns: item_id, timestamp, mean, 0.1 ... 0.9 (quantiles)
fc = load_model("Chronos-2").predict(train, horizon=len(test), freq="D")
print(fc.head())

out = test.merge(fc, on=["item_id", "timestamp"])
mape = ((out.target - out["mean"]).abs() / out.target).mean() * 100
print(f"MAPE: {mape:.1f}%")

Path("results").mkdir(exist_ok=True)
out.to_csv("results/chronos2_2023_forecast.csv", index=False)
