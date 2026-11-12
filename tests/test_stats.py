"""Run the statistical baselines end to end (needs the 'stats' env: statsforecast)."""

import numpy as np
import pytest

pytest.importorskip("statsforecast")

from fchub.benchmark import run_model
from fchub.data import SAMPLE_TASKS
from fchub.registry import STAT_MODELS, load_model
from fchub.data import load_dataset

QS = (0.1, 0.5, 0.9)


@pytest.mark.parametrize("name", STAT_MODELS)
def test_stat_model_forecasts(name):
    df = load_dataset("air_passengers")
    fc = load_model(name).predict(df, horizon=12, freq="ME", quantile_levels=QS)
    assert len(fc) == 12
    vals = fc[["0.1", "0.5", "0.9"]].to_numpy()
    assert np.isfinite(vals).all()
    assert (np.diff(vals, axis=1) >= -1e-9).all(), "quantiles must not cross"


def test_benchmark_runs_seasonal_naive():
    res = run_model("SeasonalNaive", SAMPLE_TASKS[:1])
    assert not res["failed"].any()
    assert res["MASE"].notna().all()
