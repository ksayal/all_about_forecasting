"""Core pieces that need only numpy + pandas."""

import py_compile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fchub.base import ArrayForecast, Forecaster, ModelInfo, select_quantiles
from fchub.data import SAMPLE_TASKS, load_dataset, rolling_windows
from fchub.freq import season_length
from fchub.metrics import evaluate_forecast, leaderboard
from fchub.registry import ENV_OF, MODELS


class LastValue(Forecaster):
    info = ModelInfo(name="LastValue", family="test", source="-", license="-")

    def _predict_arrays(self, contexts, horizon, quantile_levels, freq):
        mean = np.stack([np.full(horizon, c[-1]) for c in contexts])
        q = np.repeat(mean[..., None], len(quantile_levels), axis=-1)
        return ArrayForecast(mean, q, quantile_levels)


def test_predict_output_schema():
    df = load_dataset("synthetic_hourly")
    fc = LastValue().predict(df, horizon=5, freq="h", quantile_levels=(0.1, 0.5, 0.9))
    assert list(fc.columns) == ["item_id", "timestamp", "mean", "0.1", "0.5", "0.9"]
    assert len(fc) == df["item_id"].nunique() * 5
    first = fc.groupby("item_id")["timestamp"].min()
    last_ctx = df.groupby("item_id")["timestamp"].max()
    assert ((first - last_ctx) == pd.Timedelta("1h")).all()


@pytest.mark.parametrize("freq,expected", [("h", 24), ("15min", 96), ("D", 7), ("W-SAT", 52), ("MS", 12), ("ME", 12), ("YS", 1)])
def test_season_length(freq, expected):
    assert season_length(freq) == expected


def test_select_quantiles_interpolates():
    native = (0.1, 0.5, 0.9)
    q = np.array([[[1.0, 5.0, 9.0]]])
    out = select_quantiles(q, native, (0.1, 0.3, 0.9))
    np.testing.assert_allclose(out[0, 0], [1.0, 3.0, 9.0])


def test_perfect_forecast_scores_zero():
    df = load_dataset("air_passengers")
    task = SAMPLE_TASKS[0]
    _, ctx, act = next(rolling_windows(df, task))
    fc = act.rename(columns={"target": "mean"}).copy()
    for q in (0.1, 0.5, 0.9):
        fc[str(q)] = fc["mean"]
    s = evaluate_forecast(fc, ctx, act, 12, (0.1, 0.5, 0.9))
    assert s["MASE"] == pytest.approx(0) and s["SQL"] == pytest.approx(0)


def test_leaderboard_skill_and_winrate():
    res = pd.DataFrame({
        "model": ["A", "SeasonalNaive", "A", "SeasonalNaive"],
        "task": ["t1", "t1", "t2", "t2"],
        "SQL": [0.5, 1.0, 0.5, 1.0],
        "runtime_s": [1, 1, 1, 1], "failed": [False] * 4,
    })
    tab = leaderboard(res)
    assert tab.loc["A", "Skill Score (%)"] == 50.0
    assert tab.loc["SeasonalNaive", "Skill Score (%)"] == 0.0
    assert tab.loc["A", "Avg. Win Rate (%)"] == 100.0


def test_all_tasks_have_windows():
    for task in SAMPLE_TASKS:
        assert sum(1 for _ in rolling_windows(load_dataset(task.dataset), task)) == task.n_windows


def test_every_registered_model_has_a_module_folder():
    root = Path(__file__).resolve().parents[1] / "src" / "fchub" / "models"
    for name, path in MODELS.items():
        folder = root / path.split(":")[0].rsplit(".", 1)[1]
        for f in ("__init__.py", "model.py", "requirements.txt", "README.md"):
            assert (folder / f).exists(), f"{name}: missing {f}"
        py_compile.compile(str(folder / "model.py"), doraise=True)
        assert ENV_OF[name]
