"""Sample datasets and evaluation tasks.

A *task* = one dataset + forecast horizon + number of rolling evaluation
windows. For window k (counted back from the end of the data) we hide the last
``horizon`` points of that window, give the model everything before it as
context, and score the hidden points. This is the same rolling-origin scheme
fev / GIFT-Eval use, just with smaller data so it runs on a laptop.

Bundled datasets (``data/*.csv``, real data, no network needed):
  air_passengers   monthly   1 series   classic trend + multiplicative seasonality
  co2_weekly       weekly    1 series   Mauna Loa CO2, smooth trend + yearly cycle
  elnino_monthly   monthly   1 series   Pacific sea-surface temperature, irregular cycles
  sunspots_yearly  yearly    1 series   ~11-year cycle with no fixed calendar period
Generated:
  synthetic_hourly hourly   20 series   daily + weekly seasonality, trend, noise, level shifts

On a machine with Hugging Face access, :func:`load_hf_chronos_dataset` pulls
the larger public benchmark datasets (e.g. m4_hourly) for real benchmarking.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from fchub.base import ID, TARGET, TS

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


@dataclass(frozen=True)
class Task:
    dataset: str
    freq: str
    horizon: int
    n_windows: int = 1

    @property
    def name(self) -> str:
        return f"{self.dataset}/h={self.horizon}"


SAMPLE_TASKS: list[Task] = [
    Task("air_passengers", "ME", 12, n_windows=2),
    Task("co2_weekly", "W-SAT", 52, n_windows=2),
    Task("elnino_monthly", "MS", 12, n_windows=3),
    Task("sunspots_yearly", "YS", 11, n_windows=2),
    Task("synthetic_hourly", "h", 48, n_windows=2),
]


def load_dataset(name: str) -> pd.DataFrame:
    """Return a long-format frame (item_id, timestamp, target)."""
    if name == "synthetic_hourly":
        return make_synthetic_hourly()
    path = DATA_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"No bundled dataset {name!r} at {path}")
    df = pd.read_csv(path, parse_dates=[TS])
    df[ID] = df[ID].astype(str)
    return df


def make_synthetic_hourly(n_series: int = 20, length: int = 24 * 7 * 8, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(length)
    idx = pd.date_range("2024-01-01", periods=length, freq="h")
    frames = []
    for i in range(n_series):
        level = rng.uniform(50, 500)
        daily = rng.uniform(0.1, 0.4) * level * np.sin(2 * np.pi * (t % 24) / 24 + rng.uniform(0, 2 * np.pi))
        weekly = rng.uniform(0.0, 0.2) * level * np.sin(2 * np.pi * t / (24 * 7))
        trend = rng.uniform(-0.2, 0.2) * level * t / length
        shift = np.where(t > rng.integers(length // 2, length), rng.uniform(-0.2, 0.2) * level, 0.0)
        noise = rng.normal(0, rng.uniform(0.02, 0.1) * level, length)
        y = np.maximum(level + daily + weekly + trend + shift + noise, 0)
        frames.append(pd.DataFrame({ID: f"series_{i:02d}", TS: idx, TARGET: y.astype(np.float32)}))
    return pd.concat(frames, ignore_index=True)


def rolling_windows(df: pd.DataFrame, task: Task):
    """Yield (window_index, context_df, actuals_df), oldest window first."""
    df = df.sort_values([ID, TS])
    for k in range(task.n_windows):
        cut = task.horizon * (task.n_windows - k)  # points hidden after the context
        ctx, act = [], []
        for _, g in df.groupby(ID, sort=False):
            if len(g) <= cut + task.horizon:
                continue  # too short to give any context for this window
            end = len(g) - cut
            ctx.append(g.iloc[:end])
            act.append(g.iloc[end : end + task.horizon])
        if ctx:
            yield k, pd.concat(ctx, ignore_index=True), pd.concat(act, ignore_index=True)


def load_hf_chronos_dataset(config: str, max_series: int | None = None) -> pd.DataFrame:
    """Load a dataset from the public `autogluon/chronos_datasets` hub repo (needs HF access).

    Example configs: "m4_hourly", "m4_monthly", "electricity_15min", "monash_tourism_monthly".
    Requires ``pip install datasets``.
    """
    import datasets

    ds = datasets.load_dataset("autogluon/chronos_datasets", config, split="train")
    ds.set_format("numpy")
    target_col = "target" if "target" in ds.column_names else ds.column_names[-1]
    frames = []
    for i, row in enumerate(ds):
        if max_series is not None and i >= max_series:
            break
        frames.append(
            pd.DataFrame({ID: str(row["id"]), TS: pd.to_datetime(row["timestamp"]), TARGET: row[target_col]})
        )
    return pd.concat(frames, ignore_index=True)
