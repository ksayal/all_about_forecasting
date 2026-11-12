"""The one interface every model in the hub implements.

Data contract
-------------
Input  : a *long-format* pandas DataFrame with columns
         ``item_id`` (str), ``timestamp`` (datetime), ``target`` (float).
         One row per (series, time step), regular frequency, no gaps.
Output : a long-format DataFrame with columns
         ``item_id``, ``timestamp`` (future), ``mean`` (point forecast)
         and one column per quantile level, named like ``"0.1"``, ``"0.5"``.

This is the same layout used by fev, AutoGluon, Chronos ``predict_df`` and
TabPFN-TS, so data moves between tools without conversion.

Most foundation models only need the raw context arrays, so subclasses
usually implement just :meth:`Forecaster._predict_arrays`. Models that need
timestamps (e.g. TabPFN-TS) override :meth:`Forecaster.predict` directly.
"""

from __future__ import annotations

import abc
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

DEFAULT_QUANTILES: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

ID, TS, TARGET = "item_id", "timestamp", "target"


@dataclass
class ModelInfo:
    """Static facts about a model, shown in docs and leaderboards."""

    name: str
    family: str  # "statistical" | "foundation"
    source: str  # official repo / package
    license: str
    weights: str | None = None  # Hugging Face id, if any
    notes: str = ""
    supports_covariates: bool = False


@dataclass
class ArrayForecast:
    """Raw numeric forecast: ``mean`` is (n, h); ``quantiles`` is (n, h, q)."""

    mean: np.ndarray
    quantiles: np.ndarray
    quantile_levels: tuple[float, ...] = field(default=DEFAULT_QUANTILES)


class Forecaster(abc.ABC):
    """Base class. Subclasses set ``info`` and implement ``_predict_arrays``."""

    info: ModelInfo

    def __init__(self, **kwargs):
        self.config = kwargs
        self.last_runtime_s: float | None = None

    # ------------------------------------------------------------------ API
    def fit(self, train_df: pd.DataFrame, freq: str) -> "Forecaster":
        """Zero-shot models need no training, so the default is a no-op.

        Kept in the interface so fine-tunable / local models can plug in later.
        """
        return self

    def predict(
        self,
        context_df: pd.DataFrame,
        horizon: int,
        freq: str,
        quantile_levels: tuple[float, ...] = DEFAULT_QUANTILES,
    ) -> pd.DataFrame:
        """Forecast ``horizon`` steps past the end of every series."""
        validate_long_df(context_df)
        ids, contexts, last_ts = split_series(context_df)
        t0 = time.perf_counter()
        fc = self._predict_arrays(contexts, horizon, tuple(quantile_levels), freq=freq)
        self.last_runtime_s = time.perf_counter() - t0
        return to_long_forecast(ids, last_ts, freq, horizon, fc)

    @abc.abstractmethod
    def _predict_arrays(
        self,
        contexts: list[np.ndarray],
        horizon: int,
        quantile_levels: tuple[float, ...],
        freq: str,
    ) -> ArrayForecast:
        """Return forecasts for a list of 1-D context arrays."""

    def __repr__(self) -> str:  # pragma: no cover
        return f"{type(self).__name__}({self.config})"


# ---------------------------------------------------------------- helpers
def validate_long_df(df: pd.DataFrame) -> None:
    missing = {ID, TS, TARGET} - set(df.columns)
    if missing:
        raise ValueError(f"context_df is missing columns: {sorted(missing)}")


def split_series(df: pd.DataFrame) -> tuple[list[str], list[np.ndarray], list[pd.Timestamp]]:
    """Long DataFrame -> (ids, list of float32 arrays, last timestamp per series)."""
    df = df.sort_values([ID, TS])
    ids, arrays, last = [], [], []
    for item_id, g in df.groupby(ID, sort=False):
        ids.append(item_id)
        arrays.append(g[TARGET].to_numpy(dtype=np.float32))
        last.append(pd.Timestamp(g[TS].iloc[-1]))
    return ids, arrays, last


def future_index(last_ts: pd.Timestamp, freq: str, horizon: int) -> pd.DatetimeIndex:
    return pd.date_range(last_ts, periods=horizon + 1, freq=freq)[1:]


def to_long_forecast(
    ids: list[str],
    last_ts: list[pd.Timestamp],
    freq: str,
    horizon: int,
    fc: ArrayForecast,
) -> pd.DataFrame:
    mean = np.asarray(fc.mean, dtype=float)
    q = np.asarray(fc.quantiles, dtype=float)
    if mean.shape != (len(ids), horizon):
        raise ValueError(f"mean has shape {mean.shape}, expected {(len(ids), horizon)}")
    if q.shape != (len(ids), horizon, len(fc.quantile_levels)):
        raise ValueError(f"quantiles have shape {q.shape}")
    frames = []
    for i, item_id in enumerate(ids):
        cols = {ID: item_id, TS: future_index(last_ts[i], freq, horizon), "mean": mean[i]}
        for j, lvl in enumerate(fc.quantile_levels):
            cols[str(lvl)] = q[i, :, j]
        frames.append(pd.DataFrame(cols))
    return pd.concat(frames, ignore_index=True)


def select_quantiles(
    q: np.ndarray, model_levels: tuple[float, ...], wanted: tuple[float, ...]
) -> np.ndarray:
    """Pick (or linearly interpolate) wanted quantile levels from a model's native grid.

    ``q`` is (n, h, len(model_levels)). Many models are trained on the deciles
    0.1..0.9; asking for e.g. 0.05 is extrapolation and is clipped to the edge.
    """
    model_levels = np.asarray(model_levels, dtype=float)
    out = np.empty(q.shape[:2] + (len(wanted),), dtype=float)
    for j, lvl in enumerate(wanted):
        hit = np.where(np.isclose(model_levels, lvl))[0]
        if hit.size:
            out[..., j] = q[..., hit[0]]
        else:
            # vectorised linear interpolation along the quantile axis
            out[..., j] = np.apply_along_axis(lambda v: np.interp(lvl, model_levels, v), -1, q)
    return out


def quantiles_from_samples(samples: np.ndarray, levels: tuple[float, ...]) -> np.ndarray:
    """samples (n, h, s) -> quantiles (n, h, q)."""
    return np.moveaxis(np.quantile(samples, levels, axis=-1), 0, -1)


def pick_device(preferred: str | None = None) -> str:
    """cuda > mps (Apple Silicon) > cpu, unless the user forces one."""
    if preferred:
        return preferred
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"
