"""Shared plumbing for the classical baselines, all powered by Nixtla statsforecast.

statsforecast returns prediction *intervals* (``lo-80`` / ``hi-80`` ...),
so we convert the requested quantile levels to interval levels:
    q < 0.5  ->  lower bound of the (1 - 2q) interval
    q > 0.5  ->  upper bound of the (2q - 1) interval
    q = 0.5  ->  the point forecast (median == mean for the Gaussian errors these models assume)
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from fchub.base import (
    DEFAULT_QUANTILES,
    ID,
    TARGET,
    TS,
    ArrayForecast,
    Forecaster,
    validate_long_df,
)
from fchub.freq import season_length as infer_season_length

STATSFORECAST_LICENSE = "Apache-2.0"
STATSFORECAST_SOURCE = "https://github.com/Nixtla/statsforecast (pip: statsforecast)"


def _level_for(q: float) -> int | None:
    if np.isclose(q, 0.5):
        return None
    return int(round(abs(2 * q - 1) * 100))


class StatsForecastModel(Forecaster):
    """Base class: subclasses implement ``build_models(season_length)``."""

    #: default cap on history length; None = use everything. Slow models
    #: (AutoARIMA, the ensemble) override this so long weekly/hourly series finish.
    default_max_context: int | None = None

    def __init__(
        self,
        season_length: int | None = None,
        n_jobs: int = -1,
        max_context: int | None = -1,
        **kwargs,
    ):
        if max_context == -1:
            max_context = self.default_max_context
        super().__init__(season_length=season_length, n_jobs=n_jobs, max_context=max_context, **kwargs)
        self.season_length = season_length
        self.n_jobs = n_jobs
        self.max_context = max_context

    # subclasses return a list of statsforecast model objects
    def build_models(self, season_length: int) -> list:
        raise NotImplementedError

    def combine(self, fc: pd.DataFrame, cols: list[str]) -> pd.Series:
        """How to turn the per-model columns into one forecast (ensemble overrides)."""
        assert len(cols) == 1
        return fc[cols[0]]

    def predict(self, context_df, horizon, freq, quantile_levels=DEFAULT_QUANTILES):
        from statsforecast import StatsForecast

        validate_long_df(context_df)
        m = self.season_length or infer_season_length(freq)
        models = self.build_models(m)
        aliases = [getattr(mdl, "alias", repr(mdl)) for mdl in models]
        levels = sorted({lv for q in quantile_levels if (lv := _level_for(q)) is not None})

        df = context_df.rename(columns={ID: "unique_id", TS: "ds", TARGET: "y"})[
            ["unique_id", "ds", "y"]
        ].sort_values(["unique_id", "ds"])
        if self.max_context:
            df = df.groupby("unique_id", sort=False).tail(self.max_context)
        sf = StatsForecast(models=models, freq=freq, n_jobs=self.n_jobs)
        t0 = time.perf_counter()
        fc = sf.forecast(df=df, h=horizon, level=levels or None)
        self.last_runtime_s = time.perf_counter() - t0
        fc = fc.reset_index() if "unique_id" not in fc.columns else fc

        out = pd.DataFrame({ID: fc["unique_id"], TS: fc["ds"]})
        out["mean"] = self.combine(fc, aliases).to_numpy()
        for q in quantile_levels:
            lv = _level_for(q)
            if lv is None:
                cols = aliases
            else:
                side = "lo" if q < 0.5 else "hi"
                cols = [f"{a}-{side}-{lv}" for a in aliases]
            out[str(q)] = self.combine(fc, cols).to_numpy()
        # enforce non-crossing quantiles (intervals from different models can cross)
        qcols = [str(q) for q in sorted(quantile_levels)]
        out[qcols] = np.sort(out[qcols].to_numpy(), axis=1)
        return out

    def _predict_arrays(self, contexts, horizon, quantile_levels, freq) -> ArrayForecast:
        raise NotImplementedError("statsforecast models work on DataFrames; use predict()")
