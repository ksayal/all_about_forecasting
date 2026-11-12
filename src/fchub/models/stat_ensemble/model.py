"""Stat. Ensemble: equal-weight average of AutoETS, AutoARIMA and AutoTheta.

Each member is fitted per series; the point forecast and every quantile are
averaged across members. Robust but slow (it pays AutoARIMA's cost), which is
why it shows the highest runtime and most failures on the leaderboard.
"""

from __future__ import annotations

import pandas as pd

from fchub.base import ModelInfo
from fchub.models._shared.statsforecast_base import (
    STATSFORECAST_LICENSE,
    STATSFORECAST_SOURCE,
    StatsForecastModel,
)


class StatEnsembleForecaster(StatsForecastModel):
    info = ModelInfo(
        name="StatEnsemble",
        family="statistical",
        source=STATSFORECAST_SOURCE,
        license=STATSFORECAST_LICENSE,
        notes="Mean of AutoETS + AutoARIMA + AutoTheta forecasts and quantiles.",
    )

    # Seasonal ARIMA order search scales badly with long seasonal series
    # (e.g. 2000+ weekly points with season 52 took >15 min). Use the last
    # 1000 points by default; pass max_context=None to use the full history.
    default_max_context = 1000

    def build_models(self, season_length: int) -> list:
        from statsforecast.models import AutoARIMA, AutoETS, AutoTheta

        return [
            AutoETS(season_length=season_length),
            AutoARIMA(season_length=season_length),
            AutoTheta(season_length=season_length),
        ]

    def combine(self, fc: pd.DataFrame, cols: list[str]) -> pd.Series:
        return fc[cols].mean(axis=1)
