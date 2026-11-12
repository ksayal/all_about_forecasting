"""AutoARIMA: Automatic (seasonal) ARIMA order selection, a port of R's forecast::auto.arima. Slow on long series."""

from __future__ import annotations

from fchub.base import ModelInfo
from fchub.models._shared.statsforecast_base import (
    STATSFORECAST_LICENSE,
    STATSFORECAST_SOURCE,
    StatsForecastModel,
)


class AutoARIMAForecaster(StatsForecastModel):
    info = ModelInfo(
        name="AutoARIMA",
        family="statistical",
        source=STATSFORECAST_SOURCE,
        license=STATSFORECAST_LICENSE,
        notes="Automatic (seasonal) ARIMA order selection, a port of R's forecast::auto.arima. Slow on long series.",
    )

    # Seasonal ARIMA order search scales badly with long seasonal series
    # (e.g. 2000+ weekly points with season 52 took >15 min). Use the last
    # 1000 points by default; pass max_context=None to use the full history.
    default_max_context = 1000

    def build_models(self, season_length: int) -> list:
        from statsforecast.models import AutoARIMA

        return [AutoARIMA(season_length=season_length)]
