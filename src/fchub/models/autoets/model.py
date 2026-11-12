"""AutoETS: Exponential smoothing (Error/Trend/Seasonality). Picks the best ETS variant per series by AICc."""

from __future__ import annotations

from fchub.base import ModelInfo
from fchub.models._shared.statsforecast_base import (
    STATSFORECAST_LICENSE,
    STATSFORECAST_SOURCE,
    StatsForecastModel,
)


class AutoETSForecaster(StatsForecastModel):
    info = ModelInfo(
        name="AutoETS",
        family="statistical",
        source=STATSFORECAST_SOURCE,
        license=STATSFORECAST_LICENSE,
        notes="Exponential smoothing (Error/Trend/Seasonality). Picks the best ETS variant per series by AICc.",
    )

    def build_models(self, season_length: int) -> list:
        from statsforecast.models import AutoETS

        return [AutoETS(season_length=season_length)]
