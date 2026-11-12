"""AutoTheta: Theta method family (standard / optimised / dynamic), auto-selected per series. Fast and strong on monthly/quarterly data."""

from __future__ import annotations

from fchub.base import ModelInfo
from fchub.models._shared.statsforecast_base import (
    STATSFORECAST_LICENSE,
    STATSFORECAST_SOURCE,
    StatsForecastModel,
)


class AutoThetaForecaster(StatsForecastModel):
    info = ModelInfo(
        name="AutoTheta",
        family="statistical",
        source=STATSFORECAST_SOURCE,
        license=STATSFORECAST_LICENSE,
        notes="Theta method family (standard / optimised / dynamic), auto-selected per series. Fast and strong on monthly/quarterly data.",
    )

    def build_models(self, season_length: int) -> list:
        from statsforecast.models import AutoTheta

        return [AutoTheta(season_length=season_length)]
