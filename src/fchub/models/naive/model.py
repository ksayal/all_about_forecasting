"""Naive: Repeats the last observed value. The simplest possible baseline; anything worse than this is broken."""

from __future__ import annotations

from fchub.base import ModelInfo
from fchub.models._shared.statsforecast_base import (
    STATSFORECAST_LICENSE,
    STATSFORECAST_SOURCE,
    StatsForecastModel,
)


class NaiveForecaster(StatsForecastModel):
    info = ModelInfo(
        name="Naive",
        family="statistical",
        source=STATSFORECAST_SOURCE,
        license=STATSFORECAST_LICENSE,
        notes="Repeats the last observed value. The simplest possible baseline; anything worse than this is broken.",
    )

    def build_models(self, season_length: int) -> list:
        from statsforecast.models import Naive

        return [Naive()]
