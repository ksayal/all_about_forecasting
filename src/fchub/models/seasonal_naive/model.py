"""SeasonalNaive: Repeats the value from one season ago. This is the reference model for the leaderboard's Skill Score (skill = 0 by definition)."""

from __future__ import annotations

from fchub.base import ModelInfo
from fchub.models._shared.statsforecast_base import (
    STATSFORECAST_LICENSE,
    STATSFORECAST_SOURCE,
    StatsForecastModel,
)


class SeasonalNaiveForecaster(StatsForecastModel):
    info = ModelInfo(
        name="SeasonalNaive",
        family="statistical",
        source=STATSFORECAST_SOURCE,
        license=STATSFORECAST_LICENSE,
        notes="Repeats the value from one season ago. This is the reference model for the leaderboard's Skill Score (skill = 0 by definition).",
    )

    def build_models(self, season_length: int) -> list:
        from statsforecast.models import SeasonalNaive

        return [SeasonalNaive(season_length=season_length)]
