"""COSMIC (AWS AI Labs, 2025) - PLACEHOLDER, not runnable.

Paper: "Zero-Shot Time Series Forecasting with Covariates via In-Context
Learning", Auer et al., arXiv:2506.03128. An encoder-decoder (efficient T5)
model trained with "informative covariate augmentation" so it can use
covariates zero-shot. Sizes: mini 20M / small 40M / base 200M.

As of 2026-11 we found no public code or weights. The leaderboard numbers come
from the authors' own submission. Re-implementing the architecture without
the pretrained weights would give an untrained model, not COSMIC, so this
module deliberately refuses to run. If AWS releases it (most likely inside
`chronos-forecasting` or AutoGluon), implement ``_predict_arrays`` the same
way as ``models/chronos_bolt``.
"""

from __future__ import annotations

from fchub.base import ArrayForecast, Forecaster, ModelInfo


class CosmicForecaster(Forecaster):
    info = ModelInfo(
        name="COSMIC",
        family="foundation",
        source="Paper only: https://arxiv.org/abs/2506.03128 (no public code/weights found)",
        license="n/a",
        supports_covariates=True,
        notes="Placeholder until AWS publishes code or weights.",
    )

    def __init__(self, **kwargs):
        raise NotImplementedError(
            "COSMIC has no public code or weights yet (arXiv:2506.03128). "
            "See src/fchub/models/cosmic/README.md."
        )

    def _predict_arrays(self, contexts, horizon, quantile_levels, freq) -> ArrayForecast:  # pragma: no cover
        raise NotImplementedError
