"""Chronos-2 (Amazon, 2025): encoder-only foundation model with group attention.

Handles univariate, multivariate and covariate-informed forecasting zero-shot.
Here we use the univariate path to match the leaderboard setting.
Output: quantile head (21 levels incl. 0.01..0.99); we pick the requested ones.
"""

from __future__ import annotations

import numpy as np

from fchub.base import ArrayForecast, Forecaster, ModelInfo, pick_device


class Chronos2Forecaster(Forecaster):
    info = ModelInfo(
        name="Chronos-2",
        family="foundation",
        source="https://github.com/amazon-science/chronos-forecasting (pip: chronos-forecasting)",
        license="Apache-2.0 (code and weights)",
        weights="amazon/chronos-2",
        supports_covariates=True,
    )

    def __init__(
        self,
        model_id: str = "amazon/chronos-2",
        revision: str | None = None,
        device: str | None = None,
        batch_size: int = 256,
        **kwargs,
    ):
        super().__init__(model_id=model_id, revision=revision, device=device, batch_size=batch_size)
        from chronos import BaseChronosPipeline

        self.device = pick_device(device)
        self.batch_size = batch_size
        self.pipeline = BaseChronosPipeline.from_pretrained(
            model_id, revision=revision, device_map=self.device
        )

    def _predict_arrays(self, contexts, horizon, quantile_levels, freq) -> ArrayForecast:
        quantiles, mean = self.pipeline.predict_quantiles(
            [np.asarray(c, dtype=np.float32) for c in contexts],
            prediction_length=horizon,
            quantile_levels=list(quantile_levels),
            batch_size=self.batch_size,
        )
        # Chronos-2 returns one tensor per series: (n_variates=1, h, q) and (1, h)
        q = np.stack([t.squeeze(0).float().cpu().numpy() for t in quantiles])
        m = np.stack([t.squeeze(0).float().cpu().numpy() for t in mean])
        return ArrayForecast(mean=m, quantiles=q, quantile_levels=quantile_levels)
