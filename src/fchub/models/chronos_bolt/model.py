"""Chronos-Bolt (Amazon, 2024): T5 encoder-decoder that forecasts patches directly.

Up to ~250x faster than the original Chronos and more accurate. Trained on the
deciles 0.1..0.9; other quantile levels are interpolated by the library.
Sizes: tiny (9M), mini (21M), small (48M), base (205M).
"""

from __future__ import annotations

import numpy as np

from fchub.base import ArrayForecast, Forecaster, ModelInfo, pick_device


class ChronosBoltForecaster(Forecaster):
    info = ModelInfo(
        name="Chronos-Bolt",
        family="foundation",
        source="https://github.com/amazon-science/chronos-forecasting (pip: chronos-forecasting)",
        license="Apache-2.0 (code and weights)",
        weights="amazon/chronos-bolt-base",
    )

    def __init__(
        self,
        model_id: str = "amazon/chronos-bolt-base",
        revision: str | None = None,
        device: str | None = None,
        batch_size: int = 256,
        **kwargs,
    ):
        super().__init__(model_id=model_id, revision=revision, device=device, batch_size=batch_size)
        import torch
        from chronos import BaseChronosPipeline

        self.device = pick_device(device)
        self.batch_size = batch_size
        self.pipeline = BaseChronosPipeline.from_pretrained(
            model_id, revision=revision, device_map=self.device, torch_dtype=torch.float32
        )

    def _predict_arrays(self, contexts, horizon, quantile_levels, freq) -> ArrayForecast:
        import torch

        qs, ms = [], []
        for start in range(0, len(contexts), self.batch_size):
            batch = [torch.tensor(c, dtype=torch.float32) for c in contexts[start : start + self.batch_size]]
            q, m = self.pipeline.predict_quantiles(
                batch, prediction_length=horizon, quantile_levels=list(quantile_levels)
            )
            qs.append(q.float().cpu().numpy())  # (b, h, q)
            ms.append(m.float().cpu().numpy())  # (b, h)
        return ArrayForecast(
            mean=np.concatenate(ms), quantiles=np.concatenate(qs), quantile_levels=quantile_levels
        )
