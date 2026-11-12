"""Toto-1.0 (Datadog, 2025): 151M decoder-only model built for observability metrics.

Autoregressive with a Student-t mixture output, so forecasts come from
sampling: we draw ``num_samples`` paths and take empirical quantiles and the
sample median as the point forecast (the library's own recommendation).
This sampling is why Toto is one of the slower models on the leaderboard.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fchub.base import ArrayForecast, Forecaster, ModelInfo, pick_device, quantiles_from_samples


class TotoModel(Forecaster):
    info = ModelInfo(
        name="Toto-1.0",
        family="foundation",
        source="https://github.com/DataDog/toto (pip: toto-ts)",
        license="Apache-2.0 (code and weights)",
        weights="Datadog/Toto-Open-Base-1.0",
    )

    def __init__(
        self,
        model_id: str = "Datadog/Toto-Open-Base-1.0",
        device: str | None = None,
        num_samples: int = 256,
        samples_per_batch: int = 256,
        max_context: int = 4096,
        compile: bool = False,
        **kwargs,
    ):
        super().__init__(model_id=model_id, device=device, num_samples=num_samples, max_context=max_context)
        from toto.inference.forecaster import TotoForecaster as _LibForecaster
        from toto.model.toto import Toto

        self.device = pick_device(device)
        self.num_samples = num_samples
        self.samples_per_batch = samples_per_batch
        self.max_context = max_context
        toto = Toto.from_pretrained(model_id).to(self.device)
        if compile:
            toto.compile()
        self.forecaster = _LibForecaster(toto.model)

    def _predict_arrays(self, contexts, horizon, quantile_levels, freq) -> ArrayForecast:
        import torch
        from toto.data.util.dataset import MaskedTimeseries

        step_s = int(pd.tseries.frequencies.to_offset(freq).nanos // 1e9) if _is_fixed(freq) else 86400
        means, quants = [], []
        # one series at a time keeps variable-length contexts simple and memory bounded
        for ctx in contexts:
            x = torch.tensor(np.asarray(ctx[-self.max_context :], dtype=np.float32), device=self.device)[None, :]
            inputs = MaskedTimeseries(
                series=x,
                padding_mask=~torch.isnan(x),
                id_mask=torch.zeros_like(x),
                timestamp_seconds=torch.zeros_like(x, dtype=torch.long),  # unused by the released model
                time_interval_seconds=torch.full((1,), step_s, device=self.device),
            )
            with torch.inference_mode():
                fc = self.forecaster.forecast(
                    inputs,
                    prediction_length=horizon,
                    num_samples=self.num_samples,
                    samples_per_batch=self.samples_per_batch,
                )
            samples = fc.samples[0, 0].float().cpu().numpy()  # (h, num_samples)
            means.append(np.median(samples, axis=-1))
            quants.append(quantiles_from_samples(samples[None], quantile_levels)[0])
        return ArrayForecast(mean=np.stack(means), quantiles=np.stack(quants), quantile_levels=quantile_levels)


def _is_fixed(freq: str) -> bool:
    try:
        pd.tseries.frequencies.to_offset(freq).nanos
        return True
    except ValueError:  # calendar freqs like "MS" have no fixed length
        return False
