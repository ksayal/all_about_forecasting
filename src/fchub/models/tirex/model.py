"""TiRex (NX-AI, 2025): 35M-parameter xLSTM (sLSTM) forecaster.

Small and fast, with strong short- and long-horizon accuracy. On CPU/MPS it
uses the pure-torch backend; on NVIDIA GPUs `pip install tirex-ts[cuda]` enables
custom kernels. Native output: deciles 0.1..0.9 (mean = median).
"""

from __future__ import annotations

import numpy as np

from fchub.base import ArrayForecast, Forecaster, ModelInfo, pick_device, select_quantiles

_NATIVE_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


class TiRexForecaster(Forecaster):
    info = ModelInfo(
        name="TiRex",
        family="foundation",
        source="https://github.com/NX-AI/tirex (pip: tirex-ts)",
        license="NXAI Community License (restricts commercial use above a revenue threshold; read before production use)",
        weights="NX-AI/TiRex",
    )

    def __init__(self, model_id: str = "NX-AI/TiRex", device: str | None = None, batch_size: int = 512, **kwargs):
        super().__init__(model_id=model_id, device=device, batch_size=batch_size)
        from tirex import load_model

        self.device = pick_device(device)
        self.batch_size = batch_size
        backend = "cuda" if self.device.startswith("cuda") else "torch"
        self.model = load_model(model_id, device=self.device, backend=backend)

    def _predict_arrays(self, contexts, horizon, quantile_levels, freq) -> ArrayForecast:
        quant, mean = self.model.forecast(
            context=[np.asarray(c, dtype=np.float32) for c in contexts],
            prediction_length=horizon,
            output_type="numpy",
            batch_size=self.batch_size,
        )
        q = select_quantiles(np.asarray(quant)[:, :horizon], _NATIVE_LEVELS, quantile_levels)
        return ArrayForecast(mean=np.asarray(mean)[:, :horizon], quantiles=q, quantile_levels=quantile_levels)
