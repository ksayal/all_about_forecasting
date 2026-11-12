"""TimesFM-2.5 (Google, 2025): 200M decoder-only foundation model.

Context up to 16k steps, optional continuous quantile head (up to 1k horizon).
The model must be *compiled* with a ForecastConfig (max context / horizon)
before forecasting; we compile lazily and recompile when the horizon grows.
Native output: index 0 = mean, indices 1..9 = deciles 0.1..0.9.
"""

from __future__ import annotations

import numpy as np

from fchub.base import ArrayForecast, Forecaster, ModelInfo, select_quantiles

_NATIVE_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


class TimesFM25Forecaster(Forecaster):
    info = ModelInfo(
        name="TimesFM-2.5",
        family="foundation",
        source="https://github.com/google-research/timesfm (pip: timesfm[torch])",
        license="Apache-2.0 (code and weights)",
        weights="google/timesfm-2.5-200m-pytorch",
        supports_covariates=True,
    )

    def __init__(
        self,
        model_id: str = "google/timesfm-2.5-200m-pytorch",
        max_context: int = 4096,
        per_core_batch_size: int = 32,
        use_continuous_quantile_head: bool = True,
        torch_compile: bool = False,
        **kwargs,
    ):
        super().__init__(model_id=model_id, max_context=max_context, per_core_batch_size=per_core_batch_size)
        import timesfm

        self._timesfm = timesfm
        self.max_context = max_context
        self.per_core_batch_size = per_core_batch_size
        self.use_cq = use_continuous_quantile_head
        # torch.compile speeds up CUDA but is slow/flaky on CPU and Apple MPS, so off by default
        self.model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(model_id, torch_compile=torch_compile)
        self._compiled_horizon = 0

    def _ensure_compiled(self, horizon: int) -> None:
        if horizon <= self._compiled_horizon:
            return
        # round up to the output patch (128) so small horizon changes don't recompile
        max_horizon = int(np.ceil(horizon / 128) * 128)
        self.model.compile(
            self._timesfm.ForecastConfig(
                max_context=self.max_context,
                max_horizon=max_horizon,
                normalize_inputs=True,
                per_core_batch_size=self.per_core_batch_size,
                use_continuous_quantile_head=self.use_cq and max_horizon <= 1024,  # head supports <=1k
                force_flip_invariance=True,
                infer_is_positive=True,
                fix_quantile_crossing=True,
            )
        )
        self._compiled_horizon = max_horizon

    def _predict_arrays(self, contexts, horizon, quantile_levels, freq) -> ArrayForecast:
        self._ensure_compiled(horizon)
        point, quant = self.model.forecast(horizon=horizon, inputs=[np.asarray(c, dtype=np.float32) for c in contexts])
        point, quant = np.asarray(point)[:, :horizon], np.asarray(quant)[:, :horizon, :]
        q = select_quantiles(quant[..., 1:], _NATIVE_LEVELS, quantile_levels)
        return ArrayForecast(mean=point, quantiles=q, quantile_levels=quantile_levels)
