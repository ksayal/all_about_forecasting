"""Moirai-2.0 (Salesforce, 2025): decoder-only successor to Moirai / Moirai-MoE.

Switched to quantile loss + multi-token prediction; the "small" 11M model is
both faster and more accurate than earlier, larger Moirai versions.
Native output: deciles 0.1..0.9 (mean = median).

LICENSE WARNING: the weights are CC-BY-NC-4.0, i.e. research / non-commercial only.
"""

from __future__ import annotations

import numpy as np

from fchub.base import ArrayForecast, Forecaster, ModelInfo, pick_device, select_quantiles

_NATIVE_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


class Moirai2Forecaster(Forecaster):
    info = ModelInfo(
        name="Moirai-2.0",
        family="foundation",
        source="https://github.com/SalesforceAIResearch/uni2ts (pip: uni2ts)",
        license="Code Apache-2.0; weights CC-BY-NC-4.0 (non-commercial)",
        weights="Salesforce/moirai-2.0-R-small",
    )

    def __init__(
        self,
        model_id: str = "Salesforce/moirai-2.0-R-small",
        context_length: int = 1680,
        batch_size: int = 128,
        device: str | None = None,
        **kwargs,
    ):
        super().__init__(model_id=model_id, context_length=context_length, batch_size=batch_size)
        from uni2ts.model.moirai2 import Moirai2Module

        # uni2ts pins torch<2.5, whose MPS support is patchy, so default to cpu unless forced
        self.device = device or ("cuda" if pick_device() == "cuda" else "cpu")
        self.context_length = context_length
        self.batch_size = batch_size
        self.module = Moirai2Module.from_pretrained(model_id)

    def _predict_arrays(self, contexts, horizon, quantile_levels, freq) -> ArrayForecast:
        from uni2ts.model.moirai2 import Moirai2Forecast

        # prediction_length is a constructor arg, so build a light wrapper per horizon;
        # the heavy pretrained module is shared.
        model = Moirai2Forecast(
            module=self.module,
            prediction_length=horizon,
            context_length=self.context_length,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        ).to(self.device)

        out = []
        for start in range(0, len(contexts), self.batch_size):
            batch = [np.asarray(c, dtype=np.float32) for c in contexts[start : start + self.batch_size]]
            pred = np.asarray(model.predict(past_target=batch))  # (b, Q, h[, 1])
            if pred.ndim == 4:
                pred = pred[..., 0]
            out.append(np.swapaxes(pred, 1, 2))  # -> (b, h, Q)
        native = np.concatenate(out)[:, :horizon]
        q = select_quantiles(native, _NATIVE_LEVELS, quantile_levels)
        median = native[..., _NATIVE_LEVELS.index(0.5)]
        return ArrayForecast(mean=median, quantiles=q, quantile_levels=quantile_levels)
