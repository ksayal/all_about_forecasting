"""Sundial (Tsinghua THUML, ICML 2025): generative decoder-only model with a flow-matching head.

Trained on ~1 trillion time points (TimeBench). Produces *samples* of future
paths, so we draw ``num_samples`` and take empirical quantiles / the mean.
Loaded through Hugging Face transformers with ``trust_remote_code=True``,
which runs modelling code downloaded from the model repo; pin ``revision``
to a commit hash once you've reviewed it.
"""

from __future__ import annotations

import numpy as np

from fchub.base import ArrayForecast, Forecaster, ModelInfo, pick_device, quantiles_from_samples


class SundialForecaster(Forecaster):
    info = ModelInfo(
        name="Sundial",
        family="foundation",
        source="https://github.com/thuml/Sundial (weights + code via transformers)",
        license="Apache-2.0 (per the model card; verify before production use)",
        weights="thuml/sundial-base-128m",
    )

    def __init__(
        self,
        model_id: str = "thuml/sundial-base-128m",
        revision: str | None = None,
        num_samples: int = 100,
        max_context: int = 2880,
        device: str | None = None,
        **kwargs,
    ):
        super().__init__(model_id=model_id, revision=revision, num_samples=num_samples, max_context=max_context)
        from transformers import AutoModelForCausalLM

        self.device = pick_device(device)
        self.num_samples = num_samples
        self.max_context = max_context
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, revision=revision, trust_remote_code=True
        ).to(self.device)
        self.model.eval()

    def _predict_arrays(self, contexts, horizon, quantile_levels, freq) -> ArrayForecast:
        import torch

        means, quants = [], []
        for ctx in contexts:  # per series: the model expects equal-length batches
            x = np.asarray(ctx[-self.max_context :], dtype=np.float32)
            x = np.nan_to_num(x, nan=float(np.nanmean(x)) if np.isfinite(x).any() else 0.0)
            seq = torch.tensor(x, device=self.device)[None, :]
            with torch.inference_mode():
                out = self.model.generate(seq, max_new_tokens=horizon, num_samples=self.num_samples)
            samples = out[0].float().cpu().numpy()  # (num_samples, h)
            samples = samples[:, :horizon].T  # (h, num_samples)
            means.append(samples.mean(axis=-1))
            quants.append(quantiles_from_samples(samples[None], quantile_levels)[0])
        return ArrayForecast(mean=np.stack(means), quantiles=np.stack(quants), quantile_levels=quantile_levels)
