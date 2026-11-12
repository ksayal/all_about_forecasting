"""TabPFN-TS (Prior Labs, 2025): forecasting as tabular regression.

Each time step becomes a table row with features (running index, calendar
features, auto-detected seasonal sine/cosine), and the TabPFN tabular
foundation model predicts the target in-context. Because it needs real
timestamps (calendar features) it overrides ``predict`` and takes the
DataFrame directly.

Modes:
  * "local"  - downloads TabPFN weights and runs on your machine (slow without a GPU;
               this is the 305 s median runtime on the leaderboard).
  * "client" - sends data to Prior Labs' hosted API (needs a free API token via
               `tabpfn_client`); do not use with confidential data.

License: tabpfn-time-series code is Apache-2.0; TabPFN model weights carry
Prior Labs' own license terms (check the current terms on the model card,
newer TabPFN versions are non-commercial).
"""

from __future__ import annotations

import time

import numpy as np

from fchub.base import DEFAULT_QUANTILES, ID, TARGET, TS, ArrayForecast, Forecaster, ModelInfo, validate_long_df


class TabPFNTSForecaster(Forecaster):
    info = ModelInfo(
        name="TabPFN-TS",
        family="foundation",
        source="https://github.com/PriorLabs/tabpfn-time-series (pip: tabpfn-time-series)",
        license="Code Apache-2.0; TabPFN weights under Prior Labs license (check model card)",
        weights="Prior-Labs TabPFN (downloaded by the tabpfn package)",
        supports_covariates=True,
    )

    def __init__(self, mode: str = "local", max_context_length: int = 4096, **kwargs):
        super().__init__(mode=mode, max_context_length=max_context_length)
        from tabpfn_time_series import TabPFNMode, TabPFNTSPipeline

        self.pipeline = TabPFNTSPipeline(
            max_context_length=max_context_length,
            tabpfn_mode=TabPFNMode(mode),
            compile=False,  # torch.compile only pays off on long CUDA runs
        )

    def predict(self, context_df, horizon, freq, quantile_levels=DEFAULT_QUANTILES):
        validate_long_df(context_df)
        t0 = time.perf_counter()
        pred = self.pipeline.predict_df(
            context_df[[ID, TS, TARGET]],
            prediction_length=horizon,
            quantiles=list(quantile_levels),
        )
        self.last_runtime_s = time.perf_counter() - t0
        pred = pred.reset_index()
        out = pred[[ID, TS]].copy()
        out["mean"] = pred[TARGET].to_numpy()
        for q in quantile_levels:
            col = q if q in pred.columns else str(q)
            out[str(q)] = pred[col].to_numpy()
        qcols = [str(q) for q in sorted(quantile_levels)]
        out[qcols] = np.sort(out[qcols].to_numpy(), axis=1)
        return out

    def _predict_arrays(self, contexts, horizon, quantile_levels, freq) -> ArrayForecast:
        raise NotImplementedError("TabPFN-TS needs timestamps; use predict()")
