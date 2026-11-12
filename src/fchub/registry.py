"""Name -> model class, imported lazily.

Lazy import matters: each model's heavy dependencies (torch, transformers,
uni2ts, ...) are only installed in that model's own environment, so importing
the registry must never import a model module eagerly.
"""

from __future__ import annotations

import importlib

from fchub.base import Forecaster

# leaderboard name -> "module:Class"
MODELS: dict[str, str] = {
    # foundation models
    "Chronos-2": "fchub.models.chronos2:Chronos2Forecaster",
    "TimesFM-2.5": "fchub.models.timesfm25:TimesFM25Forecaster",
    "TiRex": "fchub.models.tirex:TiRexForecaster",
    "Toto-1.0": "fchub.models.toto:TotoModel",
    "TabPFN-TS": "fchub.models.tabpfn_ts:TabPFNTSForecaster",
    "COSMIC": "fchub.models.cosmic:CosmicForecaster",
    "Moirai-2.0": "fchub.models.moirai2:Moirai2Forecaster",
    "Chronos-Bolt": "fchub.models.chronos_bolt:ChronosBoltForecaster",
    "Sundial": "fchub.models.sundial:SundialForecaster",
    # statistical baselines
    "StatEnsemble": "fchub.models.stat_ensemble:StatEnsembleForecaster",
    "AutoETS": "fchub.models.autoets:AutoETSForecaster",
    "AutoARIMA": "fchub.models.autoarima:AutoARIMAForecaster",
    "AutoTheta": "fchub.models.autotheta:AutoThetaForecaster",
    "Naive": "fchub.models.naive:NaiveForecaster",
    "SeasonalNaive": "fchub.models.seasonal_naive:SeasonalNaiveForecaster",
}

STAT_MODELS = ["StatEnsemble", "AutoETS", "AutoARIMA", "AutoTheta", "Naive", "SeasonalNaive"]

# model name -> virtualenv name (scripts/make_env.sh <env>). The stat baselines share
# one light env; every foundation model gets its own because their pins conflict
# (e.g. uni2ts needs torch<2.5 while toto-ts pins torch==2.7.0).
ENV_OF: dict[str, str] = {
    name: ("stats" if name in STAT_MODELS else path.split(":")[0].rsplit(".", 1)[1])
    for name, path in MODELS.items()
}


def get_model_class(name: str) -> type[Forecaster]:
    if name not in MODELS:
        raise KeyError(f"Unknown model {name!r}. Available: {', '.join(MODELS)}")
    module, cls = MODELS[name].split(":")
    try:
        return getattr(importlib.import_module(module), cls)
    except ImportError as e:
        env = ENV_OF[name]
        raise ImportError(
            f"{name} dependencies are not installed in this environment ({e}). "
            f"Create its environment with: scripts/make_env.sh {env}"
        ) from e


def load_model(name: str, **kwargs) -> Forecaster:
    """Instantiate a model by its leaderboard name, e.g. ``load_model("TiRex")``."""
    return get_model_class(name)(**kwargs)
