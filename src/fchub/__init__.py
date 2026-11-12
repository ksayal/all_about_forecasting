"""fchub: one interface, many forecasting models.

    from fchub import load_model
    model = load_model("TiRex")
    forecast = model.predict(context_df, horizon=24, freq="h")
"""

from fchub.base import DEFAULT_QUANTILES, Forecaster, ModelInfo
from fchub.registry import MODELS, load_model

__all__ = ["DEFAULT_QUANTILES", "Forecaster", "ModelInfo", "MODELS", "load_model"]
__version__ = "0.1.0"
