"""Map a pandas frequency string to a seasonal period (used by stat models + MASE)."""

from __future__ import annotations

import pandas as pd

# Base unit -> seasonal period. Mirrors the conventions used by GIFT-Eval / fev.
_BASE_SEASONALITY = {
    "S": 60,  # seconds -> minute cycle
    "T": 1440,  # minutes -> daily cycle (divided by the multiple below)
    "MIN": 1440,
    "H": 24,
    "D": 7,
    "B": 5,
    "W": 52,
    "M": 12,
    "ME": 12,
    "MS": 12,
    "Q": 4,
    "QE": 4,
    "QS": 4,
    "A": 1,
    "Y": 1,
    "YE": 1,
    "YS": 1,
}


def season_length(freq: str) -> int:
    """Return the dominant seasonal period for ``freq`` (1 = no seasonality).

    Examples: "h" -> 24, "15min" -> 96, "D" -> 7, "W" -> 52, "MS" -> 12.
    """
    offset = pd.tseries.frequencies.to_offset(freq)
    name = offset.name.upper().split("-")[0]  # e.g. "W-SUN" -> "W"
    n = max(int(offset.n), 1)
    # strip the multiple pandas puts in front, e.g. "15MIN" -> "MIN"
    base = name.lstrip("0123456789")
    period = _BASE_SEASONALITY.get(base, 1)
    if base in {"S", "T", "MIN", "H"}:
        period = max(period // n, 1)
    return int(period)
