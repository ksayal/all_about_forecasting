"""Forecast accuracy metrics and leaderboard aggregation.

Per task (one dataset + horizon):
  * MASE - mean absolute error of the point forecast, scaled by the in-sample
           seasonal-naive MAE of each series (scale-free, comparable across series).
  * SQL  - scaled quantile loss: the same scaling applied to the mean pinball
           loss over all quantile levels (the probabilistic analogue of MASE).
  * WQL  - weighted quantile loss: pinball loss normalised by sum(|y|).

Across tasks (the leaderboard columns):
  * Skill score (%) = 100 * (1 - geometric mean over tasks of error / SeasonalNaive error).
    SeasonalNaive therefore scores exactly 0; negative = worse than SeasonalNaive.
  * Avg. win rate (%) = for each model, the share of (task, opponent) pairs where it has
    a lower error than the opponent (ties count 0.5), averaged over opponents.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fchub.base import ID, TARGET, TS


def _seasonal_scale(history: np.ndarray, m: int) -> float:
    history = history[~np.isnan(history)]
    if len(history) <= m:
        m = 1
    if len(history) <= 1:
        return np.nan
    diffs = np.abs(history[m:] - history[:-m])
    scale = float(np.mean(diffs)) if diffs.size else np.nan
    return scale if scale > 0 else np.nan


def pinball(y: np.ndarray, q_pred: np.ndarray, level: float) -> np.ndarray:
    diff = y - q_pred
    return np.maximum(level * diff, (level - 1) * diff)


def evaluate_forecast(
    forecast: pd.DataFrame,
    history: pd.DataFrame,
    actuals: pd.DataFrame,
    season_length: int,
    quantile_levels: tuple[float, ...],
) -> dict[str, float]:
    """Score one model on one task. All frames are long-format."""
    qcols = [str(q) for q in quantile_levels]
    merged = forecast.merge(actuals[[ID, TS, TARGET]], on=[ID, TS], how="inner")
    if len(merged) != len(actuals):
        raise ValueError(f"forecast covers {len(merged)} of {len(actuals)} target rows")

    mase_terms, sql_terms = [], []
    wql_num, wql_den = 0.0, 0.0
    for item_id, g in merged.groupby(ID, sort=False):
        hist = history.loc[history[ID] == item_id, TARGET].to_numpy(dtype=float)
        scale = _seasonal_scale(hist, season_length)
        y = g[TARGET].to_numpy(dtype=float)
        mae = np.nanmean(np.abs(y - g["mean"].to_numpy(dtype=float)))
        ql = np.stack([pinball(y, g[c].to_numpy(dtype=float), q) for c, q in zip(qcols, quantile_levels)])
        if np.isfinite(scale):
            mase_terms.append(mae / scale)
            sql_terms.append(np.nanmean(ql) * 2 / scale)  # x2 so SQL ~ MASE for a perfect median
        wql_num += np.nansum(ql) * 2 / len(quantile_levels)
        wql_den += np.nansum(np.abs(y))

    return {
        "MASE": float(np.mean(mase_terms)) if mase_terms else np.nan,
        "SQL": float(np.mean(sql_terms)) if sql_terms else np.nan,
        "WQL": float(wql_num / wql_den) if wql_den > 0 else np.nan,
    }


def leaderboard(
    results: pd.DataFrame,
    metric: str = "SQL",
    baseline: str = "SeasonalNaive",
) -> pd.DataFrame:
    """Aggregate per-task results into the leaderboard table.

    ``results`` needs columns: model, task, <metric>, runtime_s, failed (bool).
    Failed runs should already carry the baseline's error (fev convention), so
    a failure is penalised but does not drop the task.
    """
    pivot = results.pivot_table(index="task", columns="model", values=metric, aggfunc="first")
    models = list(pivot.columns)

    # skill score vs baseline
    if baseline in pivot.columns:
        ratio = pivot.div(pivot[baseline], axis=0).clip(lower=1e-6)
        skill = 100 * (1 - np.exp(np.log(ratio).mean(axis=0)))
    else:
        skill = pd.Series(np.nan, index=models)

    # pairwise win rate
    win = {}
    for a in models:
        rates = []
        for b in models:
            if a == b:
                continue
            both = pivot[[a, b]].dropna()
            if both.empty:
                continue
            w = (both[a] < both[b]).mean() + 0.5 * (both[a] == both[b]).mean()
            rates.append(w)
        win[a] = 100 * float(np.mean(rates)) if rates else np.nan

    agg = results.groupby("model").agg(
        median_runtime_s=("runtime_s", "median"),
        failures=("failed", "sum"),
    )
    table = pd.DataFrame(
        {
            "Avg. Win Rate (%)": pd.Series(win),
            "Skill Score (%)": skill,
        }
    ).join(agg)
    table = table.rename(columns={"median_runtime_s": "Median runtime (s)", "failures": "#Failures"})
    return table.sort_values("Avg. Win Rate (%)", ascending=False).round(1)
