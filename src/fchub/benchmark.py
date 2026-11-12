"""Run models over the evaluation tasks and write per-task scores.

Usage (inside the model's own environment, see scripts/make_env.sh):

    python -m fchub.benchmark --models TiRex                     # all sample tasks
    python -m fchub.benchmark --models AutoETS,AutoTheta --tasks air_passengers
    python -m fchub.benchmark --models Chronos-2 --model-kwargs '{"device": "cpu"}'

Each model writes ``results/<model>.csv``; ``python -m fchub.leaderboard``
combines whatever is in ``results/`` into the leaderboard table. Keeping one
file per model is what lets every model run in a different virtualenv.

Failures (exceptions, NaNs, wrong shapes) are recorded and the window is
scored with a seasonal-naive forecast instead, the same penalty fev applies,
so a model can't improve its rank by crashing on hard tasks.
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from fchub.base import DEFAULT_QUANTILES, ID, TARGET, TS, future_index
from fchub.data import SAMPLE_TASKS, Task, load_dataset, rolling_windows
from fchub.freq import season_length
from fchub.metrics import evaluate_forecast
from fchub.registry import MODELS, load_model


def seasonal_naive_fallback(context_df: pd.DataFrame, task: Task, quantiles) -> pd.DataFrame:
    """Dependency-free seasonal naive, used to score failed windows."""
    m = season_length(task.freq)
    frames = []
    for item_id, g in context_df.sort_values([ID, TS]).groupby(ID, sort=False):
        y = g[TARGET].to_numpy(dtype=float)
        period = m if len(y) >= m else 1
        last = y[-period:]
        pred = np.resize(last, task.horizon)
        f = pd.DataFrame({ID: item_id, TS: future_index(g[TS].iloc[-1], task.freq, task.horizon), "mean": pred})
        for q in quantiles:
            f[str(q)] = pred
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


def run_model(
    model_name: str,
    tasks: list[Task],
    model_kwargs: dict | None = None,
    quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
    save_forecasts: Path | None = None,
) -> pd.DataFrame:
    t_load = time.perf_counter()
    model = load_model(model_name, **(model_kwargs or {}))
    print(f"[{model_name}] loaded in {time.perf_counter() - t_load:.1f}s")

    rows = []
    for task in tasks:
        df = load_dataset(task.dataset)
        m = season_length(task.freq)
        for k, ctx, act in rolling_windows(df, task):
            err, failed = "", False
            t0 = time.perf_counter()
            try:
                fc = model.predict(ctx, task.horizon, task.freq, quantiles)
                qcols = ["mean"] + [str(q) for q in quantiles]
                if fc[qcols].isna().any().any():
                    raise ValueError("forecast contains NaN")
            except Exception as e:  # noqa: BLE001 - we want to keep benchmarking
                failed, err = True, f"{type(e).__name__}: {e}"
                traceback.print_exc()
                fc = seasonal_naive_fallback(ctx, task, quantiles)
            runtime = time.perf_counter() - t0
            scores = evaluate_forecast(fc, ctx, act, m, quantiles)
            rows.append(
                {"model": model_name, "task": task.name, "window": k, **scores,
                 "runtime_s": runtime, "failed": failed, "error": err}
            )
            print(f"[{model_name}] {task.name} w{k}: MASE={scores['MASE']:.3f} SQL={scores['SQL']:.3f} "
                  f"({runtime:.2f}s){' FAILED' if failed else ''}")
            if save_forecasts is not None:
                out = save_forecasts / model_name / f"{task.dataset}_h{task.horizon}_w{k}.csv"
                out.parent.mkdir(parents=True, exist_ok=True)
                fc.to_csv(out, index=False)
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", required=True, help=f"comma-separated, from: {', '.join(MODELS)}")
    p.add_argument("--tasks", default="all", help="comma-separated dataset names, or 'all'")
    p.add_argument("--out", default="results", help="output directory")
    p.add_argument("--model-kwargs", default="{}", help="JSON dict passed to the model constructor")
    p.add_argument("--save-forecasts", action="store_true", help="also write each forecast to CSV")
    a = p.parse_args(argv)

    tasks = SAMPLE_TASKS if a.tasks == "all" else [t for t in SAMPLE_TASKS if t.dataset in a.tasks.split(",")]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in [m.strip() for m in a.models.split(",")]:
        try:
            res = run_model(name, tasks, json.loads(a.model_kwargs),
                            save_forecasts=out / "forecasts" if a.save_forecasts else None)
        except (ImportError, NotImplementedError, OSError) as e:
            # model could not even be loaded (missing env, no weights, placeholder)
            print(f"[{name}] skipped: {e}")
            continue
        res.to_csv(out / f"{name}.csv", index=False)
        print(f"[{name}] wrote {out / f'{name}.csv'}")


if __name__ == "__main__":
    main()
