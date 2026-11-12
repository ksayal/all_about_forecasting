"""Combine results/<model>.csv files into the leaderboard table.

    python -m fchub.leaderboard                 # reads ./results, ranks by SQL
    python -m fchub.leaderboard --metric MASE   # point-forecast ranking instead
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from fchub.metrics import leaderboard


def load_results(results_dir: str | Path) -> pd.DataFrame:
    files = sorted(Path(results_dir).glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"no result CSVs in {results_dir}; run python -m fchub.benchmark first")
    raw = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    # average windows within a task; runtime is summed (time to do the whole task)
    return raw.groupby(["model", "task"], as_index=False).agg(
        MASE=("MASE", "mean"), SQL=("SQL", "mean"), WQL=("WQL", "mean"),
        runtime_s=("runtime_s", "sum"), failed=("failed", "max"),
    )


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("results", nargs="?", default="results")
    p.add_argument("--metric", default="SQL", choices=["SQL", "MASE", "WQL"])
    p.add_argument("--baseline", default="SeasonalNaive")
    p.add_argument("--markdown", help="also write the table to this .md file")
    a = p.parse_args(argv)

    table = leaderboard(load_results(a.results), metric=a.metric, baseline=a.baseline)
    with pd.option_context("display.width", 120):
        print(table)
    if a.markdown:
        Path(a.markdown).write_text(table.to_markdown())


if __name__ == "__main__":
    main()
