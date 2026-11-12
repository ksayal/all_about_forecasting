#!/usr/bin/env bash
# Run every model in its own env, then print the combined leaderboard.
# (Plain bash 3 syntax so it works with macOS's built-in /bin/bash.)
#
#   scripts/run_benchmark.sh                 # every model that has an env in .venvs/
#   scripts/run_benchmark.sh TiRex AutoETS   # just these
set -uo pipefail
cd "$(dirname "$0")/.."

env_of() {
  case "$1" in
    Naive|SeasonalNaive|AutoETS|AutoARIMA|AutoTheta|StatEnsemble) echo stats ;;
    Chronos-2) echo chronos2 ;;   Chronos-Bolt) echo chronos_bolt ;;
    TimesFM-2.5) echo timesfm25 ;; TiRex) echo tirex ;;
    Toto-1.0) echo toto ;;        Moirai-2.0) echo moirai2 ;;
    Sundial) echo sundial ;;      TabPFN-TS) echo tabpfn_ts ;;
    *) echo "" ;;
  esac
}

ALL="SeasonalNaive Naive AutoETS AutoTheta AutoARIMA StatEnsemble Chronos-2 Chronos-Bolt TimesFM-2.5 TiRex Toto-1.0 Moirai-2.0 Sundial TabPFN-TS"
MODELS="${*:-$ALL}"

for m in $MODELS; do
  env="$(env_of "$m")"
  if [ -z "$env" ] || [ ! -x ".venvs/$env/bin/python" ]; then
    echo "skip $m (no env; run scripts/make_env.sh ${env:-<unknown model>})"; continue
  fi
  ".venvs/$env/bin/python" -m fchub.benchmark --models "$m" --out results || echo "!! $m crashed"
done

PYBIN=".venvs/stats/bin/python"
[ -x "$PYBIN" ] || PYBIN="$(ls .venvs/*/bin/python | head -1)"
"$PYBIN" -m fchub.leaderboard results --markdown results/LEADERBOARD.md
