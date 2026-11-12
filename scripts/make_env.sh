#!/usr/bin/env bash
# Create an isolated virtualenv for one model (or the shared "stats" env).
#
#   scripts/make_env.sh stats        # Naive, SeasonalNaive, AutoETS, AutoARIMA, AutoTheta, StatEnsemble
#   scripts/make_env.sh chronos2     # any folder name under src/fchub/models/
#   scripts/make_env.sh all          # every model (several GB of downloads)
#
# Why one env per model: the official packages pin incompatible versions
# (uni2ts: torch<2.5 + numpy 1.26; toto-ts: torch==2.7.0; Sundial: transformers 4.40).
# Envs live in .venvs/<name>. Requires uv (https://docs.astral.sh/uv/).
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON_VERSION:-3.11}"   # 3.11 is the newest Python every model's pins support
MODELS_DIR=src/fchub/models

make_one() {
  local name="$1" req
  if [[ "$name" == "stats" ]]; then
    req="$MODELS_DIR/autoets/requirements.txt"
  else
    req="$MODELS_DIR/$name/requirements.txt"
  fi
  [[ -f "$req" ]] || { echo "unknown model env: $name" >&2; exit 1; }
  if [[ "$name" == "cosmic" ]]; then echo "skip cosmic: no public package"; return; fi

  echo "==> .venvs/$name (python $PY) from $req"
  uv venv --quiet --allow-existing --python "$PY" ".venvs/$name"
  VIRTUAL_ENV=".venvs/$name" uv pip install --quiet -e ".[dev]" -r "$req"
  echo "    done. Activate with: source .venvs/$name/bin/activate"
}

if [[ "${1:-}" == "all" ]]; then
  make_one stats
  for d in chronos2 chronos_bolt timesfm25 tirex toto moirai2 sundial tabpfn_ts; do make_one "$d"; done
else
  make_one "${1:?usage: scripts/make_env.sh <model-folder|stats|all>}"
fi
