#!/usr/bin/env bash
# setup_and_run.sh - setup venv, update repo, install deps and run GUI (Linux/macOS)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "[setup] Pull latest from git (current branch)"
git pull --ff-only || echo "[setup] git pull failed or no network; continuing"

# detect python executable preferring python3.12
PY=""
for candidate in python3.12 python3 python; do
  if command -v "$candidate" >/dev/null 2>&1; then
    v=$("$candidate" -c 'import sys; print(".".join(map(str, sys.version_info[:2])))' 2>/dev/null || true)
    if [[ -n "$v" ]]; then
      PY="$candidate"
      break
    fi
  fi
done

if [[ -z "$PY" ]]; then
  echo "[setup] No python interpreter found on PATH. Install Python 3.12."
  exit 2
fi

echo "[setup] Using python: $PY"

# Create venv
if [[ ! -d ".venv" ]]; then
  echo "[setup] Creating virtualenv .venv"
  "$PY" -m venv .venv
fi

# Activate venv
# shellcheck source=/dev/null
source .venv/bin/activate

python -m pip install --upgrade pip wheel setuptools
echo "[setup] Installing package and requirements"
pip install -e .

echo "[setup] Running GUI"
python scripts/run_gui.py
