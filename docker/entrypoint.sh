#!/usr/bin/env bash
set -euo pipefail

# Simple entrypoint: run migrations then start uvicorn.
# For development, mount code and start with --reload externally.

echo "[entrypoint] Starting startup tasks..."

# Attempt to run alembic upgrade head (best-effort)
if command -v alembic >/dev/null 2>&1; then
  echo "[entrypoint] Running alembic upgrade head..."
  alembic upgrade head || echo "[entrypoint] alembic upgrade head failed (continuing)..."
else
  echo "[entrypoint] alembic not found, skipping migrations."
fi

# Default host/port
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
WORKERS="${WORKERS:-1}"

echo "[entrypoint] Starting uvicorn on ${HOST}:${PORT} (workers=${WORKERS})..."
exec uvicorn src.forex_diffusion.inference.service:app --host "${HOST}" --port "${PORT}" --workers "${WORKERS}"
