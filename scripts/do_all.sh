#!/usr/bin/env bash
set -euo pipefail

# Usage:
# ./scripts/do_all.sh --db-url postgresql://fx:fxpass@127.0.0.1:5432/magicforex --benchmark-rows 50000 --benchmark-batch 1000

DB_URL="${1:-}"
if [ -z "$DB_URL" ] ; then
  echo "Usage: $0 <db-url> [--benchmark-rows N] [--benchmark-batch B]"
  echo "Example: $0 postgresql://fx:fxpass@127.0.0.1:5432/magicforex --benchmark-rows 50000 --benchmark-batch 1000"
  exit 1
fi

# parse optional args
BENCH_ROWS=50000
BENCH_BATCH=1000
shift || true
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --benchmark-rows)
      BENCH_ROWS="$2"; shift 2;;
    --benchmark-batch)
      BENCH_BATCH="$2"; shift 2;;
    *)
      shift;;
  esac
done

export DATABASE_URL="$DB_URL"
echo "[do_all] DATABASE_URL=$DATABASE_URL"
echo "[do_all] Apply alembic migrations..."
alembic upgrade head

# If postgres, ensure jsonb/GIN index exists (best-effort)
echo "[do_all] Ensure JSONB/GIN index if Postgres..."
python - <<PY
import os
from sqlalchemy import create_engine
engine = create_engine(os.environ['DATABASE_URL'], future=True)
dial = engine.dialect.name.lower()
if dial == 'postgresql':
    with engine.connect() as conn:
        try:
            conn.execute("ALTER TABLE features ALTER COLUMN features_json TYPE jsonb USING features_json::jsonb;")
        except Exception:
            pass
        # create GIN index if not exists
        try:
            conn.execute("CREATE INDEX IF NOT EXISTS ix_features_jsonb_gin ON features USING GIN ((features_json::jsonb) jsonb_path_ops);")
        except Exception:
            try:
                conn.execute("CREATE INDEX IF NOT EXISTS ix_features_jsonb_gin ON features USING GIN ((features_json::jsonb));")
            except Exception as e:
                print('Could not create GIN index:', e)
print('Postgres index step completed (if applicable).')
PY

# Run benchmark
echo "[do_all] Running benchmark_bulk.py rows=${BENCH_ROWS} batch=${BENCH_BATCH}..."
python scripts/benchmark_bulk.py --db-url "$DATABASE_URL" --rows ${BENCH_ROWS} --batch ${BENCH_BATCH}

# Start service
echo "[do_all] Starting uvicorn service (foreground)..."
uvicorn src.forex_diffusion.inference.service:app --host 0.0.0.0 --port 8000
