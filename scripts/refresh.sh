#!/usr/bin/env bash
set -euo pipefail

# Cron entrypoint: ingest -> load -> quality (clean + transform into
# fct_transactions) -> district rollup.
#
# Schema setup (sql/01_schema.sql, sql/02_fct_transactions.sql) is a
# one-time step, run manually once via psql -- not part of this recurring
# cycle. See sql/02_fct_transactions.sql for why the "transform" step is
# Python (quality.py + clean.py), not plain SQL.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SRC_DIR="$PROJECT_ROOT/src"
SQL_DIR="$PROJECT_ROOT/sql"

cd "$PROJECT_ROOT"

if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

# Prefer the project's own venv (where requirements.txt was actually
# installed) over whatever's on PATH -- a bare system interpreter won't
# have pandas/lightgbm/etc. installed.
PYTHON_BIN=""
if [ -x "$PROJECT_ROOT/.venv/bin/python" ]; then
    PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"
else
    for candidate in python3 python py; do
        if "$candidate" --version >/dev/null 2>&1; then
            PYTHON_BIN="$candidate"
            break
        fi
    done
fi
if [ -z "$PYTHON_BIN" ]; then
    echo "ERROR: no working Python interpreter found (.venv/bin/python, python3, python, py)" >&2
    exit 1
fi

# Cron failure alert. Fires on any command failure (set -e + this trap).
# Skips silently (one log line) if TELEGRAM_BOT_TOKEN/CHAT_ID aren't set --
# fine for local dev; fill in .env before relying on this in production.
send_failure_alert() {
    local exit_code=$1 failed_command=$2 line_no=$3
    if [ -z "${TELEGRAM_BOT_TOKEN:-}" ] || [ -z "${TELEGRAM_CHAT_ID:-}" ]; then
        echo "[refresh.sh] Telegram alert skipped: TELEGRAM_BOT_TOKEN/CHAT_ID not configured" >&2
        return 0
    fi
    local message="housing-price-drivers refresh.sh FAILED (exit ${exit_code}) at line ${line_no}: ${failed_command}"
    curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
        -d chat_id="${TELEGRAM_CHAT_ID}" \
        --data-urlencode text="$message" \
        >/dev/null || echo "[refresh.sh] Telegram alert request failed" >&2
}
trap 'send_failure_alert "$?" "$BASH_COMMAND" "$LINENO"' ERR

echo "=== [1/4] ingest ==="
"$PYTHON_BIN" "$SRC_DIR/ingest.py" --period current

echo "=== [2/4] load ==="
"$PYTHON_BIN" "$SRC_DIR/load.py"

echo "=== [3/4] quality (clean + transform into fct_transactions) ==="
"$PYTHON_BIN" "$SRC_DIR/quality.py"

echo "=== [4/4] district rollup ==="
PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" \
    -f "$SQL_DIR/03_dim_district.sql"

echo "=== refresh complete ==="
