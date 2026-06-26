#!/bin/bash
# Daily full-market analysis entrypoint for cron.
# It keeps each step isolated: a failed data source should not block the
# remaining reports from being generated.

set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8
export HTTP_PROXY="${HTTP_PROXY:-http://127.0.0.1:7897}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://127.0.0.1:7897}"
export ALL_PROXY="${ALL_PROXY:-http://127.0.0.1:7897}"

LOG_DIR="${LOG_DIR:-$SCRIPT_DIR/logs}"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/daily_analysis_$(date '+%Y%m%d').log"

run_step() {
  local name="$1"
  local seconds="$2"
  shift 2

  {
    echo
    echo "========== $(date '+%F %T') START $name =========="
    echo "COMMAND: $*"
  } >> "$LOG_FILE"

  if command -v timeout >/dev/null 2>&1; then
    timeout "$seconds" "$@" >> "$LOG_FILE" 2>&1
  else
    "$@" >> "$LOG_FILE" 2>&1
  fi

  local status=$?
  {
    echo "========== $(date '+%F %T') END $name status=$status =========="
  } >> "$LOG_FILE"

  return 0
}

PYTHON_BIN="${PYTHON_BIN:-python3}"
FACTOR_WORKERS="${FACTOR_WORKERS:-1}"
FORMULA33_WORKERS="${FORMULA33_WORKERS:-1}"
FORMULA33_SLEEP="${FORMULA33_SLEEP:-0.2}"
FORMULA33_RETRIES="${FORMULA33_RETRIES:-5}"
FORMULA33_RETRY_DELAY="${FORMULA33_RETRY_DELAY:-5}"
FORMULA33_CAPITAL_WORKERS="${FORMULA33_CAPITAL_WORKERS:-1}"
SECTOR_SLEEP="${SECTOR_SLEEP:-0.3}"
SECTOR_RETRIES="${SECTOR_RETRIES:-5}"
SECTOR_RETRY_DELAY="${SECTOR_RETRY_DELAY:-5}"

run_step "factorStock daily selection" 7200 \
  "$PYTHON_BIN" -u factorStock.py \
  --top 30 \
  --core-min-score 80 \
  --low-min-score 75 \
  --quality-min-score 80 \
  --value-min-mktcap 100 \
  --workers "$FACTOR_WORKERS" \
  --value-watch-ratio 1.08 \
  --value-watch-top 20 \
  --allow-login-fail

run_step "formula33 market structure" 7200 \
  "$PYTHON_BIN" -u formula33Stats.py \
  --lookback 21 \
  --history-days 90 \
  --workers "$FORMULA33_WORKERS" \
  --sleep "$FORMULA33_SLEEP" \
  --retries "$FORMULA33_RETRIES" \
  --retry-delay "$FORMULA33_RETRY_DELAY" \
  --capital-workers "$FORMULA33_CAPITAL_WORKERS" \
  --require-end-trade \
  --price-source akshare \
  --metadata-source akshare \
  --missing-mktcap-policy pass \
  --market-cap-source none

run_step "sector horizontal statistics" 3600 \
  "$PYTHON_BIN" -u sectorStats.py \
  --lookback 10 \
  --history-days 90 \
  --top-amount 50 \
  --sleep "$SECTOR_SLEEP" \
  --retries "$SECTOR_RETRIES" \
  --retry-delay "$SECTOR_RETRY_DELAY" \
  --fallback-sample

run_step "sector mainline watch" 3600 \
  "$PYTHON_BIN" -u sectorWatch.py \
  --top 30 \
  --days 80 \
  --limit-up-days 5 \
  --sleep "$SECTOR_SLEEP" \
  --retries "$SECTOR_RETRIES" \
  --retry-delay "$SECTOR_RETRY_DELAY" \
  --fallback-sample

{
  echo
  echo "$(date '+%F %T') daily analysis finished"
  echo "Outputs:"
  find "$SCRIPT_DIR/选股结果" "$SCRIPT_DIR/板块观察" -maxdepth 1 -type f -mtime -2 2>/dev/null | sort
} >> "$LOG_FILE"
