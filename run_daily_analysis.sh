#!/bin/bash
# Daily full-market analysis entrypoint for cron.
# It keeps each step isolated: a failed data source should not block the
# remaining reports from being generated.

set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8
if [ "${DISABLE_DEFAULT_PROXY:-0}" != "1" ]; then
  export HTTP_PROXY="${HTTP_PROXY:-http://127.0.0.1:7897}"
  export HTTPS_PROXY="${HTTPS_PROXY:-http://127.0.0.1:7897}"
  export ALL_PROXY="${ALL_PROXY:-http://127.0.0.1:7897}"
fi

LOG_DIR="${LOG_DIR:-$SCRIPT_DIR/logs}"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/daily_analysis_$(date '+%Y%m%d').log"
FAILED_STEPS=()
SKIPPED_STEPS=()
declare -A STEP_STATUS=()

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
  STEP_STATUS["$name"]="$status"
  {
    echo "========== $(date '+%F %T') END $name status=$status =========="
  } >> "$LOG_FILE"

  if [ "$status" -ne 0 ]; then
    FAILED_STEPS+=("$name:$status")
  fi

  return 0
}

step_succeeded() {
  [ "${STEP_STATUS[$1]:-999}" -eq 0 ]
}

skip_step() {
  local name="$1"
  local reason="$2"
  SKIPPED_STEPS+=("$name")
  {
    echo
    echo "========== $(date '+%F %T') SKIP $name =========="
    echo "REASON: $reason"
  } >> "$LOG_FILE"
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
FINANCIAL_UPDATES="${FINANCIAL_UPDATES:-100}"

run_step "formula33 market structure" 7200 \
  "$PYTHON_BIN" -u formula33Stats.py \
  --lookback 21 \
  --history-days 420 \
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
  --retry-delay "$SECTOR_RETRY_DELAY"

run_step "sector mainline watch" 3600 \
  "$PYTHON_BIN" -u sectorWatch.py \
  --top 30 \
  --workers 4 \
  --days 80 \
  --limit-up-days 5 \
  --sleep "$SECTOR_SLEEP" \
  --retries "$SECTOR_RETRIES" \
  --retry-delay "$SECTOR_RETRY_DELAY"

run_step "factorStock daily selection" 7200 \
  "$PYTHON_BIN" -u factorStock.py \
  --top 200 \
  --core-min-score 80 \
  --low-min-score 75 \
  --quality-min-score 80 \
  --value-min-mktcap 100 \
  --workers "$FACTOR_WORKERS" \
  --value-watch-ratio 1.08 \
  --value-watch-top 20 \
  --akshare-cache-only \
  --allow-login-fail

run_step "full market fundamental cache and snapshot" 7200 \
  "$PYTHON_BIN" -u fullMarketFundamentalUpdate.py \
  --max-updates "$FINANCIAL_UPDATES" \
  --workers 2 \
  --min-price-coverage 0.90 \
  --min-financial-coverage 0.35 \
  --target-financial-coverage 0.95 \
  --alert

if step_succeeded "full market fundamental cache and snapshot"; then
  run_step "daily fundamental sections" 600 \
    "$PYTHON_BIN" -u dailyFundamentalSelect.py \
    --value-ratio 1.08 \
    --normal-top 30
else
  skip_step "daily fundamental sections" "full market fundamental cache and snapshot failed"
fi

REPORT_ARGS=(dailyReportPush.py --top 10 --selection-top 30 --max-chars 12000)
if [ "${NO_PUSH:-0}" = "1" ]; then
  REPORT_ARGS+=(--no-push)
fi
if step_succeeded "formula33 market structure" \
  && step_succeeded "sector mainline watch" \
  && step_succeeded "daily fundamental sections"; then
  run_step "daily consolidated PushPlus report" 300 "$PYTHON_BIN" -u "${REPORT_ARGS[@]}"
else
  skip_step "daily consolidated PushPlus report" "one or more required report inputs failed or were skipped"
fi

{
  echo
  echo "$(date '+%F %T') daily analysis finished"
  echo "Outputs:"
  find "$SCRIPT_DIR/选股结果" "$SCRIPT_DIR/板块观察" -maxdepth 1 -type f -mtime -2 2>/dev/null | sort
} >> "$LOG_FILE"

if [ "${#SKIPPED_STEPS[@]}" -gt 0 ]; then
  echo "SKIPPED STEPS: ${SKIPPED_STEPS[*]}" >> "$LOG_FILE"
fi

if [ "${#FAILED_STEPS[@]}" -gt 0 ]; then
  FAILURE_SUMMARY="FAILED STEPS: ${FAILED_STEPS[*]}"
  echo "$FAILURE_SUMMARY" | tee -a "$LOG_FILE" >&2
  "$PYTHON_BIN" -u pipelineAlert.py --title "Daily selection pipeline failed" --message "$FAILURE_SUMMARY" >> "$LOG_FILE" 2>&1 || true
  exit 1
fi
