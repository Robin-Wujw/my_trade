#!/usr/bin/env bash
# A1: factor_horizon 通道验证 —— 重建 hybrid value_only 候选 + 回测对比
# 用法: bash scripts/run_fh_backtest.sh [--with-fh] [--with-pepb]
set -euo pipefail

PY="D:/ActionsRunner/my-trade/python/python.exe"
FH_RIGHT="var/backtests/quantsplaybook_factor_only_2021_to_20260721/candidates/playbook_low_corr_fh"
BASE_RIGHT="var/backtests/quantsplaybook_factor_only_2021_to_20260721/candidates/playbook_low_corr"
LEFT="var/backtests/tushare_official_candidates_2021_to_20260721_merged"
FORMULA="var/backtests/formula33_tushare_2021_to_20260721.csv"

WITH_FH=0
WITH_PEPB=0
for arg in "$@"; do
  case "$arg" in
    --with-fh) WITH_FH=1 ;;
    --with-pepb) WITH_PEPB=1 ;;
  esac
done

RIGHT="$BASE_RIGHT"
if [ "$WITH_FH" = "1" ]; then
  RIGHT="$FH_RIGHT"
  OUT_TAG="fh"
else
  OUT_TAG="nofh"
fi

EXTRA_ARG=""
if [ "$WITH_PEPB" = "1" ]; then
  EXTRA_ARG=""
  OUT_TAG="${OUT_TAG}_pepb"
else
  EXTRA_ARG="--left-extra-directory \"\""
  OUT_TAG="${OUT_TAG}_vlonly"
fi

HYBRID_ROOT="var/backtests/quantsplaybook_hybrid_${OUT_TAG}_2021_to_20260721"
HYBRID_CAND="${HYBRID_ROOT}/candidates/hybrid_low_corr_value"
BACKTEST_ROOT="var/backtests/hybrid_${OUT_TAG}_5y"

echo "== 重建 hybrid 候选 (right=$RIGHT) =="
PYTHONIOENCODING=utf-8 "$PY" -m apps.quantsplaybook_hybrid \
  --right-directory "$RIGHT" \
  --left-directory "$LEFT" \
  $EXTRA_ARG \
  --output-root "$HYBRID_ROOT" \
  --start-date 2021-01-01 --end-date 2026-07-21

FH_FLAG=""
if [ "$WITH_FH" = "1" ]; then
  FH_FLAG="--enable-factor-horizon-channel"
fi

echo "== 回测 ($OUT_TAG, FH=$WITH_FH) =="
PYTHONIOENCODING=utf-8 "$PY" -m apps.portfolio_backtest \
  --start-date 2021-01-01 --end-date 2026-07-21 \
  --candidate-directory "$HYBRID_CAND" \
  --formula-history "$FORMULA" \
  --no-refresh-inputs --allow-frozen-inputs \
  --price-database-source auto \
  --output-directory "$BACKTEST_ROOT" \
  $FH_FLAG

echo "== 完成: $BACKTEST_ROOT =="
ls "$BACKTEST_ROOT"/*_summary.json | head -1
