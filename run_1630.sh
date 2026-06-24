#!/bin/bash
# 16:30 - 每日全市场分析与推送
cd /root/my_trades/my_trade-main || exit 1
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

{
  echo "$(date '+%F %T') run_1630.sh 开始每日全市场分析"
  bash ./run_daily_analysis.sh
  echo "$(date '+%F %T') run_1630.sh 结束"
} >> /root/my_trades/my_trade-main/daily_analysis.log 2>&1
