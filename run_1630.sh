#!/bin/bash
# 16:30 - 推送最新组合选股详版
cd /root/my_trades/my_trade-main || exit 1
export PYTHONUNBUFFERED=1

{
  echo "$(date '+%F %T') run_1630.sh 开始推送最新组合选股详版"
  python3 -u portfolioPush.py --top 30
  echo "$(date '+%F %T') run_1630.sh 结束"
} >> /root/my_trades/my_trade-main/portfolioPush.log 2>&1
