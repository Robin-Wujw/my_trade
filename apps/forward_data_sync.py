"""补齐研究数据并捕获真实版本；不生成历史前向证据，不发送消息。"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from stock_research.core.paths import PATHS
from stock_research.core.console import configure_utf8_console
from stock_research.data.forward_sync import run_sync


def main(argv=None) -> int:
    configure_utf8_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", default=str(PATHS.database))
    parser.add_argument("--state-file", default=str(PATHS.state / "forward_data_sync.json"))
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--max-requests", type=int, default=0)
    parser.add_argument("--skip-financials", action="store_true")
    args = parser.parse_args(argv)
    now = pd.Timestamp.now(tz="Asia/Shanghai")
    # The calendar determines the actual final session. Before daily-data
    # publication time the default never requests today's unfinished bars.
    end = now.normalize() - pd.Timedelta(days=int((now.hour, now.minute) < (16, 30)))
    result = run_sync(
        database_path=args.database, state_path=args.state_file,
        start_date=args.start_date, end_date=args.end_date or end.strftime("%Y-%m-%d"),
        max_requests=args.max_requests, capture_financials=not args.skip_financials,
    )
    print(json.dumps({key: value for key, value in result.items() if key != "completed"},
                     ensure_ascii=False, indent=2))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
