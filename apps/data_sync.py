"""Synchronize provider datasets into the unified SQLite cache."""
from __future__ import annotations

import argparse

from stock_research.api import tushare as tushare_api
from stock_research.core.paths import PATHS
from stock_research.storage import Database, TushareRepository


TUSHARE_DEFAULT_FIELDS = {
    "stock_basic": "ts_code,symbol,name,area,industry,list_date,list_status,exchange",
    "daily_basic": "ts_code,trade_date,close,turnover_rate,volume_ratio,pe,pb,total_mv,circ_mv",
    "income": "ts_code,ann_date,f_ann_date,end_date,report_type,basic_eps,diluted_eps,total_revenue,n_income_attr_p",
    "balancesheet": "ts_code,ann_date,f_ann_date,end_date,report_type,total_assets,total_liab,total_hldr_eqy_exc_min_int",
    "cashflow": "ts_code,ann_date,f_ann_date,end_date,report_type,n_cashflow_act,n_cashflow_inv_act,n_cash_flows_fnc_act",
    "fina_indicator": "ts_code,ann_date,end_date,eps,dt_eps,bps,roe,roe_dt,netprofit_yoy,or_yoy",
    "share_float": "ts_code,ann_date,float_date,float_share,float_ratio,holder_name,share_type",
    "dividend": "ts_code,ann_date,end_date,div_proc,stk_div,stk_bo_rate,cash_div,cash_div_tax",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=sorted(TUSHARE_DEFAULT_FIELDS))
    parser.add_argument("--fields", default="", help="Override Tushare fields.")
    parser.add_argument("--all-fields", action="store_true", help="Do not send an explicit fields list.")
    parser.add_argument("--ts-code", default="")
    parser.add_argument("--trade-date", default="")
    parser.add_argument("--start-date", default="")
    parser.add_argument("--end-date", default="")
    parser.add_argument("--ann-date", default="")
    parser.add_argument("--period", default="")
    parser.add_argument("--limit", type=int, default=0, help="Local row limit after fetch, for smoke tests.")
    parser.add_argument("--database", default=str(PATHS.database))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    params = {
        key: value
        for key, value in {
            "ts_code": args.ts_code,
            "trade_date": args.trade_date,
            "start_date": args.start_date,
            "end_date": args.end_date,
            "ann_date": args.ann_date,
            "period": args.period,
        }.items()
        if value
    }
    database = Database(args.database, code_version="tushare-unified-cache-v1")
    database.initialize()
    frame = tushare_api.query(
        args.dataset,
        fields="" if args.all_fields else (args.fields or TUSHARE_DEFAULT_FIELDS[args.dataset]),
        **params,
    )
    if args.limit > 0:
        frame = frame.head(args.limit)
    rows = TushareRepository(database).upsert_dataset(
        args.dataset,
        frame,
        source="tushare/pro",
        params=params,
    )
    print(f"synced dataset={args.dataset} rows={rows} database={database.path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
