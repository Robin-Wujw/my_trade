"""Replay explicit author-case symbols without writing to DuckDB."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stock_research.core.paths import PATHS
from stock_research.strategies.portfolio_backtest import run_portfolio_backtest


DEFAULT_CASES = {
    "junsheng": {
        "start": "2026-01-01",
        "end": "2026-04-30",
        "codes": {"sh.600699": "均胜电子"},
    },
    "duofuduo_tinci": {
        "start": "2026-05-01",
        "end": "2026-07-14",
        "codes": {"sz.002407": "多氟多", "sz.002709": "天赐材料"},
    },
}


def _cache_name(code: str) -> str:
    market, symbol = code.split(".", 1)
    return f"{market}_{symbol}.csv"


def load_price_frames(codes: list[str], start_date: str, end_date: str) -> dict[str, pd.DataFrame]:
    start = pd.Timestamp(start_date) - pd.Timedelta(days=700)
    end = pd.Timestamp(end_date)
    frames = {}
    for code in codes:
        path = PATHS.cache / "formula33_kline" / "akshare" / _cache_name(code)
        frame = pd.read_csv(path)
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame = frame[(frame["date"] >= start) & (frame["date"] <= end)]
        frames[code] = frame.reset_index(drop=True)
    return frames


def load_phases(start_date: str, end_date: str) -> dict[str, dict]:
    formula = pd.read_csv(PATHS.runtime_root / "backtests" / "formula33_phase_research_v1.csv")
    formula["date"] = pd.to_datetime(formula["date"], errors="coerce")
    formula = formula[
        (formula["date"] >= pd.Timestamp(start_date))
        & (formula["date"] <= pd.Timestamp(end_date))
    ]
    return {
        row["date"].strftime("%Y-%m-%d"): {
            "phase": str(row["phase"]),
            "window_down_streak": int(row.get("window_down_streak") or 0),
            "window_up_streak": int(row.get("window_up_streak") or 0),
        }
        for _, row in formula.iterrows()
    }


def explicit_snapshots(codes: dict[str, str], start_date: str, end_date: str) -> dict[str, list[dict]]:
    dates = sorted(load_phases(start_date, end_date))
    rows = [
        {
            "code": code,
            "name": name,
            "candidate_source": "author_case_validation",
            "selection_overridden": True,
            "strategy_part": "author case validation",
            "selected_for_trading": True,
            "signal_eligible": True,
            "allow_left": False,
            "allow_right": True,
            "candidate_score": 100.0,
            "quality_score": 90.0,
            "earnings_yoy": 0.30,
            "mktcap": 500.0,
            "trade_basis_score": 8.0,
            "leadership_score": 20.0,
            "selection_reason": "作者案例买卖引擎校准池；不参与正式选股收益声明",
        }
        for code, name in codes.items()
    ]
    return {date: [dict(row) for row in rows] for date in dates}


def run_case(case_name: str, case: dict, output_root: Path) -> dict:
    start = case["start"]
    end = case["end"]
    codes = dict(case["codes"])
    result = run_portfolio_backtest(
        load_price_frames(list(codes), start, end),
        explicit_snapshots(codes, start, end),
        load_phases(start, end),
        requested_start=start,
        end_date=end,
        max_positions=3,
        max_total_held_symbols=5,
        max_same_industry=2,
        same_theme_correlation=0.60,
        min_entry_evidence_score=7.0,
        profit_tranches=5,
        profit_tail_min_return=0.50,
        signals_effective_next_day=True,
        auto_price_structure=True,
        allow_structure_pullback=True,
        allow_pullback_pilot=False,
        close_confirmed_execution="close_proxy",
        commission_rate=0.000085,
        minimum_commission=5.0,
        initial_capital=1_000_000.0,
        sell_stamp_duty_rate=0.0005,
        estimated_slippage_rate=0.0005,
    )
    output = output_root / case_name
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(result["events"]).to_csv(
        output / f"{case_name}_events.csv", index=False, encoding="utf-8-sig",
    )
    pd.DataFrame(result["trade_ledger"]).to_csv(
        output / f"{case_name}_trades.csv", index=False, encoding="utf-8-sig",
    )
    summary = {key: value for key, value in result.items() if key not in {"events", "trade_ledger", "equity_curve"}}
    (output / f"{case_name}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return summary


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=sorted(DEFAULT_CASES), action="append")
    parser.add_argument(
        "--output-directory",
        default=str(PATHS.runtime_root / "backtests" / "author_case_replay"),
    )
    args = parser.parse_args(argv)
    wanted = args.case or sorted(DEFAULT_CASES)
    output_root = Path(args.output_directory)
    summaries = {
        name: run_case(name, DEFAULT_CASES[name], output_root)
        for name in wanted
    }
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
