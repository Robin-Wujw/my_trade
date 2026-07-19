"""Search visible-data candidate models while keeping portfolio trading fixed."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from apps.portfolio_backtest import (
    infer_price_frame_source,
    load_candidate_snapshots,
    load_price_frames,
    refresh_price_cache_directory,
    validate_backtest_input_coverage,
    validate_price_frame_coverage,
)
from stock_research.core.paths import PATHS
from stock_research.market.miniqmt_data import load_miniqmt_price_frames
from stock_research.reporting.trade_reminders import load_trade_plans
from stock_research.strategies.candidate_interface import normalize_candidate_snapshots
from stock_research.strategies.portfolio_backtest import run_portfolio_backtest


def _num(row, key, default=0.0):
    try:
        value = float(row.get(key))
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _enabled_source(row):
    return {item for item in str(row.get("candidate_source") or "").split("+") if item}


def _hard_gate(row):
    return (
        _num(row, "quality_score") >= 70.0
        and _num(row, "earnings_yoy") >= 0.10
        and _num(row, "mktcap") >= 100.0
        and str(row.get("data_status") or "traded") == "traded"
        and str(row.get("valid_price_bar", "true")).lower() not in {"false", "0"}
        and str(row.get("is_traded_bar", "true")).lower() not in {"false", "0"}
    )


def _candidate_model_score(row, config):
    growth = min(max(_num(row, "earnings_yoy"), 0.0), 2.0) / 2.0 * 100.0
    quality = _num(row, "quality_score")
    trade_basis = min(max(_num(row, "trade_basis_score"), 0.0), 12.0) / 12.0 * 100.0
    leadership = min(max(_num(row, "leadership_score"), 0.0), 30.0) / 30.0 * 100.0
    right_quant = min(max(_num(row, "right_quant_score"), 0.0), 120.0) / 120.0 * 100.0
    momentum = (
        _num(row, "quant_momentum_rank", 50.0) * 0.45
        + _num(row, "quant_alpha_rank", 50.0) * 0.25
        + _num(row, "quant_trend_efficiency_rank", 50.0) * 0.30
    )
    payoff = (
        _num(row, "quant_payoff_rank", 50.0) * 0.45
        + _num(row, "quant_structure_rank", 50.0) * 0.30
        + _num(row, "quant_volume_confirm_rank", 50.0) * 0.25
    )
    risk = (
        _num(row, "quant_low_risk_rank", 50.0) * 0.45
        + _num(row, "quant_overheat_control_rank", 50.0) * 0.35
        + _num(row, "quant_divergence_rank", 50.0) * 0.20
    )
    liquidity = _num(row, "quant_liquidity_rank", 50.0)
    mainline = 100.0 if str(row.get("mainline_boards") or "").strip() else 0.0
    sources = _enabled_source(row)
    source_bonus = 0.0
    if "value_model" in sources:
        source_bonus -= float(config["value_penalty"])
    if "standard_mainline" in sources:
        source_bonus += float(config["mainline_source_bonus"])
    return (
        quality * float(config["quality"])
        + growth * float(config["growth"])
        + trade_basis * float(config["trade_basis"])
        + leadership * float(config["leadership"])
        + right_quant * float(config["right_quant"])
        + momentum * float(config["momentum"])
        + payoff * float(config["payoff"])
        + risk * float(config["risk"])
        + liquidity * float(config["liquidity"])
        + mainline * float(config["mainline"])
        + source_bonus
    )


def _strong_enough(row, config):
    if not _hard_gate(row):
        return False
    if _num(row, "avg_amount_20") < float(config["min_avg_amount"]):
        return False
    if _num(row, "drawdown_60", -1.0) < float(config["min_drawdown_60"]):
        return False
    if _num(row, "return_20d", -1.0) < float(config["min_return_20"]):
        return False
    if _num(row, "return_60d", -1.0) < float(config["min_return_60"]):
        return False
    if _num(row, "quant_momentum_rank", 0.0) < float(config["min_momentum_rank"]):
        return False
    if _num(row, "quant_payoff_rank", 0.0) < float(config["min_payoff_rank"]):
        return False
    return True


def transform_snapshots(raw_snapshots, config):
    result = {}
    top_n = int(config["top_n"])
    promote_n = int(config["promote_n"])
    for date, rows in raw_snapshots.items():
        scored = []
        for row in rows:
            item = dict(row)
            score = _candidate_model_score(item, config)
            item["candidate_model_score"] = round(score, 6)
            item["selection_reason"] = (
                f"{item.get('selection_reason') or ''}; "
                f"candidate_model={config['name']} score={score:.1f}"
            ).strip("; ")
            if _strong_enough(item, config):
                scored.append(item)
        scored.sort(key=lambda item: (-float(item["candidate_model_score"]), str(item.get("code"))))
        keep_codes = {str(item.get("code")) for item in scored[:top_n]}
        promote_codes = {str(item.get("code")) for item in scored[:promote_n]}
        transformed = []
        for row in rows:
            item = dict(row)
            code = str(item.get("code"))
            if code in keep_codes:
                score = _candidate_model_score(item, config)
                item["candidate_score"] = round(score, 6)
                sources = _enabled_source(item)
                if code in promote_codes:
                    sources.add("growth_leadership")
                item["candidate_source"] = "+".join(sorted(source for source in sources if source))
                item["signal_eligible"] = _hard_gate(item)
                item["selected_for_trading"] = _hard_gate(item)
                item["candidate_failure_reason"] = "" if _hard_gate(item) else "candidate_model_hard_gate_failed"
            else:
                item["signal_eligible"] = False
                item["selected_for_trading"] = False
                old = str(item.get("candidate_failure_reason") or "").strip()
                item["candidate_failure_reason"] = (
                    f"{old}; candidate_model_filtered_out".strip("; ")
                )
            transformed.append(item)
        result[date] = normalize_candidate_snapshots({date: transformed}, include_diagnostics=True)[date]
    return result


def configs():
    base_weights = [
        {
            "label": "attack",
            "quality": 0.10,
            "growth": 0.08,
            "trade_basis": 0.08,
            "leadership": 0.16,
            "right_quant": 0.12,
            "momentum": 0.20,
            "payoff": 0.16,
            "risk": 0.04,
            "liquidity": 0.02,
            "mainline": 0.04,
        },
        {
            "label": "payoff",
            "quality": 0.10,
            "growth": 0.10,
            "trade_basis": 0.10,
            "leadership": 0.12,
            "right_quant": 0.10,
            "momentum": 0.14,
            "payoff": 0.24,
            "risk": 0.06,
            "liquidity": 0.02,
            "mainline": 0.02,
        },
        {
            "label": "mainline_attack",
            "quality": 0.08,
            "growth": 0.08,
            "trade_basis": 0.08,
            "leadership": 0.14,
            "right_quant": 0.10,
            "momentum": 0.18,
            "payoff": 0.16,
            "risk": 0.04,
            "liquidity": 0.02,
            "mainline": 0.12,
        },
    ]
    index = 0
    for weights in base_weights:
        for min_avg_amount in [300_000_000.0, 500_000_000.0, 1_000_000_000.0]:
            for min_drawdown in [-0.40, -0.32, -0.22]:
                for min_ret20, min_ret60 in [(-0.02, 0.05), (0.00, 0.10), (0.04, 0.12)]:
                    for top_n, promote_n in [(10, 5), (14, 7), (20, 10)]:
                        index += 1
                        yield {
                            "name": f"{weights['label']}_{index}",
                            **{k: v for k, v in weights.items() if k != "label"},
                            "min_avg_amount": min_avg_amount,
                            "min_drawdown_60": min_drawdown,
                            "min_return_20": min_ret20,
                            "min_return_60": min_ret60,
                            "min_momentum_rank": 45.0,
                            "min_payoff_rank": 40.0,
                            "top_n": top_n,
                            "promote_n": promote_n,
                            "value_penalty": 20.0,
                            "mainline_source_bonus": 5.0,
                        }


def compact(name, result):
    return {
        "name": name,
        "final_return_pct": result.get("final_return_pct"),
        "realized_return_pct": result.get("realized_return_pct"),
        "unrealized_return_pct": result.get("unrealized_return_pct"),
        "maximum_drawdown_pct": result.get("maximum_drawdown_pct"),
        "transaction_cost_pct": result.get("transaction_cost_pct"),
        "buy_count": result.get("trade_summary", {}).get("buy_count"),
        "sell_count": result.get("trade_summary", {}).get("sell_count"),
        "sell_win_rate_pct": result.get("trade_summary", {}).get("sell_win_rate_pct"),
        "entry_block_count": result.get("entry_block_count"),
        "final_positions": [
            {
                "code": item.get("code"),
                "name": item.get("name"),
                "position_pct": item.get("position_pct"),
                "unrealized_pnl_pct": item.get("unrealized_pnl_pct"),
            }
            for item in result.get("final_positions", [])
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", default="2026-01-05")
    parser.add_argument("--end-date", default="2026-07-14")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output-directory", default=str(PATHS.runtime_root / "backtests" / "candidate_model_search"))
    parser.add_argument("--candidate-directory", default=str(PATHS.runtime_root / "backtests" / "candidate_snapshots" / "unified-selection-v4"))
    parser.add_argument("--formula-history", default=str(PATHS.runtime_root / "backtests" / "formula33_phase_research_v1.csv"))
    parser.add_argument("--price-source", choices=("miniqmt", "akshare-cache"), default="miniqmt")
    parser.add_argument("--price-kline-directory", default="")
    parser.add_argument("--miniqmt-dividend-type", default="front")
    parser.add_argument("--miniqmt-refresh", action="store_true")
    parser.add_argument("--no-price-database", action="store_true")
    parser.add_argument("--allow-unsafe-financial", action="store_true")
    args = parser.parse_args(argv)
    if args.price_source == "akshare-cache" and not args.price_kline_directory:
        args.price_kline_directory = str(refresh_price_cache_directory("miniqmt"))

    formula = pd.read_csv(args.formula_history)
    raw = load_candidate_snapshots(args.candidate_directory, args.start_date, args.end_date)
    snapshots = normalize_candidate_snapshots(raw, include_diagnostics=True)
    validate_backtest_input_coverage(
        snapshots,
        formula,
        args.start_date,
        args.end_date,
        candidate_directory=args.candidate_directory,
        allow_unsafe_financial=args.allow_unsafe_financial,
    )
    codes = {str(row["code"]) for rows in snapshots.values() for row in rows}
    price_start_date = (pd.Timestamp(args.start_date) - pd.Timedelta(days=700)).strftime("%Y-%m-%d")
    if args.price_source == "miniqmt":
        price_frames, price_source_summary = load_miniqmt_price_frames(
            codes,
            start_date=price_start_date,
            end_date=args.end_date,
            dividend_type=args.miniqmt_dividend_type,
            refresh=args.miniqmt_refresh,
            persist=False,
        )
        if price_source_summary["missing_count"]:
            raise RuntimeError(
                "MiniQMT price cache is incomplete; rerun with --miniqmt-refresh. "
                f"missing={price_source_summary['missing_sample']}"
            )
    else:
        price_frames = load_price_frames(
            codes,
            args.price_kline_directory,
            start_date=price_start_date,
            end_date=args.end_date,
            source=infer_price_frame_source(args.price_kline_directory),
            prefer_database=not args.no_price_database,
        )
    validate_price_frame_coverage(price_frames, codes, args.start_date, args.end_date)
    phases = {
        str(row["date"]): {
            "phase": str(row["phase"]),
            "window_down_streak": int(row.get("window_down_streak") or 0),
            "window_up_streak": int(row.get("window_up_streak") or 0),
        }
        for _, row in formula.iterrows()
    }
    trade_plans = load_trade_plans(PATHS.project_root / "config" / "trade_plans.json")
    output = Path(args.output_directory)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    config_list = list(configs())
    if args.limit:
        config_list = config_list[: max(1, int(args.limit))]
    for index, config in enumerate(config_list, start=1):
        transformed = transform_snapshots(snapshots, config)
        result = run_portfolio_backtest(
            price_frames,
            transformed,
            phases,
            requested_start=args.start_date,
            end_date=args.end_date,
            trade_plans=trade_plans,
            max_positions=3,
            max_total_held_symbols=5,
            max_same_industry=2,
            same_theme_correlation=0.60,
            min_entry_evidence_score=0.0,
            profit_tranches=2,
            profit_tail_min_return=0.50,
            left_grid_unit=0.0,
            left_grid_max_exposure=0.0,
            max_symbol_exposure=1.0,
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
        row = {**compact(config["name"], result), **config}
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if index % 25 == 0:
            pd.DataFrame(rows).sort_values(
                ["final_return_pct", "maximum_drawdown_pct"],
                ascending=[False, False],
            ).to_csv(output / "candidate_model_search_partial.csv", index=False, encoding="utf-8-sig")
    rows.sort(key=lambda item: (item["final_return_pct"], item["maximum_drawdown_pct"]), reverse=True)
    pd.DataFrame(rows).to_csv(output / "candidate_model_search.csv", index=False, encoding="utf-8-sig")
    (output / "candidate_model_search.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps(rows[:10], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
