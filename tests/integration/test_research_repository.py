import pandas as pd

from stock_research.storage import Database, ResearchRepository


def test_research_repository_persists_candidates_formula_and_backtest(tmp_path):
    database = Database(tmp_path / "research.duckdb", code_version="test")
    database.initialize()
    repository = ResearchRepository(database)

    assert repository.persist_fundamentals({
        "2025-09-30": {"600000": {
            "quality_score": 80, "yoy": 0.2, "mktcap": 1000, "value_line": 12,
        }}
    }) == 1
    assert repository.persist_candidate_snapshots({
        "2026-01-05": [{
            "code": "sh.600000", "name": "浦发银行", "selection_rank": 1,
            "candidate_score": 99, "report_period": "2025-09-30",
            "signal_eligible": True,
        }]
    }, version="test-v1") == 1
    assert repository.persist_formula_history(pd.DataFrame([{
        "date": "2026-01-05", "phase": "active", "window_up_streak": 5,
        "window_down_streak": 0,
    }]), version="test-v1") == 1

    run_id = repository.persist_backtest_result({
        "requested_start": "2026-01-01", "actual_start": "2026-01-05",
        "end_date": "2026-01-06", "initial_capital": 1_000_000,
        "final_return_pct": 1.5, "maximum_drawdown_pct": -2,
        "final_cash": 900_000,
        "trade_ledger": [{
            "date": "2026-01-05", "code": "sh.600000", "name": "浦发银行",
            "trade_side": "买入", "quantity": 100, "execution_price": 10,
            "trade_amount": 1000, "transaction_cost_amount": 5,
            "profit_loss_amount": -5, "reason": "test",
        }],
        "final_positions": [{
            "code": "sh.600000", "name": "浦发银行", "quantity": 100,
            "cost": 10, "close": 11, "market_value": 1100,
            "unrealized_pnl_amount": 95,
        }],
    }, run_id="test-run")

    assert run_id == "test-run"
    connection = database.connect(read_only=True)
    try:
        counts = {
            table: connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
            for table in [
                "raw.fundamental_metrics", "derived.candidate_snapshots",
                "derived.formula33_phase", "derived.backtest_runs",
                "derived.backtest_trades", "derived.backtest_positions",
            ]
        }
    finally:
        connection.close()
    assert set(counts.values()) == {1}
