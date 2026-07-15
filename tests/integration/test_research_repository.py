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
            "signal_eligible": True, "trade_basis_score": 7,
            "trade_basis_reason": "MA20/MA60同步上扬",
            "technical_alignment": "trade_ready",
            "ima_web_validation": "aligned",
            "return_20d": 0.12,
            "return_60d": 0.28,
            "return_120d": 0.45,
            "distance_120d_high": -0.03,
            "leadership_score": 22,
            "leadership_reason": "20日强度+12.0%；距120日高点-3.0%",
            "long_term_structure_favorable": True,
        }],
        "2026-01-06": [],
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
            "selection_reason": "主流标准基本面模型入选",
            "trade_basis_reason": "MA20/MA60同步上扬",
            "technical_alignment": "trade_ready",
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
                "derived.candidate_snapshot_coverage",
                "derived.formula33_phase", "derived.backtest_runs",
                "derived.backtest_trades", "derived.backtest_positions",
            ]
        }
    finally:
        connection.close()
    assert counts["derived.candidate_snapshot_coverage"] == 2
    assert {value for key, value in counts.items() if key != "derived.candidate_snapshot_coverage"} == {1}
    connection = database.connect(read_only=True)
    try:
        row = connection.execute(
            """
            SELECT trade_basis_score, technical_alignment, ima_web_validation,
                   leadership_score, long_term_structure_favorable
            FROM derived.candidate_snapshots
            """
        ).fetchone()
        trade = connection.execute(
            """
            SELECT selection_reason, trade_basis_reason, technical_alignment
            FROM derived.backtest_trades
            """
        ).fetchone()
        coverage = connection.execute(
            """
            SELECT observation_date, candidate_count
            FROM derived.candidate_snapshot_coverage
            ORDER BY observation_date
            """
        ).fetchall()
    finally:
        connection.close()
    assert row == (7.0, "trade_ready", "aligned", 22.0, True)
    assert trade == (
        "主流标准基本面模型入选",
        "MA20/MA60同步上扬",
        "trade_ready",
    )
    assert [item[1] for item in coverage] == [1, 0]
