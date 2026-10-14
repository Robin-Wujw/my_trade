import json
import sqlite3

import numpy as np
import pandas as pd
import pytest

from stock_research.reporting import portfolio_research_audit as audit


def test_equity_uses_only_trading_days_and_keeps_first_day_loss():
    frame = pd.DataFrame({
        "date": ["2025-01-03", "2025-01-04", "2025-01-06"],
        "equity": [0.9, 1000.0, 0.99], "is_trading_day": [True, False, True],
    })
    daily = audit.equity_returns(frame)
    np.testing.assert_allclose(daily["return"], [-0.1, 0.1])
    result = audit.risk_statistics(daily["return"])
    assert result["cumulative_return"] == pytest.approx(-0.01)
    assert result["maximum_drawdown"] == pytest.approx(-0.1)
    assert result["annualized_volatility"] == pytest.approx(np.std([-0.1, 0.1], ddof=1) * np.sqrt(252))
    assert result["annualized_return"] == pytest.approx(0.99 ** 126 - 1)
    assert result["calmar"] == pytest.approx(result["annualized_return"] / 0.1)


def test_short_constant_and_zero_equity_are_explained():
    assert audit.risk_statistics([])["reason"] == "no_trading_returns"
    assert audit.risk_statistics([0.02])["sharpe"] is None
    constant = audit.risk_statistics([0.0] * 10)
    assert constant["sharpe"] is None
    assert constant["sortino"] is None
    assert "zero_return_volatility" in constant["notes"]
    assert constant["calmar"] is None
    frame = pd.DataFrame({"date": pd.bdate_range("2025-01-01", periods=3),
                          "equity": [1.0, 0.0, 0.0], "is_trading_day": True})
    result = audit.risk_statistics(audit.equity_returns(frame)["return"])
    assert result["cumulative_return"] == result["maximum_drawdown"] == -1.0
    frame.loc[2, "equity"] = 0.5
    with pytest.raises(ValueError, match="recovered from zero"):
        audit.equity_returns(frame)


@pytest.mark.parametrize("field,value", [("is_trading_day", "unknown"), ("equity", np.nan), ("date", "bad")])
def test_bad_equity_data_does_not_silently_disappear(field, value):
    frame = pd.DataFrame({"date": ["2025-01-01"], "equity": [1.0], "is_trading_day": [True]})
    frame[field] = value
    with pytest.raises(ValueError):
        audit.equity_returns(frame)


def _daily(returns):
    dates = pd.bdate_range("2025-01-01", periods=len(returns))
    return audit.equity_returns(pd.DataFrame({"date": dates, "equity": np.cumprod(1 + np.array(returns)), "is_trading_day": True}))


def test_benchmark_uses_matching_interval_endpoints_and_known_ols_coefficients():
    x = np.array([0., .01, -.02, .015, -.005, .008])
    daily = _daily(0.001 + 1.5 * x)
    close = pd.Series(100 * np.cumprod(1 + x), index=daily.index)
    result = audit.benchmark_statistics(daily, close, covariance="ols")
    assert result["observations"] == 5
    assert result["beta"] == pytest.approx(1.5)
    assert result["annualized_alpha"] == pytest.approx(.252)
    assert result["alpha_t"] is None  # An exact line has no residual variance.
    result = audit.benchmark_statistics(daily, close.drop(daily.index[2]), covariance="ols")
    # The missing date and next day's two-session index return are both excluded.
    assert result["observations"] == 3
    assert result["beta"] == pytest.approx(1.5)


def test_hac_reports_lags_and_adjusts_standard_errors():
    rng = np.random.default_rng(9182)
    x = rng.normal(0, .01, 350)
    noise = rng.normal(0, .001, 350)
    for i in range(1, len(noise)):
        noise[i] += .75 * noise[i - 1]
    daily = _daily(.0006 + .4 * x + noise)
    close = pd.Series(100 * np.cumprod(1 + x), index=daily.index)
    ols = audit.benchmark_statistics(daily, close, covariance="ols")
    hac = audit.benchmark_statistics(daily, close, covariance="hac", hac_lags=5)
    assert hac["beta"] == ols["beta"]
    assert hac["hac_lags"] == 5
    assert hac["alpha_standard_error_daily"] > ols["alpha_standard_error_daily"]
    assert abs(hac["alpha_t"]) < abs(ols["alpha_t"])


def test_no_overlap_and_flat_benchmark_return_explicit_status():
    daily = _daily([0.0, 0.01, -0.01, 0.02])
    old = pd.Series([100., 101.], index=pd.to_datetime(["2020-01-01", "2020-01-02"]))
    assert audit.benchmark_statistics(daily, old)["reason"] == "no_common_return_intervals"
    flat = pd.Series(100.0, index=daily.index)
    assert audit.benchmark_statistics(daily, flat)["regression_status"] == "zero_benchmark_volatility"


def test_best_day_removal_retains_horizon_and_is_not_a_trading_counterfactual():
    daily = _daily([.01, .20, -.10, .03])
    result = audit.best_day_diagnostics(daily, [1, 5])
    assert result[0]["risk"]["cumulative_return"] == pytest.approx(1.01 * .9 * 1.03 - 1)
    assert result[0]["risk"]["observations"] == 4
    assert result[0]["counterfactual_is_tradable"] is False
    assert result[1]["reason"] == "insufficient_trading_days"


def test_dsr_requires_registry_inputs_and_penalizes_more_trials():
    returns = np.random.default_rng(15).normal(.0008, .01, 600)
    missing = audit.deflated_sharpe(returns)
    assert missing["missing_inputs"] == ["trial_count", "trial_sharpe_std_annual"]
    assert audit.deflated_sharpe(returns, trial_count=5)["missing_inputs"] == ["trial_sharpe_std_annual"]
    one = audit.deflated_sharpe(returns, trial_count=1)
    many = audit.deflated_sharpe(returns, trial_count=100, trial_sharpe_std_annual=.5)
    assert 0 <= many["probability"] < one["probability"] <= 1
    assert audit.deflated_sharpe([.01] * 8, trial_count=1)["reason"] == "zero_return_volatility"
    with pytest.raises(ValueError):
        audit.deflated_sharpe(returns, trial_count=0)


def test_trade_concentration_and_r_losses_use_sell_rows_and_precise_pnl():
    trades = pd.DataFrame({
        "code": ["A", "A", "B", "C", "D", "E"],
        "execution_quantity": [100, -100, -100, -100, -100, -100],
        "profit_loss_amount_raw": [-5, 100.25, 50.25, -30.0, -10, 0],
        "profit_loss_amount": [-5, 100, 50, -30, -10, 0],
        "planned_reward_r_multiple": [3, 1, 2, 3, 4, np.nan],
        "realized_r_multiple": [np.nan, 3, -1, -1.6, -2, np.nan],
    })
    result = audit.trade_diagnostics(trades)
    concentration, r = result["concentration"], result["r_diagnostics"]
    assert result["sell_rows"] == 5
    assert concentration["net_realized_profit"] == pytest.approx(110.5)
    assert concentration["top"][0]["share_of_net_realized_profit"] == pytest.approx(100.25 / 110.5)
    assert r["spearman_planned_realized"] == pytest.approx(-1)
    assert r["missing_r_sell_rows"] == 1
    assert r["loss_tails"][0]["count"] == 2  # Exactly -1R is not beyond -1R.
    assert r["loss_tails"][1]["fraction_of_r_covered_sells"] == .5


def test_missing_stock_code_or_profit_is_not_silently_reported_as_complete():
    trades = pd.DataFrame({"code": [None, "A"], "execution_quantity": [-100, -100],
                           "profit_loss_amount": [25., np.nan]})
    result = audit.trade_diagnostics(trades)
    assert result["concentration"]["missing_code_sell_rows"] == 1
    assert result["concentration"]["missing_pnl_sell_rows"] == 1
    assert result["concentration"]["net_realized_profit"] == 25
    assert result["r_diagnostics"]["spearman_status"] == "fewer_than_three_pairs"
    trades["profit_loss_amount"] = np.nan
    result = audit.trade_diagnostics(trades)
    assert result["concentration"]["status"] == "unavailable"
    assert result["concentration"]["net_realized_profit"] is None


def _database(path):
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE raw_tushare_dataset_rows(dataset TEXT, row_key TEXT, ts_code TEXT, trade_date TEXT, payload_json TEXT)")
        for i, date in enumerate(pd.bdate_range("2025-01-01", periods=5)):
            for code in audit.BENCHMARKS:
                connection.execute("INSERT INTO raw_tushare_dataset_rows VALUES(?,?,?,?,?)", (
                    "index_daily", f"{code}_{i}", code, date.strftime("%Y-%m-%d"), json.dumps({"close": 100 + i}),
                ))


def test_local_database_and_cli_emit_reproducible_sources(tmp_path):
    database = tmp_path / "prices.sqlite3"
    _database(database)
    before = database.read_bytes()
    equity = tmp_path / "equity.csv"
    pd.DataFrame({"date": pd.bdate_range("2025-01-01", periods=5),
                  "equity": [1., 1.02, 1.01, 1.03, 1.04], "is_trading_day": True}).to_csv(equity, index=False)
    trades = tmp_path / "trades.csv"
    pd.DataFrame({"code": ["A"], "execution_quantity": [-100], "profit_loss_amount": [40.0]}).to_csv(trades, index=False)
    output = tmp_path / "report"
    assert audit.main(["--equity", str(equity), "--trades", str(trades), "--database", str(database),
                       "--output-directory", str(output)]) == 0
    report = json.loads((output / "portfolio_research_audit.json").read_text(encoding="utf-8"))
    assert database.read_bytes() == before
    assert report["sources"]["equity"]["sha256"] == audit._hash(equity)
    assert report["sources"]["benchmarks"]["row_count"] == 15
    assert report["sources"]["benchmarks"]["mode"] == "read_only_transaction"
    assert report["risk"]["calendar_annualized_return"] == pytest.approx(1.04 ** (365.25 / 6) - 1)
    assert report["research_only"] is True
    assert report["strict_point_in_time"] is False
    assert report["deflated_sharpe"]["status"] == "unavailable"
    assert "不是可交易反事实" in (output / "portfolio_research_audit.md").read_text(encoding="utf-8")


def test_missing_database_is_not_created_and_missing_csv_fails(tmp_path):
    database = tmp_path / "missing.sqlite3"
    prices, metadata = audit.load_benchmarks(database, end_date="2025-01-01")
    assert not database.exists()
    assert prices == {}
    assert metadata["reason"] == "database_file_missing"
    with pytest.raises(SystemExit) as error:
        audit.main(["--equity", str(tmp_path / "missing.csv"), "--trades", "missing.csv", "--output-directory", str(tmp_path)])
    assert error.value.code == 2


def test_empty_trading_period_and_empty_trade_csv_emit_unavailable_diagnostics(tmp_path):
    equity, trades = tmp_path / "equity.csv", tmp_path / "trades.csv"
    pd.DataFrame({"date": ["2025-01-01"], "equity": [1.0], "is_trading_day": [False]}).to_csv(equity, index=False)
    trades.write_text("", encoding="utf-8")
    report = audit.build_audit(equity, trades, tmp_path / "absent.sqlite3")
    assert report["risk"]["status"] == "unavailable"
    assert report["trades"]["sell_rows"] == 0
    assert report["trades"]["concentration"]["net_realized_profit"] == 0
    assert "交易日 0" in audit.render_markdown(report)
    assert not (tmp_path / "absent.sqlite3").exists()


def test_engine_frames_match_file_calculation_and_accept_empty_frames(tmp_path):
    equity = pd.DataFrame({"date": pd.bdate_range("2025-01-01", periods=5),
                           "equity": [1., 1.02, 1.01, 1.03, 1.04], "is_trading_day": True})
    result = audit.audit_frames(equity, pd.DataFrame(), tmp_path / "absent.sqlite3", trial_count=2,
                                trial_sharpe_std_annual=0.5)
    assert result["risk"]["cumulative_return"] == pytest.approx(.04)
    assert result["sources"]["equity"]["kind"] == "in_memory_engine_output"
    assert result["deflated_sharpe"]["status"] == "ok"
    empty = audit.audit_frames(pd.DataFrame(), pd.DataFrame(), tmp_path / "absent.sqlite3")
    assert empty["risk"]["status"] == "unavailable"
