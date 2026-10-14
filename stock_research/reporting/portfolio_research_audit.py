"""Offline diagnostics for frozen portfolio outputs, never trading decisions."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
import sqlite3
from statistics import NormalDist

import numpy as np
import pandas as pd


BENCHMARKS = {"000300.SH": "沪深300", "000905.SH": "中证500", "000852.SH": "中证1000"}
TRADING_DAYS = 252


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def equity_returns(frame: pd.DataFrame, *, initial_equity: float = 1.0) -> pd.DataFrame:
    """Include the first day's P&L relative to the explicit initial equity."""
    required = {"date", "equity", "is_trading_day"}
    if missing := required - set(frame):
        raise ValueError(f"equity CSV missing columns: {sorted(missing)}")
    if not math.isfinite(initial_equity) or initial_equity <= 0:
        raise ValueError("initial_equity must be finite and positive")
    flags = frame["is_trading_day"].astype(str).str.strip().str.lower()
    if not flags.isin({"true", "false", "1", "0", "1.0", "0.0"}).all():
        raise ValueError("is_trading_day contains unknown values")
    work = frame.loc[flags.isin({"true", "1", "1.0"}), ["date", "equity"]].copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce").dt.normalize()
    work["equity"] = pd.to_numeric(work["equity"], errors="coerce")
    if work["date"].isna().any() or work["date"].duplicated().any():
        raise ValueError("trading dates must be valid and unique")
    if not np.isfinite(work["equity"]).all() or work["equity"].lt(0).any():
        raise ValueError("equity must be finite and nonnegative")
    work = work.sort_values("date").set_index("date")
    previous = work["equity"].shift(1, fill_value=initial_equity)
    if (previous.eq(0) & work["equity"].gt(0)).any():
        raise ValueError("equity recovered from zero: returns require a cash-flow model")
    work["return"] = (work["equity"] / previous - 1).where(previous.ne(0), 0.0)
    work["previous_date"] = pd.Series(work.index, index=work.index).shift(1)
    return work


def risk_statistics(returns, *, risk_free_annual: float = 0.0) -> dict:
    """252 sessions/year; sample volatility; downside RMS over all sessions."""
    values = np.asarray(returns, dtype=float)
    if not np.isfinite(values).all() or (values < -1).any():
        raise ValueError("returns must be finite and at least -100%")
    if not math.isfinite(risk_free_annual) or risk_free_annual <= -1:
        raise ValueError("annual risk-free rate must be finite and greater than -1")
    n = len(values)
    notes = []
    if n == 0:
        return {"status": "unavailable", "observations": 0, "reason": "no_trading_returns"}
    growth = np.cumprod(1 + values)
    peaks = np.maximum.accumulate(np.r_[1.0, growth])[1:]
    drawdowns = np.divide(growth, peaks, out=np.zeros_like(growth), where=peaks > 0) - 1
    maximum_drawdown = float(drawdowns.min())
    annual_return = float(growth[-1] ** (TRADING_DAYS / n) - 1)
    rf_daily = math.expm1(math.log1p(risk_free_annual) / TRADING_DAYS)
    excess = values - rf_daily
    std = float(np.std(excess, ddof=1)) if n >= 2 else None
    downside = float(np.sqrt(np.mean(np.minimum(excess, 0) ** 2)))
    if n < 2:
        notes.append("fewer_than_two_returns")
    if n >= 2 and std <= 1e-15:
        notes.append("zero_return_volatility")
    if downside <= 1e-15:
        notes.append("no_downside_deviation")
    if maximum_drawdown == 0:
        notes.append("no_drawdown_for_calmar")
    if n < TRADING_DAYS:
        notes.append("annualization_from_less_than_one_year")
    return {
        "status": "ok", "observations": n,
        "cumulative_return": float(growth[-1] - 1), "annualized_return": annual_return,
        "annualized_volatility": std * math.sqrt(TRADING_DAYS) if std is not None else None,
        "sharpe": float(excess.mean() / std * math.sqrt(TRADING_DAYS)) if std is not None and std > 1e-15 else None,
        "sortino": float(excess.mean() / downside * math.sqrt(TRADING_DAYS)) if n >= 2 and downside > 1e-15 else None,
        "maximum_drawdown": maximum_drawdown,
        "calmar": annual_return / abs(maximum_drawdown) if maximum_drawdown < 0 else None,
        "risk_free_annual": risk_free_annual, "notes": notes,
    }


def benchmark_statistics(daily: pd.DataFrame, close: pd.Series, *, risk_free_annual=0.0,
                         covariance="hac", hac_lags=None) -> dict:
    """Match both endpoints of each return interval; never forward-fill prices."""
    if covariance not in {"ols", "hac"}:
        raise ValueError("covariance must be ols or hac")
    if hac_lags is not None and hac_lags < 0:
        raise ValueError("hac_lags cannot be negative")
    close = close.sort_index()
    if close.index.has_duplicates or close.index.isna().any():
        raise ValueError("benchmark dates must be valid and unique")
    if not np.isfinite(close).all() or close.le(0).any():
        raise ValueError("benchmark prices must be finite and positive")
    benchmark = pd.DataFrame({"close": close, "benchmark_return": close.pct_change(fill_method=None)})
    benchmark["benchmark_previous_date"] = pd.Series(close.index, index=close.index).shift(1)
    joined = daily.join(benchmark, how="inner")
    matched = joined[joined["previous_date"].eq(joined["benchmark_previous_date"])].dropna(
        subset=["return", "benchmark_return"]
    )
    n = len(matched)
    result = {"observations": n, "excluded_strategy_intervals": len(daily) - n,
              "covariance": covariance,
              "method": "OLS系数，Newey-West HAC标准误及t值" if covariance == "hac" else "OLS系数，独立同分布残差标准误及t值",
              "note": "仅比较起止日均一致的收益区间；缺失区间不填充。指数为价格指数，不含股息再投资。"}
    if n == 0:
        return {**result, "status": "unavailable", "reason": "no_common_return_intervals"}
    strategy = matched["return"].to_numpy(dtype=float)
    index_returns = matched["benchmark_return"].to_numpy(dtype=float)
    strategy_total = float(np.prod(1 + strategy) - 1)
    benchmark_total = float(np.prod(1 + index_returns) - 1)
    result.update({
        "status": "ok", "start_date": matched["previous_date"].iloc[0].strftime("%Y-%m-%d"),
        "end_date": matched.index[-1].strftime("%Y-%m-%d"),
        "strategy_cumulative_return": strategy_total, "benchmark_cumulative_return": benchmark_total,
        "excess_return_percentage_points": (strategy_total - benchmark_total) * 100,
        "relative_cumulative_return": (1 + strategy_total) / (1 + benchmark_total) - 1,
        "annualized_excess_return_percentage_points": ((1 + strategy_total) ** (252 / n) - (1 + benchmark_total) ** (252 / n)) * 100,
        "beta": None, "annualized_alpha": None, "alpha_t": None,
    })
    if n < 3:
        return {**result, "regression_status": "fewer_than_three_matched_returns"}
    rf = math.expm1(math.log1p(risk_free_annual) / TRADING_DAYS)
    x = np.column_stack([np.ones(n), index_returns - rf])
    if np.linalg.matrix_rank(x) < 2:
        return {**result, "regression_status": "zero_benchmark_volatility"}
    y = strategy - rf
    coefficients = np.linalg.lstsq(x, y, rcond=None)[0]
    residual = y - x @ coefficients
    inverse = np.linalg.inv(x.T @ x)
    if covariance == "ols":
        cov = inverse * float(residual @ residual) / (n - 2)
        lags = 0
    else:
        lags = min(n - 1, int(hac_lags) if hac_lags is not None else int(4 * (n / 100) ** (2 / 9)))
        scores = x * residual[:, None]
        meat = scores.T @ scores
        for lag in range(1, lags + 1):
            cross = scores[lag:].T @ scores[:-lag]
            meat += (1 - lag / (lags + 1)) * (cross + cross.T)
        cov = inverse @ meat @ inverse * (n / (n - 2))
    stderr = math.sqrt(max(0.0, float(cov[0, 0])))
    result.update({"beta": float(coefficients[1]), "annualized_alpha": float(coefficients[0] * 252),
                   "alpha_daily": float(coefficients[0]), "alpha_standard_error_daily": stderr,
                   "alpha_t": float(coefficients[0] / stderr) if stderr > 1e-15 else None,
                   "regression_status": "ok" if stderr > 1e-15 else "zero_residual_variance",
                   "hac_lags": lags if covariance == "hac" else None})
    return result


def deflated_sharpe(returns, *, trial_count=None, trial_sharpe_std_annual=None,
                    risk_free_annual=0.0) -> dict:
    """Bailey/Lopez de Prado DSR approximation; no inferred trial counts."""
    result = {"status": "unavailable", "trial_count": trial_count,
              "trial_sharpe_std_annual": trial_sharpe_std_annual,
              "assumptions": "试验次数须来自完整登记；使用有效独立试验数及跨试验Sharpe标准差。正态极值近似与收益矩估计不修复事后选样或序列相关。"}
    if trial_count is not None and (isinstance(trial_count, bool) or int(trial_count) != trial_count or trial_count < 1):
        raise ValueError("trial_count must be a positive integer")
    if trial_sharpe_std_annual is not None and (not math.isfinite(trial_sharpe_std_annual) or trial_sharpe_std_annual < 0):
        raise ValueError("trial_sharpe_std_annual must be finite and nonnegative")
    missing = []
    if trial_count is None:
        missing.append("trial_count")
    if trial_count is None or trial_count > 1:
        if trial_sharpe_std_annual is None:
            missing.append("trial_sharpe_std_annual")
    if missing:
        return {**result, "reason": "missing_trial_registry_inputs", "missing_inputs": missing}
    values = pd.Series(np.asarray(returns, dtype=float))
    if len(values) < 4:
        return {**result, "reason": "fewer_than_four_returns"}
    rf = math.expm1(math.log1p(risk_free_annual) / 252)
    excess = values - rf
    std = float(excess.std(ddof=1))
    if std <= 1e-15:
        return {**result, "reason": "zero_return_volatility"}
    sharpe = float(excess.mean() / std)
    skew, kurtosis = float(excess.skew()), float(excess.kurt() + 3)
    variance = 1 - skew * sharpe + (kurtosis - 1) * sharpe * sharpe / 4
    if not math.isfinite(variance) or variance <= 0:
        return {**result, "reason": "invalid_sharpe_sampling_variance"}
    expected_maximum = 0.0
    if trial_count > 1:
        gamma = 0.5772156649015329
        normal = NormalDist()
        expected_maximum = trial_sharpe_std_annual / math.sqrt(252) * (
            -(1 - gamma) * normal.inv_cdf(1 / trial_count)
            - gamma * normal.inv_cdf(1 / (trial_count * math.e))
        )
    statistic = (sharpe - expected_maximum) * math.sqrt(len(values) - 1) / math.sqrt(variance)
    return {**result, "status": "ok", "probability": NormalDist().cdf(statistic),
            "sharpe_daily": sharpe, "expected_maximum_sharpe_daily": expected_maximum,
            "return_skewness": skew, "return_pearson_kurtosis": kurtosis, "observations": len(values)}


def best_day_diagnostics(daily: pd.DataFrame, counts=(1, 5, 10, 20), *, risk_free_annual=0.0) -> list[dict]:
    """Set hindsight-selected best returns to zero, retaining the time horizon."""
    rows = []
    values = daily["return"]
    for count in counts:
        if int(count) != count or count <= 0:
            raise ValueError("best-day counts must be positive integers")
        if count > len(values):
            rows.append({"removed_count": count, "status": "unavailable", "reason": "insufficient_trading_days"})
            continue
        selected = values.sort_values(ascending=False, kind="stable").head(count)
        modified = values.copy()
        modified.loc[selected.index] = 0.0
        log_total = float(np.log1p(values).sum()) if values.gt(-1).all() else None
        log_selected = float(np.log1p(selected).sum()) if selected.gt(-1).all() else None
        rows.append({
            "removed_count": count, "status": "ok", "counterfactual_is_tradable": False,
            "method": "事后选出最好N天并将其收益置零，保留全部交易日；不重跑仓位、成本或现金流。",
            "dates": [date.strftime("%Y-%m-%d") for date in selected.index],
            "removed_share_of_total_log_return": log_selected / log_total if log_total is not None and abs(log_total) > 1e-15 and log_selected is not None else None,
            "risk": risk_statistics(modified, risk_free_annual=risk_free_annual),
        })
    return rows


def trade_diagnostics(trades: pd.DataFrame) -> dict:
    if trades.empty:
        trades = pd.DataFrame(columns=["code", "execution_quantity", "profit_loss_amount"])
    if "execution_quantity" in trades:
        sells = trades[pd.to_numeric(trades["execution_quantity"], errors="coerce").lt(0)].copy()
    elif "trade_side" in trades:
        sells = trades[trades["trade_side"].astype(str).str.lower().isin({"卖出", "sell"})].copy()
    else:
        raise ValueError("trades CSV needs execution_quantity or trade_side")
    concentration = {"status": "unavailable", "reason": "missing_profit_loss_or_code"}
    if "code" in sells and ("profit_loss_amount_raw" in sells or "profit_loss_amount" in sells):
        raw = pd.to_numeric(sells.get("profit_loss_amount_raw", pd.Series(index=sells.index, dtype=float)), errors="coerce").replace([np.inf, -np.inf], np.nan)
        display = pd.to_numeric(sells.get("profit_loss_amount", pd.Series(index=sells.index, dtype=float)), errors="coerce").replace([np.inf, -np.inf], np.nan)
        pnl = raw.fillna(display)
        codes = sells["code"].fillna("<missing_code>").astype(str).str.strip().replace("", "<missing_code>")
        by_code = pnl.groupby(codes).sum(min_count=1).dropna().sort_values(ascending=False)
        total = float(by_code.sum())
        positive = float(by_code.clip(lower=0).sum())
        top = []
        for count in (1, 3):
            amount = float(by_code.head(count).clip(lower=0).sum())
            top.append({"count": count, "profit": amount,
                        "share_of_net_realized_profit": amount / total if total > 0 else None,
                        "share_of_positive_stock_profit": amount / positive if positive > 0 else None,
                        "remaining_realized_profit": total - amount})
        concentration = {
            "status": "ok" if len(by_code) or not len(sells) else "unavailable",
            "net_realized_profit": total if len(by_code) or not len(sells) else None, "positive_stock_profit": positive,
            "missing_pnl_sell_rows": int(pnl.isna().sum()), "rounded_fallback_rows": int((raw.isna() & display.notna()).sum()),
            "missing_code_sell_rows": int(codes.eq("<missing_code>").sum()),
            "top": top, "stocks": [{"code": str(code), "profit": float(value)} for code, value in by_code.items()],
            "note": "仅卖出已实现价格净损益；不含股息/未实现收益。去Top1/3为静态利润扣除，不是组合反事实收益。净盈利非正时不报告净盈利占比。",
        }
    realized = pd.to_numeric(sells.get("realized_r_multiple", pd.Series(index=sells.index, dtype=float)), errors="coerce").replace([np.inf, -np.inf], np.nan)
    planned = pd.to_numeric(sells.get("planned_reward_r_multiple", pd.Series(index=sells.index, dtype=float)), errors="coerce").replace([np.inf, -np.inf], np.nan)
    valid_r = realized.dropna()
    paired = pd.DataFrame({"planned": planned, "realized": realized}).dropna()
    paired = paired[paired["planned"].gt(0)]
    correlation = None
    if len(paired) >= 3 and paired["planned"].nunique() > 1 and paired["realized"].nunique() > 1:
        correlation = float(paired["planned"].rank().corr(paired["realized"].rank()))
    correlation_status = "ok" if correlation is not None else "fewer_than_three_pairs" if len(paired) < 3 else "constant_planned_or_realized_r"
    groups = []
    if len(paired) >= 3:
        for label, positions in zip(("低", "中", "高"), np.array_split(np.arange(len(paired)), 3)):
            group = paired.sort_values("planned", kind="stable").iloc[positions]
            groups.append({"group": label, "count": len(group), "planned_median": float(group["planned"].median()),
                           "realized_mean": float(group["realized"].mean()), "realized_median": float(group["realized"].median()),
                           "win_rate": float(group["realized"].gt(0).mean())})
    tails = [{"loss_threshold_r": threshold, "count": int(valid_r.lt(-threshold).sum()),
              "fraction_of_r_covered_sells": float(valid_r.lt(-threshold).mean()) if len(valid_r) else None}
             for threshold in (1.0, 1.5)]
    return {"sell_rows": len(sells), "concentration": concentration,
            "r_diagnostics": {"status": "ok" if len(valid_r) else "unavailable", "covered_sell_rows": len(valid_r),
                              "missing_r_sell_rows": len(sells) - len(valid_r), "paired_sell_rows": len(paired),
                              "spearman_planned_realized": correlation, "worst_realized_r": float(valid_r.min()) if len(valid_r) else None,
                              "spearman_status": correlation_status,
                              "loss_tails": tails, "planned_terciles": groups,
                              "note": "样本为分批卖出记录，非独立交易；不报告显著性p值。三等分按计划R稳定排序，同值可跨组；超过阈值采用严格小于-1/-1.5。"}}


def load_benchmarks(database, *, end_date) -> tuple[dict[str, pd.Series], dict]:
    path = Path(database).resolve()
    if not path.is_file():
        return {}, {"status": "unavailable", "reason": "database_file_missing", "path": str(path)}
    try:
        with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as connection:
            connection.execute("PRAGMA query_only=ON")
            connection.execute("BEGIN")
            records = connection.execute(
                "SELECT ts_code, trade_date, payload_json FROM raw_tushare_dataset_rows "
                "WHERE dataset='index_daily' AND ts_code IN (?,?,?) AND trade_date<=? ORDER BY ts_code,trade_date,row_key",
                (*BENCHMARKS, end_date),
            ).fetchall()
    except sqlite3.Error as exc:
        return {}, {"status": "unavailable", "reason": "benchmark_query_failed", "detail": str(exc), "path": str(path)}
    frames = {code: [] for code in BENCHMARKS}
    invalid = 0
    for code, date, payload in records:
        try:
            close = float(json.loads(payload)["close"])
            date = pd.Timestamp(date).normalize()
            if not math.isfinite(close) or close <= 0 or pd.isna(date):
                raise ValueError("invalid index price/date")
            frames[code].append((date, close))
        except (KeyError, TypeError, ValueError):
            invalid += 1
    result = {}
    for code, rows in frames.items():
        frame = pd.DataFrame(rows, columns=["date", "close"])
        if not frame.empty:
            conflicts = frame.groupby("date")["close"].nunique().gt(1)
            if conflicts.any():
                raise ValueError(f"conflicting benchmark prices for {code}: {list(conflicts[conflicts].index)}")
            result[code] = frame.drop_duplicates("date").set_index("date")["close"].sort_index()
    fingerprint = hashlib.sha256(json.dumps(records, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()
    return result, {"status": "ok", "path": str(path), "mode": "read_only_transaction", "row_count": len(records),
                    "invalid_rows": invalid, "selected_rows_sha256": fingerprint}


def audit_frames(equity, trades, database, *, initial_equity=1.0, risk_free_annual=0.0,
                 covariance="hac", hac_lags=None, trial_count=None, trial_sharpe_std_annual=None,
                 best_days=(1, 5, 10, 20), sources=None) -> dict:
    """Shared calculation for engine frames and frozen CSVs; never refresh data."""
    if equity.empty and not len(equity.columns):
        equity = pd.DataFrame(columns=["date", "equity", "is_trading_day"])
    if sources is None:
        sources = {name: {"kind": "in_memory_engine_output", "rows": len(frame),
                          "serialization": "pandas_json_split_iso",
                          "sha256": hashlib.sha256(frame.to_json(orient="split", date_format="iso", double_precision=15).encode()).hexdigest()}
                   for name, frame in (("equity", equity), ("trades", trades))}
    daily = equity_returns(equity, initial_equity=initial_equity)
    if daily.empty:
        benchmark_prices, database_source = {}, {"status": "unavailable", "reason": "no_trading_dates"}
    else:
        benchmark_prices, database_source = load_benchmarks(database, end_date=daily.index[-1].strftime("%Y-%m-%d"))
    risk = risk_statistics(daily["return"], risk_free_annual=risk_free_annual)
    elapsed_days = (daily.index[-1] - daily.index[0]).days if len(daily) >= 2 else 0
    calendar_return = (1 + risk["cumulative_return"]) ** (365.25 / elapsed_days) - 1 if elapsed_days > 0 else None
    risk["calendar_annualized_return"] = calendar_return
    risk["calendar_calmar"] = calendar_return / abs(risk["maximum_drawdown"]) if calendar_return is not None and risk.get("maximum_drawdown", 0) < 0 else None
    return {
        "version": "portfolio-research-audit-v1", "research_only": True, "strict_point_in_time": False,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(), "sources": {**sources, "benchmarks": database_source},
        "methodology": {"trading_days_per_year": 252, "initial_equity": initial_equity,
                        "nontrading_rows_excluded": len(equity) - len(daily),
                        "returns": "仅交易日权益；首日相对显式initial_equity计算；零权益后保持零视为破产吸收态。",
                        "ratios": "Sharpe用样本标准差；Sortino下行均方根含全部交易日；Calmar=CAGR/绝对最大回撤；初始权益纳入回撤高点。",
                        "annualization": "收益主口径按252/交易日数年化；另报首末交易日实际间隔/365.25的日历年化，与旧冻结报告核对。两种Calmar分别使用对应年化收益。",
                        "alpha": "日超额收益OLS截距乘252；HAC采用Bartlett权重及n/(n-2)修正。",
                        "limitations": "冻结产物的历史财务修订、行业时点、成交成本及公司行动限制继续适用；本报告不认证策略优势或修改买卖规则。"},
        "period": {"start": daily.index[0].strftime("%Y-%m-%d") if len(daily) else None,
                   "end": daily.index[-1].strftime("%Y-%m-%d") if len(daily) else None},
        "risk": risk,
        "benchmarks": {code: {"name": name, **(
            benchmark_statistics(daily, benchmark_prices[code], risk_free_annual=risk_free_annual, covariance=covariance, hac_lags=hac_lags)
            if code in benchmark_prices else {"status": "unavailable", "reason": "benchmark_prices_missing"})}
            for code, name in BENCHMARKS.items()},
        "best_days": best_day_diagnostics(daily, best_days, risk_free_annual=risk_free_annual),
        "trades": trade_diagnostics(trades),
        "deflated_sharpe": deflated_sharpe(daily["return"], trial_count=trial_count,
                                           trial_sharpe_std_annual=trial_sharpe_std_annual, risk_free_annual=risk_free_annual),
    }


def build_audit(equity_path, trades_path, database, **options) -> dict:
    sources = {name: {"path": str(Path(path).resolve()), "sha256": _hash(path)}
               for name, path in (("equity", equity_path), ("trades", trades_path))}
    try:
        equity = pd.read_csv(equity_path, low_memory=False)
    except pd.errors.EmptyDataError:
        equity = pd.DataFrame()
    try:
        trades = pd.read_csv(trades_path, dtype={"code": str}, low_memory=False)
    except pd.errors.EmptyDataError:
        trades = pd.DataFrame()
    report = audit_frames(equity, trades, database, sources=sources, **options)
    for source in sources.values():
        if _hash(source["path"]) != source["sha256"]:
            raise ValueError(f"input changed during audit: {source['path']}")
    return report


def render_markdown(report: dict) -> str:
    def number(value, percent=False):
        if value is None:
            return "不可计算"
        return f"{value:.2%}" if percent else f"{value:.4f}"
    risk = report["risk"]
    lines = ["# 组合研究诊断", "", "口径：research_only；不构成严格时点认证。", "",
             f"区间：{report['period']['start']} 至 {report['period']['end']}；交易日 {risk['observations']}。", "",
             "| 指标 | 数值 |", "|---|---:|"]
    for label, key, percent in (("累计收益", "cumulative_return", True), ("年化收益（252交易日）", "annualized_return", True),
                                ("年化收益（日历间隔）", "calendar_annualized_return", True),
                                ("年化波动", "annualized_volatility", True), ("Sharpe", "sharpe", False),
                                ("Sortino", "sortino", False), ("最大回撤", "maximum_drawdown", True),
                                ("Calmar（252交易日）", "calmar", False), ("Calmar（日历间隔）", "calendar_calmar", False)):
        lines.append(f"| {label} | {number(risk.get(key), percent)} |")
    lines += ["", f"风险统计诊断：{', '.join(risk.get('notes', [])) or risk.get('reason', '无')}。", "",
              "## 共同日期基准回归", "", "| 基准 | 收益区间数 | beta | 年化alpha | alpha t | 累计超额百分点 |",
              "|---|---:|---:|---:|---:|---:|"]
    for item in report["benchmarks"].values():
        lines.append(f"| {item['name']} | {item.get('observations', 0)} | {number(item.get('beta'))} | {number(item.get('annualized_alpha'), True)} | {number(item.get('alpha_t'))} | {number(item.get('excess_return_percentage_points'))} |")
    for item in report["benchmarks"].values():
        lines.append(f"\n{item['name']}：{item.get('start_date', '无')} 至 {item.get('end_date', '无')}；{item.get('method', item.get('reason'))}；状态 {item.get('regression_status', item['status'])}；HAC滞后 {item.get('hac_lags', '不适用')}。")
    lines += ["", "指数使用价格指数；仅配对起止日相同的收益区间。alpha未校正事后选模，多重检验另见DSR。", "",
              "## 最好交易日移除诊断", "", "事后把最好N天收益置零，保持原日数；不是可交易反事实。", "",
              "| 移除天数 | 剩余累计收益 | 最好N天占总对数收益 |", "|---|---:|---:|"]
    for item in report["best_days"]:
        lines.append(f"| {item['removed_count']} | {number(item.get('risk', {}).get('cumulative_return'), True)} | {number(item.get('removed_share_of_total_log_return'), True)} |")
    concentration = report["trades"]["concentration"]
    lines += ["", "## 已实现盈利集中度", "", concentration.get("note", concentration.get("reason", "")), "",
              f"已实现净盈利：{number(concentration.get('net_realized_profit'))}。", ""]
    for item in concentration.get("top", []):
        lines.append(f"- Top{item['count']}占净盈利 {number(item['share_of_net_realized_profit'], True)}；静态扣除后盈利 {number(item['remaining_realized_profit'])}。")
    r = report["trades"]["r_diagnostics"]
    lines += ["", "## 计划R与亏损尾部", "", r["note"], "",
              f"全部卖出 {report['trades']['sell_rows']} 笔；有R {r['covered_sell_rows']} 笔；成对计划/实际R {r['paired_sell_rows']} 笔。",
              f"Spearman：{number(r['spearman_planned_realized'])}；最差实际R：{number(r['worst_realized_r'])}。", ""]
    for item in r["loss_tails"]:
        lines.append(f"- 亏损超过{item['loss_threshold_r']:g}R：{item['count']}笔，占有R卖出 {number(item['fraction_of_r_covered_sells'], True)}。")
    lines += ["", "| 计划R组 | 样本数 | 计划R中位 | 实际R均值 | 实际R中位 | 胜率 |", "|---|---:|---:|---:|---:|---:|"]
    for group in r["planned_terciles"]:
        lines.append(f"| {group['group']} | {group['count']} | {number(group['planned_median'])} | {number(group['realized_mean'])} | {number(group['realized_median'])} | {number(group['win_rate'], True)} |")
    dsr = report["deflated_sharpe"]
    lines += ["", "## 多重检验诊断", "", f"Deflated Sharpe概率：{number(dsr.get('probability'), True)}；试验次数：{dsr['trial_count'] if dsr['trial_count'] is not None else '未提供'}。",
              f"状态：{dsr.get('reason', dsr['status'])}；缺失输入：{', '.join(dsr.get('missing_inputs', [])) or '无'}。",
              dsr["assumptions"], "", "## 复算口径", ""]
    lines += [f"- {value}" for value in report["methodology"].values() if isinstance(value, str)]
    lines += ["", "完整输入指纹、逐股利润、移除日期及数值见研究诊断JSON或组合summary.json中的research_diagnostics。", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description="只读审计冻结组合风险、基准、盈利集中度及R尾部")
    parser.add_argument("--equity", required=True)
    parser.add_argument("--trades", required=True)
    parser.add_argument("--database", default="var/data/my_trade.sqlite3")
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--initial-equity", type=float, default=1.0, help="CSV首日前权益；项目归一化equity默认1")
    parser.add_argument("--risk-free-annual", type=float, default=0.0)
    parser.add_argument("--alpha-covariance", choices=("ols", "hac"), default="hac")
    parser.add_argument("--hac-lags", type=int)
    parser.add_argument("--trial-count", type=int, help="已登记的有效独立试验次数，禁止由目录数猜测")
    parser.add_argument("--trial-sharpe-std-annual", type=float, help="跨已登记试验的年化Sharpe标准差")
    parser.add_argument("--best-days", type=int, nargs="+", default=[1, 5, 10, 20])
    args = parser.parse_args(argv)
    try:
        report = build_audit(args.equity, args.trades, args.database, initial_equity=args.initial_equity,
                             risk_free_annual=args.risk_free_annual, covariance=args.alpha_covariance, hac_lags=args.hac_lags,
                             trial_count=args.trial_count, trial_sharpe_std_annual=args.trial_sharpe_std_annual,
                             best_days=args.best_days)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    output = Path(args.output_directory)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "portfolio_research_audit.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    markdown_path = output / "portfolio_research_audit.md"
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    print(json_path)
    print(markdown_path)
    return 0
