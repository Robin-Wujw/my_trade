"""Persistence boundary for point-in-time research and backtest artifacts."""
from __future__ import annotations

import json
from uuid import uuid4

import pandas as pd

from .database import Database


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, default=str, sort_keys=True)


class ResearchRepository:
    def __init__(self, database: Database):
        self.database = database

    @staticmethod
    def _replace_frame(connection, table: str, frame: pd.DataFrame, delete_sql: str, params) -> None:
        connection.execute(delete_sql, params)
        if frame.empty:
            return
        connection.register("incoming_research_rows", frame)
        try:
            columns = ", ".join(frame.columns)
            connection.execute(
                f"INSERT INTO {table} ({columns}) SELECT {columns} FROM incoming_research_rows"
            )
        finally:
            connection.unregister("incoming_research_rows")

    def persist_fundamentals(self, financial_by_period: dict) -> int:
        rows = []
        for period, values in financial_by_period.items():
            for code, payload in values.items():
                rows.append({
                    "code": str(code).split(".")[-1].zfill(6),
                    "report_period": pd.Timestamp(period),
                    "quality_score": pd.to_numeric(payload.get("quality_score"), errors="coerce"),
                    "earnings_yoy": pd.to_numeric(payload.get("yoy"), errors="coerce"),
                    "market_cap": pd.to_numeric(payload.get("mktcap"), errors="coerce"),
                    "value_line": pd.to_numeric(payload.get("value_line"), errors="coerce"),
                    "payload_json": _json(payload),
                })
        frame = pd.DataFrame(rows)
        if frame.empty:
            return 0
        connection = self.database.connect()
        connection.execute("BEGIN TRANSACTION")
        try:
            connection.register("incoming_fundamental_rows", frame)
            connection.execute(
                """
                DELETE FROM raw.fundamental_metrics AS stored
                USING incoming_fundamental_rows AS incoming
                WHERE stored.code = incoming.code
                  AND stored.report_period = incoming.report_period
                """
            )
            columns = ", ".join(frame.columns)
            connection.execute(
                f"INSERT INTO raw.fundamental_metrics ({columns}) SELECT {columns} FROM incoming_fundamental_rows"
            )
            connection.unregister("incoming_fundamental_rows")
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()
        return len(frame)

    def persist_candidate_snapshots(self, snapshots: dict, *, version: str) -> int:
        rows = []
        coverage_rows = []
        for date, candidates in snapshots.items():
            eligible_count = sum(bool(candidate.get("signal_eligible", True)) for candidate in candidates)
            coverage_rows.append({
                "observation_date": pd.Timestamp(date),
                "snapshot_version": str(version),
                "candidate_count": int(len(candidates)),
                "signal_eligible_count": int(eligible_count),
                "payload_json": _json({
                    "date": date,
                    "version": version,
                    "candidate_count": int(len(candidates)),
                    "signal_eligible_count": int(eligible_count),
                }),
            })
            for candidate in candidates:
                rows.append({
                    "observation_date": pd.Timestamp(date),
                    "snapshot_version": str(version),
                    "code": str(candidate.get("code") or ""),
                    "name": str(candidate.get("name") or ""),
                    "selection_rank": candidate.get("selection_rank"),
                    "candidate_score": candidate.get("candidate_score"),
                    "candidate_source": str(candidate.get("candidate_source") or ""),
                    "selection_reason": str(candidate.get("selection_reason") or ""),
                    "report_period": pd.to_datetime(candidate.get("report_period"), errors="coerce"),
                    "signal_eligible": bool(candidate.get("signal_eligible", True)),
                    "trade_basis_score": pd.to_numeric(candidate.get("trade_basis_score"), errors="coerce"),
                    "trade_basis_reason": str(candidate.get("trade_basis_reason") or ""),
                    "technical_alignment": str(candidate.get("technical_alignment") or ""),
                    "ima_web_validation": str(candidate.get("ima_web_validation") or ""),
                    "validation_sources_json": _json(candidate.get("validation_sources") or []),
                    "payload_json": _json(candidate),
                })
        frame = pd.DataFrame(rows)
        coverage = pd.DataFrame(coverage_rows)
        dates = sorted(pd.Timestamp(value) for value in snapshots)
        if not dates:
            return 0
        connection = self.database.connect()
        connection.execute("BEGIN TRANSACTION")
        try:
            self._replace_frame(
                connection,
                "derived.candidate_snapshot_coverage",
                coverage,
                """
                DELETE FROM derived.candidate_snapshot_coverage
                WHERE snapshot_version = ? AND observation_date >= ? AND observation_date <= ?
                """,
                [str(version), dates[0], dates[-1]],
            )
            self._replace_frame(
                connection,
                "derived.candidate_snapshots",
                frame,
                """
                DELETE FROM derived.candidate_snapshots
                WHERE snapshot_version = ? AND observation_date >= ? AND observation_date <= ?
                """,
                [str(version), dates[0], dates[-1]],
            )
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()
        return len(frame)

    def persist_formula_history(self, frame: pd.DataFrame, *, version: str) -> int:
        if frame is None or frame.empty:
            return 0
        rows = []
        for row in frame.to_dict("records"):
            rows.append({
                "observation_date": pd.Timestamp(row["date"]),
                "version": str(version),
                "phase": str(row.get("phase") or "waiting"),
                "window_up_streak": int(row.get("window_up_streak") or 0),
                "window_down_streak": int(row.get("window_down_streak") or 0),
                "payload_json": _json(row),
            })
        data = pd.DataFrame(rows)
        connection = self.database.connect()
        connection.execute("BEGIN TRANSACTION")
        try:
            self._replace_frame(
                connection,
                "derived.formula33_phase",
                data,
                """
                DELETE FROM derived.formula33_phase
                WHERE version = ? AND observation_date >= ? AND observation_date <= ?
                """,
                [str(version), data["observation_date"].min(), data["observation_date"].max()],
            )
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()
        return len(data)

    def persist_backtest_result(self, result: dict, *, run_id: str | None = None) -> str:
        run_id = str(run_id or uuid4())
        summary = {
            key: value for key, value in result.items()
            if key not in {"events", "equity_curve", "trade_ledger"}
        }
        run = pd.DataFrame([{
            "run_id": run_id,
            "requested_start": pd.Timestamp(result["requested_start"]),
            "actual_start": pd.to_datetime(result.get("actual_start"), errors="coerce"),
            "end_date": pd.Timestamp(result["end_date"]),
            "initial_capital": float(result.get("initial_capital") or 0),
            "final_return_pct": result.get("final_return_pct"),
            "maximum_drawdown_pct": result.get("maximum_drawdown_pct"),
            "final_cash": result.get("final_cash"),
            "summary_json": _json(summary),
        }])
        trades = pd.DataFrame([{
            "run_id": run_id,
            "sequence": sequence,
            "trade_date": pd.Timestamp(event["date"]),
            "code": str(event.get("code") or ""),
            "name": str(event.get("name") or ""),
            "trade_side": str(event.get("trade_side") or ""),
            "quantity": float(event.get("quantity") or 0),
            "execution_price": event.get("execution_price", event.get("price")),
            "trade_amount": event.get("trade_amount"),
            "transaction_cost_amount": event.get("transaction_cost_amount"),
            "profit_loss_amount": event.get("profit_loss_amount"),
            "reason": str(event.get("reason") or ""),
            "selection_reason": str(event.get("selection_reason") or ""),
            "trade_basis_reason": str(event.get("trade_basis_reason") or ""),
            "technical_alignment": str(event.get("technical_alignment") or ""),
            "payload_json": _json(event),
        } for sequence, event in enumerate(result.get("trade_ledger") or [], 1)])
        positions = pd.DataFrame([{
            "run_id": run_id,
            "code": str(item.get("code") or ""),
            "name": str(item.get("name") or ""),
            "quantity": item.get("quantity"),
            "cost": item.get("cost"),
            "close": item.get("close"),
            "market_value": item.get("market_value"),
            "unrealized_pnl_amount": item.get("unrealized_pnl_amount"),
            "payload_json": _json(item),
        } for item in result.get("final_positions") or []])
        connection = self.database.connect()
        connection.execute("BEGIN TRANSACTION")
        try:
            for table, data in [
                ("derived.backtest_runs", run),
                ("derived.backtest_trades", trades),
                ("derived.backtest_positions", positions),
            ]:
                connection.execute(f"DELETE FROM {table} WHERE run_id = ?", [run_id])
                if data.empty:
                    continue
                connection.register("incoming_backtest_rows", data)
                columns = ", ".join(data.columns)
                connection.execute(
                    f"INSERT INTO {table} ({columns}) SELECT {columns} FROM incoming_backtest_rows"
                )
                connection.unregister("incoming_backtest_rows")
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()
        return run_id
