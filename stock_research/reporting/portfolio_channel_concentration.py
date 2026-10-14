"""Descriptive original-channel contribution; never a tradable counterfactual.

This report reads existing executions. It cannot establish the causal benefit
of a channel sharing capital with another, or validate historical PIT sources.
"""
from __future__ import annotations

import argparse
import ast
from collections import defaultdict
import csv
from datetime import date
import hashlib
import json
import math
from pathlib import Path


def _blank(value):
    return value is None or (isinstance(value, str) and not value.strip()) or (
        isinstance(value, float) and math.isnan(value))


def _number(value, name, *, optional=False, minimum=None):
    if _blank(value) and optional:
        return 0.
    if isinstance(value, bool):
        raise ValueError(f"invalid_{name}")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"missing_or_invalid_{name}") from exc
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise ValueError(f"invalid_{name}")
    return result


def _structured(value):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, TypeError):
            try:
                value = ast.literal_eval(value)  # pandas CSV exports Python list/dict literals.
            except (ValueError, SyntaxError) as exc:
                raise ValueError("invalid_origin_allocations") from exc
    return value


def _allocations(value):
    value = _structured(value)
    if not isinstance(value, list) or not value:
        raise ValueError("missing_origin_allocations")
    result = []
    for row in value:
        if not isinstance(row, dict) or row.get("origin_account") not in {"left", "right"}:
            raise ValueError("unknown_original_channel")
        quantity = _number(row.get("quantity"), "allocated_quantity", minimum=0)
        if quantity == 0:
            raise ValueError("zero_allocated_quantity")
        result.append({"channel": row["origin_account"], "quantity": quantity,
                       "cost": _number(row.get("cost_amount"), "allocated_cost", minimum=0),
                       "entry_fee": _number(row.get("entry_fee_amount"), "allocated_entry_fee", minimum=0)})
    return result


def _concentration(stocks):
    total = sum(row["realized_net_profit"] for row in stocks)
    winners = sorted((row for row in stocks if row["realized_net_profit"] > 0),
                     key=lambda row: (-row["realized_net_profit"], row["code"]))
    positive = sum(row["realized_net_profit"] for row in winners)
    return [{"count": count, "selected_stock_count": len(winners[:count]),
             "codes": [row["code"] for row in winners[:count]],
             "profit": sum(row["realized_net_profit"] for row in winners[:count]),
             "share_of_net_realized_profit": sum(row["realized_net_profit"] for row in winners[:count]) / total if total > 0 else None,
             "share_of_positive_stock_profit": sum(row["realized_net_profit"] for row in winners[:count]) / positive if positive > 0 else None,
             "remaining_static_realized_profit": total - sum(row["realized_net_profit"] for row in winners[:count]),
             "counterfactual_is_tradable": False} for count in (1, 3)]


def build_channel_contribution(events, final_positions, *, initial_capital):
    """Allocate net sales and settled dividends to the original left/right lots.

    Open-position fees remain in unrealized PnL. Fees paid are also displayed,
    but are never deducted a second time. Missing attribution is disclosed and
    excluded from covered totals instead of guessed from current management.
    """
    capital = _number(initial_capital, "initial_capital", minimum=0)
    if capital == 0:
        raise ValueError("initial_capital must be positive")
    groups, issues = {}, []
    counts = {"event_rows": 0, "covered_cash_or_trade_rows": 0, "excluded_cash_or_trade_rows": 0,
              "rounded_fee_fallback_rows": 0, "final_position_rows": 0, "excluded_final_position_rows": 0}
    def bucket(channel, code):
        return groups.setdefault((channel, code), {"origin_channel": channel, "code": code,
            "realized_price_net_profit": 0., "settled_dividends": 0., "dividend_receivable": 0.,
            "unrealized_net_profit": 0., "fees_paid": 0., "open_quantity": 0.,
            "pending_listing_quantity": 0., "sell_allocations": 0})
    def code_of(row):
        value = row.get("code")
        if _blank(value):
            raise ValueError("missing_code")
        return str(value).strip()
    for index, event in enumerate(events):
        counts["event_rows"] += 1
        try:
            if _blank(event.get("execution_quantity")) and event.get("trade_side") in {"买入", "卖出", "buy", "sell"}:
                raise ValueError("missing_execution_quantity")
            quantity = _number(event.get("execution_quantity"), "execution_quantity", optional=True)
            dividend = _number(event.get("corporate_action_cash_amount"), "dividend_cash", optional=True, minimum=0)
            receivable = _number(event.get("dividend_receivable_change"), "dividend_receivable", optional=True)
            if not quantity and not dividend and not receivable:
                continue
            code = code_of(event)
            allocations = _allocations(event.get("origin_allocations"))
            allocated_quantity = sum(row["quantity"] for row in allocations)
            if quantity and not math.isclose(abs(quantity), allocated_quantity, rel_tol=1e-10, abs_tol=1e-6):
                raise ValueError("allocated_quantity_mismatch")
            fee, fallback = 0., False
            if quantity:
                fallback = _blank(event.get("transaction_cost_amount_raw"))
                fee = _number(event.get("transaction_cost_amount") if fallback else event["transaction_cost_amount_raw"],
                              "transaction_cost", minimum=0)
            price = _number(event.get("execution_price"), "execution_price", minimum=0) if quantity < 0 else 0.
            if quantity < 0:
                net = price * allocated_quantity - sum(row["cost"] + row["entry_fee"] for row in allocations) - fee
                if not _blank(event.get("profit_loss_amount_raw")) and not math.isclose(
                        net, _number(event["profit_loss_amount_raw"], "realized_profit"), rel_tol=1e-10, abs_tol=1e-5):
                    raise ValueError("allocated_profit_mismatch")
            # Validate the entire row before changing any channel totals.
            for allocation in allocations:
                item = bucket(allocation["channel"], code)
                share = allocation["quantity"] / allocated_quantity
                item["fees_paid"] += fee * share
                item["settled_dividends"] += dividend * share
                item["dividend_receivable"] += receivable * share
                if quantity < 0:
                    item["sell_allocations"] += 1
                    item["realized_price_net_profit"] += price * allocation["quantity"] - allocation["cost"] - allocation["entry_fee"] - fee * share
            counts["covered_cash_or_trade_rows"] += 1
            counts["rounded_fee_fallback_rows"] += int(fallback)
        except ValueError as exc:
            counts["excluded_cash_or_trade_rows"] += 1
            issues.append({"scope": "event", "row": index, "reason": str(exc)})
    positions_known = isinstance(final_positions, list)
    if not positions_known:
        issues.append({"scope": "final_positions", "reason": "missing_terminal_inventory"})
    terminal_codes = set()
    for index, position in enumerate(final_positions if positions_known else []):
        counts["final_position_rows"] += 1
        try:
            code = code_of(position)
            if code in terminal_codes:
                raise ValueError("duplicate_terminal_position")
            terminal_codes.add(code)
            quantity = _number(position.get("quantity"), "position_quantity", minimum=0)
            close = _number(position.get("close"), "terminal_raw_close", minimum=0)
            lots = position.get("left_batches", []) + position.get("batches", [])
            if not isinstance(lots, list) or (quantity and not lots):
                raise ValueError("missing_terminal_lots")
            allocations = []
            for lot in lots:
                allocation = _allocations([lot.get("origin_allocation")])[0]
                if not math.isclose(_number(lot.get("quantity"), "terminal_lot_quantity", minimum=0),
                                    allocation["quantity"], rel_tol=1e-10, abs_tol=1e-6):
                    raise ValueError("terminal_lot_quantity_mismatch")
                pending = lot.get("corporate_action_pending", False)
                if type(pending) is not bool:
                    raise ValueError("invalid_pending_listing_flag")
                allocations.append((allocation, pending))
            if not math.isclose(quantity, sum(row["quantity"] for row, _ in allocations), rel_tol=1e-10, abs_tol=1e-6):
                raise ValueError("terminal_position_quantity_mismatch")
            for allocation, pending in allocations:
                item = bucket(allocation["channel"], code)
                item["open_quantity"] += allocation["quantity"]
                item["pending_listing_quantity"] += allocation["quantity"] if pending else 0.
                item["unrealized_net_profit"] += close * allocation["quantity"] - allocation["cost"] - allocation["entry_fee"]
        except (ValueError, TypeError) as exc:
            counts["excluded_final_position_rows"] += 1
            issues.append({"scope": "final_position", "row": index, "reason": str(exc)})
    inventory_complete = positions_known and counts["excluded_final_position_rows"] == 0
    stocks = []
    for item in groups.values():
        item["realized_net_profit"] = item["realized_price_net_profit"] + item["settled_dividends"]
        item["realized_contribution_pct_initial_capital"] = item["realized_net_profit"] / capital * 100
        item["outcome_status"] = ("terminal_inventory_unknown" if not inventory_complete else
            "open_position" if item["open_quantity"] > 0 else
            "attribution_incomplete" if counts["excluded_cash_or_trade_rows"] else
            "shares_closed_pending_dividend" if abs(item["dividend_receivable"]) > 1e-6 else "closed_and_settled")
        if item["dividend_receivable"] < -1e-6:
            issues.append({"scope": "stock", "code": item["code"], "reason": "negative_dividend_receivable"})
        stocks.append(item)
    by_channel = []
    amount_fields = ("realized_net_profit", "realized_price_net_profit", "settled_dividends",
                     "dividend_receivable", "unrealized_net_profit", "fees_paid")
    for channel in ("left", "right"):
        rows = sorted((row for row in stocks if row["origin_channel"] == channel),
                      key=lambda row: (-row["realized_net_profit"], row["code"]))
        amounts = {field: sum(row[field] for row in rows) for field in amount_fields}
        by_channel.append({"origin_channel": channel, **amounts, "stock_count": len(rows),
            "realized_contribution_pct_initial_capital": amounts["realized_net_profit"] / capital * 100,
            "unsettled_stock_count": sum(row["outcome_status"] != "closed_and_settled" for row in rows),
            "closed_and_settled_realized_profit": sum(row["realized_net_profit"] for row in rows if row["outcome_status"] == "closed_and_settled"),
            "top_stock_concentration": _concentration(rows), "stocks": rows})
    return {"version": "original-channel-concentration-v1", "status": "data_limited",
        "research_only": True, "strict_point_in_time": False,
        "attribution_coverage_status": "complete" if not issues and not counts["rounded_fee_fallback_rows"] else "data_limited",
        "initial_capital": capital, "coverage": counts, "issues": issues, "by_channel": by_channel,
        "covered_realized_net_profit": sum(row["realized_net_profit"] for row in stocks),
        "counterfactual_is_tradable": False, "independent_channel_effect_established": False,
        "method": "Original executed lot allocations; left-to-right transfers retain left origin. Sales include allocated entry fees and exit costs; settled dividends are separate. Open-lot fees remain in unrealized PnL.",
        "limitations": ["Historical descriptive attribution does not repair financial revisions, industry snapshot timing or post-selection bias.",
            "Top1/3 removal subtracts observed stock profits; it does not rerun cash, costs, order competition or positions and is not a tradable return.",
            "Shared-capital channel contribution is not the independent causal benefit of enabling that channel.",
            "Open shares, pending listings and unpaid dividends are incomplete outcomes; concentration includes only their already realized portion.",
            "Missing source allocations are excluded and disclosed; covered totals can be partial. Stock groups are not independent campaigns."]}


def _digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _verify_legacy_comparison(receipt, events, summary, report, events_sha256):
    """Legacy receipt binds all events and audited aggregates, not summary bytes."""
    recorded = receipt.get("determinism", {}).get("events", {})
    if recorded.get("identical") is not True or recorded.get("sha256") != [events_sha256, events_sha256]:
        raise ValueError("Comparison receipt does not bind the entire events CSV")
    repaired = receipt.get("repaired", {})
    for summary_key, receipt_key in (("frozen_run", "run_manifest"), ("accounting_audit", "accounting_audit"),
                                     ("capital_risk_attribution", "capital_risk_attribution")):
        if not isinstance(summary.get(summary_key), dict) or summary[summary_key] != repaired.get(receipt_key):
            raise ValueError(f"Summary {summary_key} differs from comparison receipt")
    manifest = summary["frozen_run"]
    if not receipt.get("input_bundle_sha256") or manifest.get("input_bundle_sha256") != receipt["input_bundle_sha256"]:
        raise ValueError("Comparison input bundle differs from summary")
    parameters = manifest.get("engine_parameters", {})
    if parameters.get("initial_capital") != summary["initial_capital"] or parameters.get("end_date") != summary["end_date"]:
        raise ValueError("Summary capital or cutoff differs from comparison run")
    trades = [row for row in events if _number(row.get("execution_quantity"), "execution_quantity", optional=True)]
    ledger = summary.get("trade_ledger")
    if not isinstance(ledger, list) or len(ledger) != len(trades):
        raise ValueError("Summary trade ledger differs from events")
    numeric = ("event_sequence", "execution_quantity", "execution_price", "transaction_cost_amount_raw",
               "profit_loss_amount_raw", "cash_change_amount_raw", "cash_before", "cash_after")
    for event, trade in zip(trades, ledger):
        if any(event.get(key) != trade.get(key) for key in ("date", "code", "execution_phase")) or (
                _structured(event.get("origin_allocations")) != trade.get("origin_allocations")):
            raise ValueError("Summary trade ledger identity or origin differs from events")
        for key in numeric:
            if not math.isclose(_number(event.get(key), key), _number(trade.get(key), key), rel_tol=1e-12, abs_tol=1e-8):
                raise ValueError(f"Summary trade ledger {key} differs from events")
    if report["attribution_coverage_status"] != "complete":
        raise ValueError("Comparison attribution cannot be verified with incomplete coverage")
    recorded_origins = summary["capital_risk_attribution"].get("by_origin", [])
    origins = {row.get("origin_account"): row for row in recorded_origins}
    if len(origins) != len(recorded_origins) or set(origins) - {"left", "right"}:
        raise ValueError("Invalid comparison origin attribution")
    fields = {"realized_price_net_profit": "realized_net_pnl", "unrealized_net_profit": "unrealized_net_pnl",
              "settled_dividends": "dividend_cash", "dividend_receivable": "dividend_receivable", "fees_paid": "transaction_cost"}
    for channel in report["by_channel"]:
        recorded_origin = origins.get(channel["origin_channel"], {})
        for current, archived in fields.items():
            expected = _number(recorded_origin.get(archived), archived) if recorded_origin else 0.
            if not math.isclose(channel[current], expected, rel_tol=0, abs_tol=.005001):
                raise ValueError(f"Recomputed origin {archived} differs from comparison receipt")
        total = channel["realized_net_profit"] + channel["unrealized_net_profit"] + channel["dividend_receivable"]
        if recorded_origin and not math.isclose(total, _number(recorded_origin.get("total_net_pnl"), "total_net_pnl"), rel_tol=0, abs_tol=.005001):
            raise ValueError("Recomputed total origin profit differs from comparison receipt")
    # Transfers do not change origin. Corporate-action quantity deltas carry the
    # pre-action allocation; cost and entry fees are conserved across bonus lots.
    balances = defaultdict(lambda: [0., 0., 0.])
    for event in events:
        quantity = _number(event.get("execution_quantity"), "execution_quantity", optional=True)
        delta = _number(event.get("corporate_action_quantity_delta"), "corporate_action_quantity_delta", optional=True)
        if not quantity and not delta:
            continue
        allocations = _allocations(event.get("origin_allocations"))
        allocated = sum(row["quantity"] for row in allocations)
        sign = 1 if quantity > 0 else -1 if quantity < 0 else 0
        for row in allocations:
            balance = balances[(row["channel"], event["code"])]
            balance[0] += sign * row["quantity"] + delta * row["quantity"] / allocated
            balance[1] += sign * row["cost"]
            balance[2] += sign * row["entry_fee"]
    terminal = defaultdict(lambda: [0., 0., 0.])
    for position in summary["final_positions"]:
        for lot in position.get("left_batches", []) + position.get("batches", []):
            row = _allocations([lot["origin_allocation"]])[0]
            balance = terminal[(row["channel"], position["code"])]
            for index, key in enumerate(("quantity", "cost", "entry_fee")):
                balance[index] += row[key]
    for key in balances.keys() | terminal.keys():
        if any(not math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-5)
               for left, right in zip(balances[key], terminal[key])):
            raise ValueError("Terminal origin inventory differs from complete events")


def report_from_files(events_path, summary_path, *, source_receipt=None):
    """Use explicit {path, sha256} evidence; fresh hashes alone prove no audit.

    New receipts use output_sha256 keyed by exact source basenames. Historical
    comparison receipts bind the complete events CSV and selected audited
    summary fields; they never certify an absent historical summary byte hash.
    """
    paths = {"events": Path(events_path).resolve(), "summary": Path(summary_path).resolve()}
    hashes = {key: _digest(path) for key, path in paths.items()}
    receipt, receipt_metadata = None, None
    if source_receipt is not None:
        if not isinstance(source_receipt, dict) or not source_receipt.get("path") or not source_receipt.get("sha256"):
            raise ValueError("source_receipt requires an explicit path and SHA256")
        receipt_path = Path(source_receipt["path"]).resolve()
        receipt_sha = str(source_receipt["sha256"]).lower()
        if _digest(receipt_path) != receipt_sha:
            raise ValueError("Source receipt SHA256 mismatch")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if not isinstance(receipt, dict):
            raise ValueError("Source receipt must be an object")
        receipt_metadata = {"path": str(receipt_path), "sha256": receipt_sha}
    with paths["events"].open(encoding="utf-8-sig", newline="") as stream:
        events = list(csv.DictReader(stream))
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    audit = summary.get("accounting_audit", {})
    if audit.get("status") not in {"passed", "data_limited"} or audit.get("error_count") != 0 or audit.get("errors") != []:
        raise ValueError("Channel report requires an existing zero-error independent accounting receipt")
    if summary.get("event_count") != len(events):
        raise ValueError("Summary and event CSV row counts differ")
    try:
        end = date.fromisoformat(summary["end_date"])
        event_dates = [date.fromisoformat(row["date"]) for row in events]
    except (KeyError, ValueError, TypeError) as exc:
        raise ValueError("Events or summary lack a valid observation cutoff") from exc
    if event_dates != sorted(event_dates) or any(day > end for day in event_dates):
        raise ValueError("Events exceed or lack the declared observation cutoff")
    report = build_channel_contribution(events, summary.get("final_positions"), initial_capital=summary["initial_capital"])
    binding = "unverified_source_pair"
    if receipt is not None:
        if "output_sha256" in receipt:
            output_hashes = receipt["output_sha256"]
            if not isinstance(output_hashes, dict) or any(output_hashes.get(path.name) != hashes[key] for key, path in paths.items()):
                raise ValueError("Source receipt output hashes do not bind both events and summary")
            binding = "verified_output_hashes"
        else:
            _verify_legacy_comparison(receipt, events, summary, report, hashes["events"])
            binding = "verified_legacy_comparison_scope"
            report["limitations"].append("Legacy comparison has no summary byte hash: it binds complete events, run/audit/attribution fields, trade economics and reconstructed terminal origin quantities/costs/fees. Terminal valuations reconcile by channel; ancillary terminal fields are not independently authenticated.")
        if _digest(receipt_path) != receipt_sha:
            raise ValueError("Source receipt changed while reading")
    else:
        report["limitations"].append("No external source receipt: observed file hashes and embedded audit do not establish that the events and summary came from the same audited run.")
    if hashes != {key: _digest(path) for key, path in paths.items()}:
        raise ValueError("Channel report source changed while reading")
    report.update(source_files={key: {"path": str(paths[key]), "sha256": value} for key, value in hashes.items()},
                  observation_end_date=summary["end_date"], source_binding_status=binding,
                  source_receipt=receipt_metadata, declared_source_accounting_status=audit["status"],
                  source_data_limitations=summary.get("data_limitations", audit.get("data_limitations", [])))
    return report


def render_markdown(report):
    lines = ["# 左右来源贡献与股票集中度", "", "研究诊断；仍属 data_limited。左转右按最初买入来源归因。", "",
             "| 原始通道 | 已实现净利润 | 其中到账股息 | 未实现净利润 | 应收股息 | 未结算股票数 |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in report["by_channel"]:
        lines.append(f"| {row['origin_channel']} | {row['realized_net_profit']:.2f} | {row['settled_dividends']:.2f} | {row['unrealized_net_profit']:.2f} | {row['dividend_receivable']:.2f} | {row['unsettled_stock_count']} |")
    lines += ["", "已实现净利润含已售股份分摊买入费与卖出成本；未卖出买入费保留在浮盈，费用不重复扣除。",
              "", "| 通道 | Top数量 | 股票 | 已实现利润贡献 | 占通道净盈利 | 静态扣除后利润 |", "|---|---:|---|---:|---:|---:|"]
    for row in report["by_channel"]:
        for top in row["top_stock_concentration"]:
            share = top["share_of_net_realized_profit"]
            display = f"{share:.2%}" if share is not None else "不可计算"
            lines.append(f"| {row['origin_channel']} | {top['count']} | {', '.join(top['codes']) or '无正盈利股票'} | {top['profit']:.2f} | {display} | {top['remaining_static_realized_profit']:.2f} |")
    lines += ["", "Top1/3为事后静态利润扣除，不是可交易反事实收益；没有重跑现金、订单竞争、仓位或成本。共享资金归因不能证明左侧的独立因果贡献。",
              "", "| 未完整结算股票 | 原始通道 | 状态 | 剩余股数 | 未上市股数 | 应收股息 |", "|---|---|---|---:|---:|---:|"]
    for channel in report["by_channel"]:
        for row in channel["stocks"]:
            if row["outcome_status"] != "closed_and_settled":
                lines.append(f"| {row['code']} | {row['origin_channel']} | {row['outcome_status']} | {row['open_quantity']:g} | {row['pending_listing_quantity']:g} | {row['dividend_receivable']:.2f} |")
    lines += ["", f"归因覆盖：{report['attribution_coverage_status']}；排除成交/现金记录 {report['coverage']['excluded_cash_or_trade_rows']} 条；排除终仓记录 {report['coverage']['excluded_final_position_rows']} 条。",
              "历史财务修订、行业快照时点和事后选样限制仍保留；缺失来源不视为零收益。", ""]
    if "source_binding_status" in report:
        lines += [f"源文件绑定：{report['source_binding_status']}。嵌入审计状态仅为来源声明；缺外部收据不能证明CSV与摘要属于同次已审计运行。",
                  "旧comparison仅校验完整事件、审计/运行/归因字段、成交经济字段及重建终仓来源股数/成本/买入费；终仓估值按通道核对，不认证未记录的summary字节哈希及终仓辅助字段。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--source-receipt", help="External comparison or output_sha256 receipt JSON")
    parser.add_argument("--source-receipt-sha256", help="Previously recorded SHA256 of that receipt")
    args = parser.parse_args()
    if bool(args.source_receipt) != bool(args.source_receipt_sha256):
        parser.error("--source-receipt and --source-receipt-sha256 must be provided together")
    receipt = {"path": args.source_receipt, "sha256": args.source_receipt_sha256} if args.source_receipt else None
    report = report_from_files(args.events, args.summary, source_receipt=receipt)
    output = Path(args.output_directory)
    output.mkdir(parents=True, exist_ok=False)
    (output / "channel_contribution.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    (output / "channel_contribution.md").write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"status": report["status"], "attribution_coverage_status": report["attribution_coverage_status"],
                      "output_directory": str(output.resolve())}))


if __name__ == "__main__":
    main()
