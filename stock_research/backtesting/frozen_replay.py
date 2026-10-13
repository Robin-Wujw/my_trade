"""Freeze one consistent input snapshot and audit an offline portfolio replay."""
from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack
from datetime import datetime, timezone
import importlib.metadata
import importlib.util
import inspect
import json
import math
from pathlib import Path
import pickle
import platform
import shutil
import subprocess
import sys
from unittest.mock import patch

import pandas as pd

from stock_research.backtesting import workflow as app
from stock_research.core.paths import PATHS
from stock_research.reporting.portfolio_accounting_audit import audit_portfolio_accounting
from stock_research.reporting.backtest_trade_report import (
    apply_stock_basic_display_names, build_readable_trade_frame, render_trade_report_markdown,
)
from stock_research.storage import Database, TushareRepository
from stock_research.backtesting.engine import run_portfolio_backtest


ROOT = Path(__file__).resolve().parents[2]
DATA_ARGUMENTS = {"price_frames", "candidate_snapshots", "formula_phases", "trade_plans",
                  "corporate_actions", "security_end_dates"}


def _json(path, value):
    def clean(item):
        if isinstance(item, dict):
            return {key: clean(value) for key, value in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(value) for value in item]
        if isinstance(item, float) and not math.isfinite(item):
            return None
        return item
    Path(path).write_text(json.dumps(clean(value), ensure_ascii=False, indent=2,
                                    default=str, allow_nan=False), encoding="utf-8")


class _SharedReadConnection:
    def __init__(self, connection):
        self.connection = connection

    def __getattr__(self, name):
        return getattr(self.connection, name)

    def close(self):
        pass  # The transaction owner closes this after all repositories finish.

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class _SnapshotDatabase:
    def __init__(self, connection):
        self.connection = _SharedReadConnection(connection)

    def connect(self, *, read_only=False):
        if not read_only:
            raise RuntimeError("Frozen inputs require read-only database access")
        return self.connection


@contextmanager
def _readonly_snapshot(path):
    with Database(path).connect(read_only=True) as connection:
        connection.execute("BEGIN")
        connection.execute("SELECT name FROM sqlite_master LIMIT 1").fetchall()
        yield _SnapshotDatabase(connection)


def _dependency_versions():
    return {"python": platform.python_version(), **{
        name: importlib.metadata.version(name) for name in ("pandas", "numpy", "scipy")}}


@contextmanager
def _offline():
    def denied(*args, **kwargs):
        raise RuntimeError("Frozen replay cannot access the database or network")
    with ExitStack() as stack:
        for target in ("sqlite3.connect", "socket.create_connection", "socket.socket.connect",
                       "socket.socket.connect_ex"):
            stack.enter_context(patch(target, denied))
        yield


def _validate_prepared(frames):
    for code, frame in frames.items():
        factor = pd.to_numeric(frame["qfq_to_raw_factor"], errors="coerce")
        if not factor.gt(0).all():
            raise ValueError(f"Invalid prepared price factors: {code}")
        for column in ("open", "high", "low", "close"):
            raw = pd.to_numeric(frame[f"raw_{column}"], errors="coerce")
            signal = pd.to_numeric(frame[column], errors="coerce")
            difference = (raw - signal * factor).abs()
            valid = (raw.isna() & signal.isna()) | difference.le(1e-8 * raw.abs().clip(lower=1))
            if not valid.all():
                raise ValueError(f"Prepared raw/qfq price units disagree: {code}:{column}")


def _code_hashes():
    paths = sorted((ROOT / "stock_research").rglob("*.py")) + [
        ROOT / "apps/portfolio_backtest.py", ROOT / "apps/portfolio_frozen_replay.py",
        ROOT / "apps/__init__.py", ROOT / "pyproject.toml",
    ]
    return {str(p.relative_to(ROOT)): app.file_sha256(p) for p in paths}


def _engine_parameters(config):
    parameters = inspect.signature(run_portfolio_backtest).parameters
    configured = config["engine_parameters"]
    unknown = set(configured) - (set(parameters) - DATA_ARGUMENTS)
    if unknown:
        raise ValueError(f"Unknown engine parameters: {sorted(unknown)}")
    resolved = {name: p.default for name, p in parameters.items()
                if name not in DATA_ARGUMENTS and p.default is not inspect.Parameter.empty}
    resolved.update(configured)
    for name, p in parameters.items():
        if name not in DATA_ARGUMENTS and p.default is inspect.Parameter.empty and name not in resolved:
            raise ValueError(f"Missing required engine parameter: {name}")
    return resolved


def freeze(config_path, bundle_directory):
    config_path = Path(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    params = _engine_parameters(config)
    start, end = params["requested_start"], params["end_date"]
    inputs = config["inputs"]
    candidate_dir = Path(inputs["candidate_directory"])
    formula_path = Path(inputs["formula_history"])
    plans_path = Path(inputs["trade_plans"])
    price_dir = Path(inputs["price_kline_directory"])
    manifest_path = candidate_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if config.get("research_only") is not True:
        raise ValueError("This frozen historical replay must be labelled research_only")
    for field, path in (("candidate_manifest_sha256", manifest_path),
                        ("formula_history_sha256", formula_path)):
        if app.file_sha256(path) != inputs[field]:
            raise ValueError(f"Frozen input changed: {field}")
    app.validate_candidate_manifest_financial_point_in_time(candidate_dir, allow_unsafe_financial=False)
    app.validate_candidate_manifest_industry_point_in_time(candidate_dir, allow_unsafe_industry=False)
    sources = [config_path, manifest_path, formula_path, plans_path]
    sources += [candidate_dir / row["file"] for row in manifest["snapshots"]]
    source_hashes = {str(p): app.file_sha256(p) for p in sources}
    snapshots = app.load_candidate_snapshots(candidate_dir, start, end)
    codes = sorted({str(row["code"]) for rows in snapshots.values() for row in rows})
    for code in codes:
        stem = code.replace(".", "_")
        path = price_dir / f"{stem}.csv"
        if not path.exists():
            path = price_dir / f"{stem}.pkl"
        source_hashes[str(path)] = app.file_sha256(path)
        if price_dir.name == "akshare_raw":
            qfq = price_dir.parent / "akshare" / f"{stem}.csv"
            if qfq.exists():
                source_hashes[str(qfq)] = app.file_sha256(qfq)
    print(f"Loading frozen prices for {len(codes)} symbols", flush=True)
    with _readonly_snapshot(PATHS.database) as database:
        frames = app.load_price_frames(
            codes, price_dir, start_date=(pd.Timestamp(start) - pd.Timedelta(days=700)).strftime("%Y-%m-%d"),
            end_date=end, prefer_database=False, database=database,
        )
        repository = TushareRepository(database)
        corporate_actions = repository.load_dividend_actions(codes, start_date=start, end_date=end)
        stock_basic = repository.load_stock_basic_universe()
        suspensions = repository.load_dataset_for_codes("suspend_d", codes, start_date=start, end_date=end)
        exchange_calendar = repository.load_dataset("trade_cal")
    security_ends = {str(row["code"]): pd.Timestamp(row["delist_date"]).normalize()
                     for _, row in stock_basic.iterrows()
                     if str(row["code"]) in codes and pd.notna(row["delist_date"])}
    trade_calendar = None
    if not exchange_calendar.empty and {"cal_date", "is_open"}.issubset(exchange_calendar):
        trade_calendar = pd.to_datetime(exchange_calendar.loc[
            pd.to_numeric(exchange_calendar["is_open"], errors="coerce").eq(1), "cal_date"], errors="coerce").dropna()
    suspension_dates = {} if suspensions.empty else {
        str(code): {pd.Timestamp(value).normalize() for value in group["date"].dropna()}
        for code, group in suspensions.groupby("code")
    }
    coverage = app.validate_price_frame_coverage(
        frames, codes, start, end, code_start_dates=app.first_candidate_dates(snapshots),
        security_end_dates=security_ends, security_suspension_dates=suspension_dates,
        trade_calendar=trade_calendar, require_internal_complete=False,
    )
    limits = app.validate_daily_price_limit_coverage(frames, start, end)
    formula = pd.read_csv(formula_path)
    app.validate_backtest_input_coverage(snapshots, formula, start, end, candidate_directory=candidate_dir,
                                       allow_unsafe_financial=False, allow_unsafe_industry=False)
    phases = {str(row["date"]): {
        "phase": str(row["phase"]), "window_down_streak": int(row.get("window_down_streak") or 0),
        "window_up_streak": int(row.get("window_up_streak") or 0),
        "window_unique_count": None if pd.isna(row.get("window_unique_count")) else int(row["window_unique_count"]),
    } for _, row in formula.iterrows()}
    required = {"date", "open", "high", "low", "close", "volume", "raw_open", "raw_high", "raw_low", "raw_close",
                "qfq_to_raw_factor", "ma5", "ma10", "ma20", "ma60", "ma120", "kd_divergence", "rsi_divergence",
                "wr_divergence", "bearish_divergence_count"}
    if params.get("price_frames_prepared"):
        incomplete = {code: frame for code, frame in frames.items() if not required.issubset(frame.columns)}
        if incomplete:
            frames.update(app.prepare_portfolio_price_frames(incomplete))
        if any(not required.issubset(frame.columns) for frame in frames.values()):
            raise ValueError("Incomplete prepared frames")
        _validate_prepared(frames)
    for path, expected in source_hashes.items():
        if app.file_sha256(path) != expected:
            raise ValueError(f"Input changed while freezing: {path}")
    output = Path(bundle_directory)
    output.mkdir(parents=True, exist_ok=False)
    payload = {"engine_kwargs": {**params, "price_frames": frames, "candidate_snapshots": snapshots,
                                 "formula_phases": phases, "trade_plans": app.load_trade_plans(plans_path),
                                 "corporate_actions": corporate_actions, "security_end_dates": security_ends},
               "stock_basic_display_only": stock_basic,
               "source_suspensions": suspensions, "source_trade_calendar": exchange_calendar}
    bundle = output / "inputs.pkl"
    with bundle.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    frozen = {
        "version": "portfolio-frozen-inputs-v2", "research_only": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "database_read_contract": "one_read_only_transaction_for_all_database_inputs",
        "dependencies": _dependency_versions(),
        "parameter_provenance": config.get("parameter_provenance", {}),
        "historical_reconstruction_limitations": config.get("historical_reconstruction_limitations", []),
        "input_bundle_sha256": app.file_sha256(bundle), "source_file_sha256": source_hashes,
        "engine_parameters": params, "code_sha256_at_freeze": _code_hashes(),
        "price_coverage": coverage, "daily_limit_coverage": limits,
        "corporate_action_symbols": len(corporate_actions),
        "corporate_action_events": sum(len(rows) for rows in corporate_actions.values()),
        "note": "只读一致快照固定实际输入；回放禁用数据库和网络。财报修订版本与原始历史环境的恢复范围另行披露。",
    }
    for relative, expected in frozen["code_sha256_at_freeze"].items():
        target = output / "code" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
        if app.file_sha256(target) != expected:
            raise ValueError(f"Code changed while freezing: {relative}")
    _json(output / "manifest.json", frozen)
    print(f"Frozen input bundle: {output}", flush=True)
    return frozen


def _reference_hashes(directory):
    root = Path(directory)
    manifest = json.loads((root / "source_manifest.json").read_text(encoding="utf-8"))
    for relative, expected in manifest["sha256"].items():
        if app.file_sha256(root / relative) != expected:
            raise ValueError(f"Reference source changed: {relative}")
    return manifest["sha256"]


def _run_reference(directory, bundle, output, engine_path):
    worker = r'''
import importlib.util, pickle, socket, sqlite3, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
def denied(*args, **kwargs):
    raise RuntimeError("Frozen reference cannot access the database or network")
sqlite3.connect = denied
socket.create_connection = denied
socket.socket.connect = denied
socket.socket.connect_ex = denied
spec = importlib.util.spec_from_file_location("archived_portfolio_engine", sys.argv[4])
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
with Path(sys.argv[2]).open("rb") as handle:
    payload = pickle.load(handle)
result = module.run_portfolio_backtest(**payload["engine_kwargs"])
with Path(sys.argv[3]).open("wb") as handle:
    pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
'''
    result_path = Path(output) / "reference_result.pkl"
    subprocess.run([sys.executable, "-c", worker, str(Path(directory).resolve()),
                    str(Path(bundle).resolve()), str(result_path.resolve()), str(Path(engine_path).resolve())],
                   cwd=directory, check=True)
    with result_path.open("rb") as handle:
        return pickle.load(handle)


def replay(bundle_directory, output_directory, *, reference_engine=None, reference_code_directory=None):
    directory = Path(bundle_directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    bundle = directory / "inputs.pkl"
    if app.file_sha256(bundle) != manifest["input_bundle_sha256"]:
        raise ValueError("Frozen input bundle hash mismatch")
    if manifest.get("version") != "portfolio-frozen-inputs-v2":
        raise ValueError("Frozen replay requires the complete v2 contract")
    if _code_hashes() != manifest["code_sha256_at_freeze"]:
        raise ValueError("Code hash differs from the frozen source; run using the bundled code")
    if _dependency_versions() != manifest["dependencies"]:
        raise ValueError("Dependency versions differ from the frozen environment")
    # Only locally created bundles are supported; pickle is not an untrusted interchange format.
    with bundle.open("rb") as handle:
        payload = pickle.load(handle)
    effective_parameters = {key: value for key, value in payload["engine_kwargs"].items() if key not in DATA_ARGUMENTS}
    if effective_parameters != manifest["engine_parameters"]:
        raise ValueError("Effective parameters disagree with the frozen manifest")
    reference_hashes = _reference_hashes(reference_code_directory) if reference_code_directory else None
    engine = run_portfolio_backtest
    if reference_engine and not reference_code_directory:
        spec = importlib.util.spec_from_file_location("portfolio_reference_engine", reference_engine)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        engine = module.run_portfolio_backtest
    engine_path = (Path(reference_engine) if reference_engine else
                   Path(reference_code_directory) / "stock_research/strategies/portfolio_backtest.py") if reference_code_directory else Path(inspect.getfile(engine))
    code_hashes = _code_hashes()
    engine_hash = app.file_sha256(engine_path)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=False)
    run_manifest = {"input_bundle_sha256": manifest["input_bundle_sha256"], "engine_parameters": manifest["engine_parameters"],
                    "engine_file": str(engine_path), "engine_sha256": engine_hash, "code_sha256": code_hashes,
                    "command": sys.argv, "status": "running", "research_only": True,
                    "acceptance_status": "pending", "strict_point_in_time": False,
                    "historical_reconstruction_limitations": manifest.get("historical_reconstruction_limitations", []),
                    "reference_code_sha256": reference_hashes,
                    "reference_scope": ("archived_pre_audit_full_source" if reference_code_directory else
                                        "single_engine_with_frozen_current_dependencies" if reference_engine else "frozen_full_source")}
    _json(output / "run_manifest.json", run_manifest)
    try:
        if reference_code_directory:
            result = _run_reference(reference_code_directory, bundle, output, engine_path)
            if _reference_hashes(reference_code_directory) != reference_hashes:
                raise ValueError("Reference source changed during replay")
        else:
            with _offline():
                result = engine(**payload["engine_kwargs"])
        parameters = payload["engine_kwargs"]
        audit = audit_portfolio_accounting(result, parameters["price_frames"], parameters["corporate_actions"], parameters)
        _json(output / "accounting_audit.json", audit)
        data_limitations = []
        if manifest["price_coverage"].get("internal_missing_bar_count", 0):
            data_limitations.append("unexplained_internal_price_gaps")
        if not manifest["price_coverage"].get("independent_trade_calendar"):
            data_limitations.append("independent_trade_calendar_missing")
        if not manifest["daily_limit_coverage"]["complete"]:
            data_limitations.append("exact_price_limits_incomplete")
        data_limitations.extend(audit["data_limitations"])
        research_only_reasons = list(manifest.get("historical_reconstruction_limitations") or [])
        run_manifest["accounting_status"] = audit["status"]
        run_manifest["data_limitations"] = data_limitations
        run_manifest["research_only_reasons"] = research_only_reasons
        run_manifest["acceptance_status"] = (
            "failed" if audit["status"] == "failed" else
            "data_limited" if data_limitations else
            "research_only" if manifest.get("research_only") or research_only_reasons else
            "passed"
        )
        result["accounting_audit"] = audit
        if app.file_sha256(engine_path) != engine_hash or _code_hashes() != code_hashes:
            raise ValueError("Source code changed during replay; result is not frozen")
        apply_stock_basic_display_names(result, payload["stock_basic_display_only"])
        result["research_only"] = True
        result["strict_point_in_time"] = False
        result["acceptance_status"] = run_manifest["acceptance_status"]
        result["data_limitations"] = data_limitations
        result["research_only_reasons"] = research_only_reasons
        result["frozen_run"] = run_manifest
        result["input_fingerprints"] = {"bundle_sha256": manifest["input_bundle_sha256"]}
        result["daily_limit_coverage"] = manifest["daily_limit_coverage"]
        stem = f"portfolio_{manifest['engine_parameters']['requested_start']}_{manifest['engine_parameters']['end_date']}"
        for key, suffix in (("events", "events"), ("trade_ledger", "trades"), ("equity_curve", "equity")):
            pd.DataFrame(result[key]).to_csv(output / f"{stem}_{suffix}.csv", index=False, encoding="utf-8-sig")
        build_readable_trade_frame(result["trade_ledger"]).to_csv(output / f"{stem}_买卖流水.csv", index=False, encoding="utf-8-sig")
        (output / f"{stem}_买卖报告.md").write_text(render_trade_report_markdown(result), encoding="utf-8")
        run_manifest["status"] = "finished"
        _json(output / f"{stem}_summary.json", {k: v for k, v in result.items() if k not in {"events", "equity_curve"}})
        print(json.dumps({"status": run_manifest["status"], "acceptance_status": run_manifest["acceptance_status"],
                          "accounting_status": audit["status"], "accounting_errors": audit["error_count"],
                          **{k: result.get(k) for k in ("final_return_pct", "maximum_drawdown_pct", "trade_summary")}},
                         ensure_ascii=False, indent=2), flush=True)
        if audit["status"] == "failed" and not (reference_engine or reference_code_directory):
            raise ValueError("Independent accounting audit failed; diagnostic results retained")
        return run_manifest
    except Exception as exc:
        run_manifest.update(acceptance_status="failed", error=str(exc))
        if run_manifest["status"] != "finished":
            run_manifest["status"] = "failed"
        raise
    finally:
        _json(output / "run_manifest.json", run_manifest)


def main(argv=None):
    parser = argparse.ArgumentParser(description="冻结本地研究输入并离线复跑组合基线")
    sub = parser.add_subparsers(dest="action", required=True)
    create = sub.add_parser("freeze")
    create.add_argument("--config", required=True)
    create.add_argument("--bundle-directory", required=True)
    run = sub.add_parser("replay")
    run.add_argument("--bundle-directory", required=True)
    run.add_argument("--output-directory", required=True)
    run.add_argument("--reference-engine", help="仅用于诊断的单文件旧引擎，共用冻结的当前依赖")
    run.add_argument("--reference-code-directory", help="完整修复前源码目录，须包含 source_manifest.json；由隔离子进程加载")
    args = parser.parse_args(argv)
    if args.action == "freeze":
        freeze(args.config, args.bundle_directory)
    else:
        replay(args.bundle_directory, args.output_directory, reference_engine=args.reference_engine,
               reference_code_directory=args.reference_code_directory)


if __name__ == "__main__":
    main()
