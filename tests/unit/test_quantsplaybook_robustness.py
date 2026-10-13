import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from stock_research.research.quantsplaybook_robustness import (
    MATRIX_NAME,
    MANIFEST_NAME,
    RobustnessError,
    analyze_equity_curve,
    file_sha256,
    identify_positive_top_codes,
    run_robustness,
    validate_portfolio_template,
    validate_strict_candidate_manifest,
    _verify_rerun_summary,
)


def _write_equity(path: Path, rows: list[tuple[str, float]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["date", "equity"])
        writer.writeheader()
        for current_date, equity in rows:
            writer.writerow({"date": current_date, "equity": equity})


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _option_value(command: list[str], option: str) -> str:
    return command[command.index(option) + 1]


def _baseline_fixture(tmp_path: Path) -> dict:
    candidate_directory = tmp_path / "candidates"
    candidate_directory.mkdir()
    candidate_manifest = candidate_directory / "manifest.json"
    _write_json(candidate_manifest, {
        "version": "test",
        "financial_point_in_time": True,
        "strict_financial_point_in_time": True,
        "uses_financial_data": True,
        "industry_point_in_time": True,
        "research_only": False,
        "snapshots": [
            {
                "date": current_date,
                "financial_point_in_time": True,
                "industry_point_in_time": True,
            }
            for current_date in (
                "2021-01-04", "2021-12-31", "2022-12-30",
            )
        ],
    })

    formula_history = tmp_path / "formula.csv"
    formula_history.write_text("date,code\n", encoding="utf-8")

    baseline_directory = tmp_path / "baseline"
    baseline_directory.mkdir()
    stem = "portfolio_2021-01-01_2022-12-30"
    summary_path = baseline_directory / f"{stem}_summary.json"
    equity_path = baseline_directory / f"{stem}_equity.csv"
    _write_equity(equity_path, [
        ("2021-01-04", 1.0),
        ("2021-12-31", 1.10),
        ("2022-12-30", 1.32),
    ])
    fingerprints = {
        "candidate_directory": str(candidate_directory),
        "candidate_manifest": str(candidate_manifest),
        "candidate_manifest_sha256": file_sha256(candidate_manifest),
        "formula_history": str(formula_history),
        "formula_history_sha256": file_sha256(formula_history),
    }
    top_symbols = [
        {"code": "sz.000001", "name": "one", "total_pnl_amount": 300.0},
        {"code": "sh.600002", "name": "two", "total_pnl_amount": 200.0},
        {"code": "sz.000003", "name": "three", "total_pnl_amount": 100.0},
        {"code": "sh.600004", "name": "loss", "total_pnl_amount": -10.0},
    ]
    summary = {
        "requested_start": "2021-01-01",
        "actual_start": "2021-01-04",
        "end_date": "2022-12-30",
        "requested_end_date": "2022-12-30",
        "effective_end_date": "2022-12-30",
        "coverage_complete": True,
        "missing_candidate_trade_dates": [],
        "candidate_snapshot_dates": [
            "2021-01-04",
            "2021-12-31",
            "2022-12-30",
        ],
        "initial_capital": 1000.0,
        "commission_rate": 0.0003,
        "final_return_pct": 32.0,
        "maximum_drawdown_pct": 0.0,
        "trade_ledger": [],
        "final_positions": [],
        "profit_concentration_summary": {
            "top1_symbol": top_symbols[0],
            "top_symbols": top_symbols,
        },
        "input_fingerprints": fingerprints,
    }
    _write_json(summary_path, summary)
    template = [
        "--start-date",
        "2021-01-01",
        "--end-date",
        "2022-12-30",
        "--candidate-directory",
        str(candidate_directory),
        "--formula-history",
        str(formula_history),
        "--no-refresh-inputs",
        "--allow-frozen-inputs",
    ]
    return {
        "summary": summary,
        "summary_path": summary_path,
        "candidate_manifest": candidate_manifest,
        "template": template,
    }


def _successful_runner(baseline: dict):
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        output_directory = Path(_option_value(command, "--output-directory"))
        excluded_codes = _option_value(command, "--exclude-codes").split(",")
        if len(excluded_codes) == 1:
            equity_rows = [
                ("2021-01-04", 1.0),
                ("2021-12-31", 1.05),
                ("2022-12-30", 1.26),
            ]
            final_return = 26.0
            drawdown = 0.0
        else:
            equity_rows = [
                ("2021-01-04", 1.0),
                ("2021-12-31", 0.90),
                ("2022-12-30", 0.99),
            ]
            final_return = -1.0
            drawdown = -10.0

        stem = "portfolio_2021-01-01_2022-12-30"
        summary_path = output_directory / f"{stem}_summary.json"
        equity_path = output_directory / f"{stem}_equity.csv"
        _write_equity(equity_path, equity_rows)
        rerun_summary = {
            **{
                key: baseline["summary"][key]
                for key in (
                    "requested_start",
                    "actual_start",
                    "end_date",
                    "requested_end_date",
                    "effective_end_date",
                    "candidate_snapshot_dates",
                )
            },
            "coverage_complete": True,
            "missing_candidate_trade_dates": [],
            "final_return_pct": final_return,
            "maximum_drawdown_pct": drawdown,
            "commission_rate": baseline["summary"]["commission_rate"],
            "initial_capital": baseline["summary"]["initial_capital"],
            "excluded_codes": excluded_codes,
            "trade_ledger": [
                {"code": "sh.601398", "profit_loss_amount": 10.0},
            ],
            "final_positions": [],
            "input_fingerprints": baseline["summary"]["input_fingerprints"],
        }
        _write_json(summary_path, rerun_summary)
        return SimpleNamespace(returncode=0, stdout="completed\n", stderr="")

    return run, calls


def test_analyze_equity_uses_continuous_year_boundaries(tmp_path):
    path = tmp_path / "equity.csv"
    _write_equity(path, [
        ("2021-01-04", 1.0),
        ("2021-12-31", 1.10),
        ("2022-12-30", 1.32),
    ])

    result = analyze_equity_curve(
        path,
        expected_start="2021-01-04",
        expected_end="2022-12-30",
    )

    assert [row["return_pct"] for row in result["annual_returns"]] == [10.0, 20.0]
    assert result["full_period_return_pct"] == 32.0
    assert result["best_year"] == 2022
    assert result["without_best_year_compound_return_pct"] == 10.0


def test_analyze_equity_rejects_duplicate_dates(tmp_path):
    path = tmp_path / "equity.csv"
    _write_equity(path, [
        ("2021-01-04", 1.0),
        ("2021-01-04", 1.01),
        ("2022-12-30", 1.10),
    ])

    with pytest.raises(RobustnessError, match="strictly increasing"):
        analyze_equity_curve(path)


def test_analyze_equity_reports_rolling_five_year_windows(tmp_path):
    path = tmp_path / "equity.csv"
    _write_equity(path, [
        ("2020-01-02", 1.0),
        ("2021-01-02", 1.2),
        ("2022-01-02", 1.5),
        ("2023-01-02", 1.8),
        ("2024-01-02", 2.2),
        ("2025-01-02", 3.0),
        ("2026-01-02", 2.5),
    ])

    result = analyze_equity_curve(path)

    assert result["rolling_five_year_window_count"] == 2
    assert result["rolling_five_year_three_times_count"] == 1
    assert result["rolling_five_year_minimum_multiple"] == pytest.approx(
        2.5 / 1.2
    )
    assert result["rolling_five_year_maximum_multiple"] == pytest.approx(3.0)
    assert result["rolling_five_year_latest_multiple"] == pytest.approx(
        2.5 / 1.2
    )


def test_identify_positive_top_codes_requires_ordered_positive_symbols():
    summary = {
        "profit_concentration_summary": {
            "top_symbols": [
                {"code": "A", "total_pnl_amount": 3},
                {"code": "B", "total_pnl_amount": 2},
                {"code": "C", "total_pnl_amount": 1},
            ],
        },
    }

    assert [
        row["code"] for row in identify_positive_top_codes(summary)
    ] == ["A", "B", "C"]


def test_template_rejects_options_that_can_bypass_exact_exclusion(tmp_path):
    baseline = _baseline_fixture(tmp_path)

    with pytest.raises(RobustnessError, match="--codes"):
        validate_portfolio_template(
            [*baseline["template"], "--codes", "sz.000001"],
            baseline_summary=baseline["summary"],
            candidate_manifest_path=baseline["candidate_manifest"],
        )


def test_runner_executes_exact_top1_and_top3_and_writes_evidence(tmp_path):
    baseline = _baseline_fixture(tmp_path)
    command_runner, calls = _successful_runner(baseline)
    output = tmp_path / "robustness"

    result = run_robustness(
        baseline_summary_path=baseline["summary_path"],
        output_directory=output,
        portfolio_arguments=baseline["template"],
        python_executable="python-test",
        runner_command=["runner", "--test"],
        command_runner=command_runner,
    )

    assert result["status"] == "completed"
    assert result["approximate_profit_subtraction_used"] is False
    assert result["formal_oos_claim"] is False
    assert result["realized_outcome_selected_exclusions"] is True
    assert len(calls) == 2
    assert _option_value(calls[0][0], "--exclude-codes") == "sz.000001"
    assert _option_value(calls[1][0], "--exclude-codes") == (
        "sz.000001,sh.600002,sz.000003"
    )
    assert calls[0][0][:3] == ["python-test", "-m", "apps.portfolio_backtest"]

    manifest = json.loads((output / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["parent_summary"]["sha256"] == file_sha256(
        baseline["summary_path"]
    )
    assert manifest["parent_candidate_manifest"]["sha256"] == file_sha256(
        baseline["candidate_manifest"]
    )
    assert [run["return_code"] for run in manifest["runs"]] == [0, 0]
    assert all(run["command"] for run in manifest["runs"])
    assert all(run["summary_path"] for run in manifest["runs"])

    with (output / MATRIX_NAME).open("r", encoding="utf-8-sig", newline="") as handle:
        matrix = list(csv.DictReader(handle))
    assert [row["scenario"] for row in matrix] == [
        "baseline",
        "exclude_top1",
        "exclude_top3",
    ]
    assert matrix[0]["annual_return_2021_pct"] == "10.0"
    assert matrix[0]["annual_return_2022_pct"] == "20.0"
    assert matrix[0]["without_best_year_compound_return_pct"] == "10.0"
    assert matrix[1]["exact_stock_counterfactual"] == "True"
    assert matrix[1]["stock_counterfactual_method"] == (
        "full_portfolio_rerun_with_fixed_candidate_stream_exclusion"
    )
    assert matrix[1]["candidate_selection_recomputed"] == "False"


def test_runner_failure_closes_and_records_nonzero_return_code(tmp_path):
    baseline = _baseline_fixture(tmp_path)
    output = tmp_path / "failed"

    def failed_runner(command, **kwargs):
        return SimpleNamespace(returncode=7, stdout="", stderr="failed")

    with pytest.raises(RobustnessError, match="return_code=7"):
        run_robustness(
            baseline_summary_path=baseline["summary_path"],
            output_directory=output,
            portfolio_arguments=baseline["template"],
            command_runner=failed_runner,
        )

    manifest = json.loads((output / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["runs"][0]["status"] == "failed"
    assert manifest["runs"][0]["return_code"] == 7
    assert manifest["runs"][0]["command"]
    assert not (output / MATRIX_NAME).exists()


def test_runner_rejects_changed_candidate_manifest_before_any_rerun(tmp_path):
    baseline = _baseline_fixture(tmp_path)
    baseline["candidate_manifest"].write_text("changed", encoding="utf-8")
    calls = []

    def should_not_run(command, **kwargs):
        calls.append(command)
        raise AssertionError("command runner must not be called")

    output = tmp_path / "hash-failure"
    with pytest.raises(RobustnessError, match="changed after"):
        run_robustness(
            baseline_summary_path=baseline["summary_path"],
            output_directory=output,
            portfolio_arguments=baseline["template"],
            command_runner=should_not_run,
        )

    assert calls == []
    manifest = json.loads((output / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert not (output / MATRIX_NAME).exists()


@pytest.mark.parametrize(
    ("field", "baseline_value", "rerun_value"),
    (
        ("right_entry_priority_mode", "legacy_weighted", "structure_lexicographic"),
        ("right_entry_size_multiplier", 2.1, 1.0),
        ("right_entry_risk_budget_pct", 0.008, None),
        ("sell_stamp_duty_rate_before_2023_08_28", 0.001, 0.0005),
        ("sell_stamp_duty_reduction_date", "2023-08-28", "2024-01-01"),
        ("profit_tail_minimum_maximum_return_pct", 10.0, 20.0),
        ("price_amount_multiplier", 1000.0, 1.0),
        ("factor_only_min_avg_amount", 30_000_000.0, None),
        ("max_entry_signal_amount_participation", 0.01, None),
        ("max_entry_avg_amount_participation", 0.05, None),
    ),
)
def test_rerun_rejects_changed_strategy_configuration(
    tmp_path,
    field,
    baseline_value,
    rerun_value,
):
    baseline = _baseline_fixture(tmp_path)
    baseline_summary = baseline["summary"]
    baseline_summary[field] = baseline_value
    rerun_summary = {
        **baseline_summary,
        field: rerun_value,
        "excluded_codes": ["sz.000001"],
        "trade_ledger": [],
        "final_positions": [],
    }

    with pytest.raises(RobustnessError, match=field):
        _verify_rerun_summary(
            rerun_summary,
            baseline_summary=baseline_summary,
            excluded_codes=["sz.000001"],
            candidate_manifest_sha256=(
                baseline_summary["input_fingerprints"][
                    "candidate_manifest_sha256"
                ]
            ),
        )


def test_strict_manifest_rejects_research_only_candidate_input(tmp_path):
    baseline = _baseline_fixture(tmp_path)
    manifest = json.loads(
        baseline["candidate_manifest"].read_text(encoding="utf-8")
    )
    manifest["research_only"] = True
    _write_json(baseline["candidate_manifest"], manifest)

    with pytest.raises(RobustnessError, match="research_only"):
        validate_strict_candidate_manifest(baseline["candidate_manifest"])

    accepted = validate_strict_candidate_manifest(
        baseline["candidate_manifest"],
        allow_research_inputs=True,
    )
    assert accepted["research_only"] is True


def test_first_observation_loss_counts_against_initial_capital(tmp_path):
    path = tmp_path / "equity.csv"
    _write_equity(path, [("2024-01-02", .9), ("2024-12-31", 1.), ("2025-01-02", 1.1)])
    result = analyze_equity_curve(path)
    assert result["full_period_return_pct"] == 10.
    assert result["maximum_drawdown_pct"] == -10.
    assert result["annual_returns"][0]["return_pct"] == 0.


def test_zero_equity_and_a_single_observation_are_defined(tmp_path):
    path = tmp_path / "equity.csv"
    _write_equity(path, [("2024-01-02", 0.), ("2025-01-02", 0.)])
    result = analyze_equity_curve(path)
    assert result["full_period_return_pct"] == -100.
    assert result["maximum_drawdown_pct"] == -100.
    assert result["annual_returns"][1]["return_pct"] is None
    assert result["without_best_year_compound_return_pct"] is None
    _write_equity(path, [("2024-01-02", .9)])
    assert analyze_equity_curve(path)["maximum_drawdown_pct"] == -10.
