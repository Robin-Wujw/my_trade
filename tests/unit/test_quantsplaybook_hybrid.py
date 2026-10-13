import json
import hashlib

import pandas as pd
import pytest

from stock_research.backtesting.workflow import apply_candidate_exclusions
from stock_research.research.quantsplaybook_hybrid import (
    _apply_right_inverse_volatility_sizing,
    _apply_right_rank_persistence_selection,
    _load_right_industry_lookup,
    _select_right_rows,
    build_hybrid_candidates,
    build_parser,
    merge_hybrid_rows,
)
from stock_research.strategies.fundamental_selection import (
    VALUE_INDUSTRY_RULE_VERSION,
)
from stock_research.strategies.historical_candidates import SNAPSHOT_VERSION


def _right(code="A", factor="playbook_low_corr"):
    return {
        "code": code,
        "name": code,
        "candidate_score": 99.0,
        "playbook_factor": factor,
        "playbook_factor_score": 0.9,
        "selected_for_trading": True,
        "signal_eligible": True,
        "allow_right": True,
        "allow_left": False,
    }


def _left(code="B"):
    return {
        "code": code,
        "name": f"value-{code}",
        "candidate_source": "value_model",
        "candidate_score": 80.0,
        "selected_for_trading": True,
        "signal_eligible": True,
        "allow_left": True,
        "allow_right": False,
        "value_industry_allowed": True,
        "financial_point_in_time": True,
        "industry_point_in_time": True,
        "quality_score": 90.0,
        "earnings_yoy": 0.3,
        "mktcap": 200.0,
        "price_to_value": 1.0,
    }


def test_hybrid_merge_keeps_independent_lanes_and_merges_overlap():
    left_b = _left("B")
    left_b["value_falsification_reason"] = float("nan")
    rows = merge_hybrid_rows(
        [_right("A"), _right("B")],
        [left_b, _left("C")],
    )
    by_code = {row["code"]: row for row in rows}

    assert by_code["A"]["candidate_source"] == "quantsplaybook_factor"
    assert by_code["A"]["allow_right"] is True
    assert by_code["A"]["allow_left"] is False
    assert by_code["B"]["name"] == "value-B"
    assert by_code["B"]["candidate_source"] == (
        "value_model+quantsplaybook_factor"
    )
    assert by_code["B"]["allow_left"] is True
    assert by_code["B"]["allow_right"] is True
    assert by_code["B"]["candidate_score"] == 99.0
    assert by_code["C"]["candidate_source"] == "value_model"
    assert by_code["C"]["allow_left"] is True
    assert by_code["C"]["allow_right"] is False


def test_right_depth_exclusions_recall_deeper_saved_ranks():
    rows = []
    for rank, code in enumerate(("A", "B", "C", "D"), 1):
        row = _right(code)
        row["playbook_factor_rank"] = rank
        row["playbook_factor_score"] = 1.0 - rank / 10.0
        rows.append(row)

    selected = _select_right_rows(
        rows,
        top_n=2,
        exclude_codes=("A",),
    )

    assert [row["code"] for row in selected] == ["B", "C"]
    with pytest.raises(ValueError, match="right_top_n"):
        _select_right_rows(rows, top_n=0)


def test_winner_exclusion_recalls_rank_51_and_blocks_left_lane_reentry():
    rows = []
    for rank in range(1, 56):
        code = f"C{rank:02d}"
        row = _right(code)
        row["playbook_factor_rank"] = rank
        row["playbook_factor_score"] = 1.0 - rank / 100.0
        rows.append(row)

    selected = _select_right_rows(
        rows,
        top_n=50,
        exclude_codes=("C01",),
    )
    merged = merge_hybrid_rows(
        selected,
        [_left("C01")],
    )
    filtered = apply_candidate_exclusions(
        {"2024-01-02": merged},
        {"C01"},
    )["2024-01-02"]
    codes = {row["code"] for row in filtered}

    assert len(codes) == 50
    assert "C01" not in codes
    assert "C51" in codes


def test_right_industry_cap_refills_from_deeper_saved_ranks():
    industries = {
        "A": "I1", "B": "I1", "C": "I1", "D": "I2", "E": "I2",
    }
    rows = []
    for rank, code in enumerate(industries, 1):
        row = _right(code)
        row["playbook_factor_rank"] = rank
        rows.append(row)

    selected = _select_right_rows(
        rows,
        top_n=4,
        industry_lookup=lambda code, date: industries[code],
        industry_max_per_day=2,
        observation_date="2024-01-02",
    )

    assert [row["code"] for row in selected] == ["A", "B", "D", "E"]
    assert [row["right_industry_l2"] for row in selected] == [
        "I1", "I1", "I2", "I2",
    ]


def test_right_industry_cap_fails_closed_on_missing_membership():
    rows = [_right("A"), _right("B")]
    for rank, row in enumerate(rows, 1):
        row["playbook_factor_rank"] = rank

    with pytest.raises(RuntimeError, match="missing Shenwan level-2"):
        _select_right_rows(
            rows,
            top_n=2,
            industry_lookup=lambda code, date: "I1" if code == "A" else "",
            industry_max_per_day=1,
            observation_date="2024-01-02",
        )


def test_right_industry_lookup_uses_l2_membership_intervals(tmp_path):
    source = tmp_path / "industry_source_tushare_sw_member.csv"
    source.write_text(
        "code,l2_name,in_date,out_date\n"
        "sh.600000,旧行业,20200101,20221231\n"
        "sh.600000,新行业,20230101,\n",
        encoding="utf-8",
    )
    manifest = {
        "industry_point_in_time": True,
        "industry_point_in_time_status": "safe",
        "industry_data_source": "tushare/pro:sw_member",
        "industry_source_path": str(source),
        "industry_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }

    lookup = _load_right_industry_lookup(manifest, tmp_path)

    assert lookup("600000.SH", "2022-12-30") == "旧行业"
    assert lookup("sh.600000", "2023-01-03") == "新行业"


def test_inverse_volatility_sizing_uses_only_observation_date_history(tmp_path):
    dates = pd.bdate_range("2023-12-25", periods=7)
    for code, closes in {
        "sh.600000": [10.0, 10.1, 10.0, 10.1, 10.0, 10.1, 20.0],
        "sz.000001": [10.0, 11.0, 9.5, 11.5, 9.0, 12.0, 1.0],
        "sz.000002": [10.0] * 7,
    }.items():
        pd.DataFrame({"date": dates, "close": closes}).to_pickle(
            tmp_path / f"{code.replace('.', '_')}.pkl"
        )
    observation_date = dates[-2].strftime("%Y-%m-%d")
    snapshots = {
        observation_date: [
            _right("sh.600000"), _right("sz.000001"), _right("sz.000002"),
        ],
    }

    metadata = _apply_right_inverse_volatility_sizing(
        snapshots, price_directory=tmp_path, window=3,
    )

    by_code = {row["code"]: row for row in snapshots[observation_date]}
    assert by_code["sh.600000"]["right_inverse_volatility_multiplier"] > 1
    assert by_code["sz.000001"]["right_inverse_volatility_multiplier"] < 1
    assert by_code["sz.000002"]["right_inverse_volatility_multiplier"] == 1
    assert metadata["available_candidate_rows"] == 2
    assert metadata["fallback_candidate_rows"] == 1


def test_rank_persistence_uses_trailing_ranks_and_current_source_universe():
    snapshots = {}
    dates = pd.bdate_range("2024-01-02", periods=3)
    for position, date in enumerate(dates):
        rows = []
        order = ("A", "B", "C") if position < 2 else ("C", "B", "A")
        for rank, code in enumerate(order, 1):
            row = _right(code)
            row["playbook_factor_rank"] = rank
            rows.append(row)
        snapshots[date.strftime("%Y-%m-%d")] = rows

    selected, metadata = _apply_right_rank_persistence_selection(
        snapshots, top_n=2, source_top_n=3, window=3,
    )

    final_date = dates[-1].strftime("%Y-%m-%d")
    assert [row["code"] for row in selected[final_date]] == ["A", "B"]
    assert selected[final_date][0]["right_rank_persistence_20_appearances"] == 3
    assert metadata["warmup_fallback_dates"] == 2
    assert metadata["selection_changed_dates"] == 1


def test_rank_persistence_does_not_read_future_snapshot():
    snapshots = {}
    dates = pd.bdate_range("2024-01-02", periods=4)
    for date in dates:
        rows = []
        for rank, code in enumerate(("A", "B", "C"), 1):
            row = _right(code)
            row["playbook_factor_rank"] = rank
            rows.append(row)
        snapshots[date.strftime("%Y-%m-%d")] = rows

    original, _ = _apply_right_rank_persistence_selection(
        snapshots, top_n=2, source_top_n=3, window=3,
    )
    changed = {date: [dict(row) for row in rows] for date, rows in snapshots.items()}
    for row in changed[dates[-1].strftime("%Y-%m-%d")]:
        row["playbook_factor_rank"] = 4 - row["playbook_factor_rank"]
    replayed, _ = _apply_right_rank_persistence_selection(
        changed, top_n=2, source_top_n=3, window=3,
    )

    prior_date = dates[-2].strftime("%Y-%m-%d")
    assert [row["code"] for row in original[prior_date]] == ["A", "B"]
    assert [row["code"] for row in replayed[prior_date]] == ["A", "B"]


def test_hybrid_cli_right_depth_controls_are_opt_in():
    defaults = build_parser().parse_args([])
    assert defaults.right_top_n is None
    assert defaults.exclude_right_codes == ""
    assert defaults.right_industry_max_per_day is None
    assert defaults.include_right_inverse_volatility_252 is False
    assert defaults.include_right_rank_persistence_20 is False

    args = build_parser().parse_args([
        "--right-top-n", "50",
        "--exclude-right-codes", "A,B",
        "--right-industry-max-per-day", "4",
        "--include-right-inverse-volatility-252",
        "--include-right-rank-persistence-20",
    ])
    assert args.right_top_n == 50
    assert args.exclude_right_codes == "A,B"
    assert args.right_industry_max_per_day == 4
    assert args.include_right_inverse_volatility_252 is True
    assert args.include_right_rank_persistence_20 is True


def test_hybrid_builder_records_source_fingerprints(tmp_path):
    right = tmp_path / "right"
    left = tmp_path / "left"
    output = tmp_path / "output"
    right.mkdir()
    left.mkdir()
    right_frame = pd.DataFrame([_right()])
    left_frame = pd.DataFrame([_left()])
    right_frame.to_csv(right / "candidates_2024-01-02.csv", index=False)
    left_frame.to_csv(left / "candidates_2024-01-02.csv", index=False)
    industry_source = tmp_path / "industry.csv"
    industry_source.write_text("code,industry\nA,test\n", encoding="utf-8")
    common = {
        "version": SNAPSHOT_VERSION,
        "value_industry_rule_version": VALUE_INDUSTRY_RULE_VERSION,
        "financial_point_in_time": True,
        "snapshots": [{
            "date": "2024-01-02",
            "file": "candidates_2024-01-02.csv",
            "candidate_count": 1,
            "financial_point_in_time": True,
            "industry_point_in_time": True,
        }],
    }
    (right / "manifest.json").write_text(
        json.dumps(common), encoding="utf-8",
    )
    (left / "manifest.json").write_text(
        json.dumps({
            **common,
            "strict_financial_point_in_time": True,
            "industry_point_in_time": True,
            "industry_source_path": str(industry_source),
            "industry_source_sha256": hashlib.sha256(
                industry_source.read_bytes()
            ).hexdigest(),
            "industry_as_of_date": "2024-01-02",
            "industry_data_source": "test:dated_membership",
            "industry_point_in_time_status": "safe",
            "industry_coverage_ratio": 1.0,
        }),
        encoding="utf-8",
    )

    build_hybrid_candidates(
        right_directory=right,
        left_directory=left,
        output_directory=output,
        start_date="2024-01-01",
        end_date="2024-12-31",
        right_top_n=1,
        exclude_right_codes=("ZZ",),
    )

    manifest = json.loads(
        (output / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["snapshot_count"] == 1
    assert manifest["total_left_candidate_rows"] == 1
    assert manifest["total_right_candidate_rows"] == 1
    assert manifest["total_overlap_candidate_rows"] == 0
    assert manifest["right_lane"]["manifest_sha256"]
    assert manifest["right_lane"]["selected_top_n"] == 1
    assert manifest["right_lane"]["excluded_codes"] == ["ZZ"]
    assert manifest["realized_outcome_exclusions_used"] is True
    assert manifest["research_only"] is True
    assert manifest["left_lane"]["manifest_sha256"]
    assert manifest["industry_source_path"] == str(industry_source)
    assert manifest["industry_point_in_time_status"] == "safe"
    assert manifest["industry_coverage_ratio"] == 1.0


def test_hybrid_builder_records_non_default_right_factor(tmp_path):
    right = tmp_path / "right"
    left = tmp_path / "left"
    output = tmp_path / "hybrid_strategy_aligned_value"
    right.mkdir()
    left.mkdir()
    right_frame = pd.DataFrame([
        _right(factor="playbook_strategy_aligned"),
    ])
    left_frame = pd.DataFrame([_left()])
    right_frame.to_csv(right / "candidates_2024-01-02.csv", index=False)
    left_frame.to_csv(left / "candidates_2024-01-02.csv", index=False)
    industry_source = tmp_path / "industry.csv"
    industry_source.write_text("code,industry\nA,test\n", encoding="utf-8")
    common = {
        "version": SNAPSHOT_VERSION,
        "value_industry_rule_version": VALUE_INDUSTRY_RULE_VERSION,
        "financial_point_in_time": True,
        "snapshots": [{
            "date": "2024-01-02",
            "file": "candidates_2024-01-02.csv",
            "candidate_count": 1,
            "financial_point_in_time": True,
            "industry_point_in_time": True,
        }],
    }
    (right / "manifest.json").write_text(
        json.dumps({
            **common,
            "factor": "playbook_strategy_aligned",
        }),
        encoding="utf-8",
    )
    (left / "manifest.json").write_text(
        json.dumps({
            **common,
            "strict_financial_point_in_time": True,
            "industry_point_in_time": True,
            "industry_source_path": str(industry_source),
            "industry_source_sha256": hashlib.sha256(
                industry_source.read_bytes()
            ).hexdigest(),
            "industry_as_of_date": "2024-01-02",
            "industry_data_source": "test:dated_membership",
            "industry_point_in_time_status": "safe",
            "industry_coverage_ratio": 1.0,
        }),
        encoding="utf-8",
    )

    build_hybrid_candidates(
        right_directory=right,
        left_directory=left,
        output_directory=output,
        start_date="2024-01-01",
        end_date="2024-12-31",
    )

    manifest = json.loads(
        (output / "manifest.json").read_text(encoding="utf-8")
    )
    candidates = pd.read_csv(output / "candidates_2024-01-02.csv")
    assert manifest["right_lane"]["factor"] == "playbook_strategy_aligned"
    assert manifest["selection_profile"] == "hybrid_strategy_aligned_value"
    assert candidates.iloc[0]["hybrid_lane"] == "right_strategy_aligned"


def test_hybrid_marks_revision_incomplete_financial_right_lane_as_research(tmp_path):
    right = tmp_path / "right"
    left = tmp_path / "left"
    output = tmp_path / "output"
    right.mkdir()
    left.mkdir()
    pd.DataFrame([_right()]).to_csv(right / "candidates_2024-01-02.csv", index=False)
    pd.DataFrame([_left()]).to_csv(left / "candidates_2024-01-02.csv", index=False)
    industry_source = tmp_path / "industry.csv"
    industry_source.write_text("code,industry\nA,test\n", encoding="utf-8")
    common = {
        "version": SNAPSHOT_VERSION,
        "value_industry_rule_version": VALUE_INDUSTRY_RULE_VERSION,
        "financial_point_in_time": True,
        "snapshots": [{
            "date": "2024-01-02",
            "file": "candidates_2024-01-02.csv",
            "candidate_count": 1,
            "financial_point_in_time": True,
            "industry_point_in_time": True,
        }],
    }
    (right / "manifest.json").write_text(json.dumps({
        **common,
        "uses_financial_data": True,
        "strict_financial_point_in_time": False,
    }), encoding="utf-8")
    (left / "manifest.json").write_text(json.dumps({
        **common,
        "strict_financial_point_in_time": True,
        "industry_point_in_time": True,
        "industry_source_path": str(industry_source),
        "industry_source_sha256": hashlib.sha256(industry_source.read_bytes()).hexdigest(),
        "industry_as_of_date": "2024-01-02",
        "industry_data_source": "test:dated_membership",
        "industry_point_in_time_status": "safe",
        "industry_coverage_ratio": 1.0,
    }), encoding="utf-8")

    build_hybrid_candidates(
        right_directory=right,
        left_directory=left,
        output_directory=output,
        start_date="2024-01-01",
        end_date="2024-12-31",
        allow_unsafe_financial=True,
    )

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["strict_financial_point_in_time"] is False
    assert manifest["research_only"] is True


def _write_lane(directory, dates, row, *, industry_source=None):
    directory.mkdir()
    snapshots = []
    for date in dates:
        file_name = f"candidates_{date}.csv"
        pd.DataFrame([row]).to_csv(directory / file_name, index=False)
        snapshots.append({
            "date": date,
            "file": file_name,
            "candidate_count": 1,
            "financial_point_in_time": True,
            "industry_point_in_time": True,
        })
    manifest = {
        "version": SNAPSHOT_VERSION,
        "value_industry_rule_version": VALUE_INDUSTRY_RULE_VERSION,
        "financial_point_in_time": True,
        "strict_financial_point_in_time": True,
        "uses_financial_data": bool(industry_source),
        "snapshots": snapshots,
    }
    if industry_source is not None:
        manifest.update({
            "industry_point_in_time": True,
            "industry_source_path": str(industry_source),
            "industry_source_sha256": hashlib.sha256(
                industry_source.read_bytes()
            ).hexdigest(),
            "industry_as_of_date": "2024-01-02",
            "industry_data_source": "test:dated_membership",
            "industry_point_in_time_status": "safe",
            "industry_coverage_ratio": 1.0,
        })
    (directory / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8",
    )


def test_hybrid_propagates_nested_factor_research_only(tmp_path):
    source = tmp_path / "industry.csv"
    source.write_text("code,industry\nA,test\n", encoding="utf-8")
    right = tmp_path / "right"
    left = tmp_path / "left"
    output = tmp_path / "output"
    _write_lane(right, ["2024-01-02"], _right())
    _write_lane(left, ["2024-01-02"], _left(), industry_source=source)
    manifest_path = right / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["factor_metadata"] = {"research_only": True}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    build_hybrid_candidates(
        right_directory=right,
        left_directory=left,
        output_directory=output,
        start_date="2024-01-01",
        end_date="2024-12-31",
    )

    result = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert result["right_lane"]["research_only"] is True
    assert result["research_only"] is True


def test_hybrid_rejects_left_extra_lane_with_missing_trade_date(tmp_path):
    source = tmp_path / "industry.csv"
    source.write_text("code,industry\nA,test\n", encoding="utf-8")
    right = tmp_path / "right"
    left = tmp_path / "left"
    extra = tmp_path / "extra"
    _write_lane(right, ["2024-01-02", "2024-01-03"], _right())
    _write_lane(
        left, ["2024-01-02", "2024-01-03"], _left(),
        industry_source=source,
    )
    _write_lane(extra, ["2024-01-02"], _left("C"), industry_source=source)

    with pytest.raises(RuntimeError, match="left-extra candidate coverage differs"):
        build_hybrid_candidates(
            right_directory=right,
            left_directory=left,
            left_extra_directory=extra,
            output_directory=tmp_path / "output",
            start_date="2024-01-01",
            end_date="2024-12-31",
        )


def test_hybrid_rejects_left_extra_lane_without_industry_evidence(tmp_path):
    source = tmp_path / "industry.csv"
    source.write_text("code,industry\nA,test\n", encoding="utf-8")
    right = tmp_path / "right"
    left = tmp_path / "left"
    extra = tmp_path / "extra"
    _write_lane(right, ["2024-01-02"], _right())
    _write_lane(left, ["2024-01-02"], _left(), industry_source=source)
    _write_lane(extra, ["2024-01-02"], _left("C"))
    manifest_path = extra / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["uses_financial_data"] = True
    manifest["industry_point_in_time"] = True
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match="metadata_errors"):
        build_hybrid_candidates(
            right_directory=right,
            left_directory=left,
            left_extra_directory=extra,
            output_directory=tmp_path / "output",
            start_date="2024-01-01",
            end_date="2024-12-31",
        )
