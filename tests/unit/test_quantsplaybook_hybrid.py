import json

import pandas as pd

from apps.quantsplaybook_hybrid import (
    build_hybrid_candidates,
    merge_hybrid_rows,
)
from stock_research.strategies.fundamental_selection import (
    VALUE_INDUSTRY_RULE_VERSION,
)
from stock_research.strategies.historical_candidates import SNAPSHOT_VERSION


def _right(code="A"):
    return {
        "code": code,
        "name": code,
        "candidate_score": 99.0,
        "playbook_factor": "playbook_low_corr",
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
    common = {
        "version": SNAPSHOT_VERSION,
        "value_industry_rule_version": VALUE_INDUSTRY_RULE_VERSION,
        "financial_point_in_time": True,
        "snapshots": [{
            "date": "2024-01-02",
            "file": "candidates_2024-01-02.csv",
            "candidate_count": 1,
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
    assert manifest["snapshot_count"] == 1
    assert manifest["total_left_candidate_rows"] == 1
    assert manifest["total_right_candidate_rows"] == 1
    assert manifest["total_overlap_candidate_rows"] == 0
    assert manifest["right_lane"]["manifest_sha256"]
    assert manifest["left_lane"]["manifest_sha256"]
