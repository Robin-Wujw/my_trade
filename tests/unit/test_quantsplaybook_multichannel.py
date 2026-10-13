import hashlib
import json

import pandas as pd

from stock_research.research.quantsplaybook_multichannel import (
    build_multichannel_candidates,
    merge_multichannel_rows,
)
from stock_research.strategies.fundamental_selection import (
    VALUE_INDUSTRY_RULE_VERSION,
)
from stock_research.strategies.historical_candidates import SNAPSHOT_VERSION


def _right(code, score, rank, **extra):
    return {
        "code": code,
        "name": code,
        "candidate_source": "quantsplaybook_factor",
        "candidate_score": score,
        "playbook_factor_rank": rank,
        "playbook_factor_score": score / 100,
        "structure_proximity_score": 50.0,
        "avg_amount_20": 800_000_000.0,
        "return_60d": -0.20,
        "selected_for_trading": True,
        "signal_eligible": True,
        "allow_right": True,
        "allow_left": False,
        **extra,
    }


def _left(code):
    return {
        "code": code,
        "name": f"价值-{code}",
        "candidate_source": "value_model",
        "candidate_score": 80.0,
        "selected_for_trading": True,
        "signal_eligible": True,
        "allow_left": True,
        "allow_right": False,
        "value_industry_allowed": True,
        "financial_point_in_time": True,
        "industry_point_in_time": True,
        "industry": "半导体",
        "value_industry_allowlist_match": "半导体",
        "value_industry_rule_version": VALUE_INDUSTRY_RULE_VERSION,
        "quality_score": 90.0,
        "earnings_yoy": 0.30,
        "mktcap": 200.0,
        "price_to_value": 1.0,
    }


def test_multichannel_merge_keeps_independent_eligibility_and_rank_fields():
    rows = merge_multichannel_rows(
        [_right("A", 91.0, 9), _right("B", 90.0, 10)],
        [
            _right(
                "A", 99.0, 2,
                factor_horizon_channel_eligible=True,
            ),
            _left("C"),
        ],
        [_right("A", 95.0, 4), _right("D", 94.0, 5), _left("C")],
    )
    by_code = {row["code"]: row for row in rows}

    assert by_code["A"]["compressed_structure_channel_eligible"] is True
    assert by_code["A"]["factor_horizon_channel_eligible"] is True
    assert by_code["A"]["breadth_reversal_channel_eligible"] is True
    assert by_code["A"]["compressed_structure_playbook_factor_rank"] == 9
    assert by_code["A"]["factor_horizon_playbook_factor_rank"] == 2
    assert by_code["A"]["breadth_reversal_playbook_factor_rank"] == 4
    assert by_code["A"]["candidate_score"] == 99.0
    assert by_code["C"]["candidate_source"] == "value_model"
    assert by_code["C"]["allow_left"] is True
    assert by_code["C"]["allow_right"] is False
    assert by_code["D"]["selection_reason"] == "低广度反转候选"


def _manifest(
    industry_source=None,
    *,
    industry_point_in_time=False,
    candidate_count=1,
):
    manifest = {
        "version": SNAPSHOT_VERSION,
        "value_industry_rule_version": VALUE_INDUSTRY_RULE_VERSION,
        "financial_point_in_time": True,
        "strict_financial_point_in_time": True,
        "industry_point_in_time": industry_point_in_time,
        "snapshots": [{
            "date": "2023-01-03",
            "file": "candidates_2023-01-03.csv",
            "candidate_count": candidate_count,
            "financial_point_in_time": True,
            "industry_point_in_time": industry_point_in_time,
        }],
    }
    if industry_source is not None:
        manifest.update({
            "industry_source_path": str(industry_source),
            "industry_source_sha256": hashlib.sha256(
                industry_source.read_bytes()
            ).hexdigest(),
            "industry_as_of_date": "2023-01-03",
            "industry_data_source": "test:dated_membership",
            "industry_point_in_time_status": "safe",
            "industry_coverage_ratio": 1.0,
        })
    return manifest


def test_multichannel_builder_records_three_source_fingerprints(tmp_path):
    compressed = tmp_path / "compressed"
    factor = tmp_path / "factor"
    breadth = tmp_path / "breadth"
    output = tmp_path / "output"
    for directory in (compressed, factor, breadth):
        directory.mkdir()
    industry_source = tmp_path / "industry.csv"
    industry_source.write_text("code,industry\nA,半导体\n", encoding="utf-8")

    pd.DataFrame([_right("A", 91.0, 9)]).to_csv(
        compressed / "candidates_2023-01-03.csv", index=False,
    )
    pd.DataFrame([
        _right("A", 99.0, 2, factor_horizon_channel_eligible=True),
        _left("C"),
    ]).to_csv(factor / "candidates_2023-01-03.csv", index=False)
    pd.DataFrame([_right("D", 94.0, 5)]).to_csv(
        breadth / "candidates_2023-01-03.csv", index=False,
    )
    (compressed / "manifest.json").write_text(
        json.dumps(_manifest()), encoding="utf-8",
    )
    for directory, candidate_count in ((factor, 2), (breadth, 1)):
        (directory / "manifest.json").write_text(
            json.dumps(
                _manifest(
                    industry_source,
                    industry_point_in_time=True,
                    candidate_count=candidate_count,
                )
            ),
            encoding="utf-8",
        )

    build_multichannel_candidates(
        compressed_directory=compressed,
        factor_horizon_directory=factor,
        breadth_reversal_directory=breadth,
        output_directory=output,
        start_date="2023-01-01",
        end_date="2023-12-31",
    )

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    candidates = pd.read_csv(output / "candidates_2023-01-03.csv")
    assert manifest["snapshot_count"] == 1
    assert manifest["recommended_max_daily_candidates"] == 120
    assert set(manifest["channels"]) == {
        "compressed_structure", "factor_horizon", "breadth_reversal",
    }
    assert all(
        item["manifest_sha256"] for item in manifest["channels"].values()
    )
    assert set(candidates["code"]) == {"A", "C", "D"}
