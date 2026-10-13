import json

import pandas as pd
import pytest

from stock_research.research.selection_audit import (
    attach_forward_returns,
    load_manifest_candidates,
    summarize_audit,
)


def test_selection_audit_uses_manifest_and_available_bar_horizons(tmp_path):
    candidate_dir = tmp_path / "candidates"
    candidate_dir.mkdir()
    frame = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-02", "2024-01-03"],
            "code": ["sz.000001", "sz.000002", "sz.000001"],
            "candidate_source": ["factor_quant"] * 3,
            "candidate_score": [100.0, 90.0, 100.0],
            "signal_eligible": [True, True, False],
        }
    )
    frame.to_csv(candidate_dir / "candidates_2024-01-02.csv", index=False)
    frame.iloc[[2]].to_csv(candidate_dir / "unlisted.csv", index=False)
    (candidate_dir / "manifest.json").write_text(
        json.dumps(
            {
                "strict_financial_point_in_time": True,
                "snapshots": [
                    {"date": "2024-01-02", "file": "candidates_2024-01-02.csv"}
                ],
            }
        ),
        encoding="utf-8",
    )
    candidates, _ = load_manifest_candidates(candidate_dir)
    assert len(candidates) == 2
    assert candidates["audit_rank"].tolist() == [1.0, 2.0]

    prices = {
        "sz.000001": pd.Series(
            [10.0, 11.0, 12.0],
            index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
        ),
        "sz.000002": pd.Series(
            [20.0, 19.0, 18.0],
            index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
        ),
    }
    labeled = attach_forward_returns(candidates, prices, horizons=(1, 2))
    assert labeled.loc[labeled["code"] == "sz.000001", "forward_1d"].iloc[0] == pytest.approx(0.1)
    assert labeled.loc[labeled["code"] == "sz.000002", "forward_1d"].iloc[0] == pytest.approx(-0.05)

    metrics = summarize_audit(labeled, horizons=(1,), buckets=(1, 2))
    top1 = metrics[metrics["bucket"] == "top1"].iloc[0]
    universe = metrics[metrics["bucket"] == "universe"].iloc[0]
    assert top1["mean_forward_return"] > universe["mean_forward_return"]
    assert top1["mean_top_minus_universe"] > 0
