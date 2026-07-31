import hashlib
import json

import pandas as pd

from apps.baostock_risk_history_backfill import (
    load_manifest_verified_candidate_codes,
    required_suspension_dates,
)


def test_load_manifest_verified_candidate_codes_filters_dates(tmp_path):
    factor = tmp_path / "factor_a"
    factor.mkdir()
    old = factor / "candidates_2020-12-31.csv"
    current = factor / "candidates_2021-01-04.csv"
    pd.DataFrame({"code": ["sh.600001"]}).to_csv(old, index=False)
    pd.DataFrame({"code": ["sz.000002"]}).to_csv(current, index=False)
    manifest = {
        "snapshots": [
            {
                "date": "2020-12-31",
                "file": old.name,
                "sha256": hashlib.sha256(old.read_bytes()).hexdigest(),
            },
            {
                "date": "2021-01-04",
                "file": current.name,
                "sha256": hashlib.sha256(current.read_bytes()).hexdigest(),
            },
        ],
    }
    (factor / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8",
    )

    result = load_manifest_verified_candidate_codes(
        tmp_path,
        start_date="2021-01-01",
        end_date="2021-12-31",
    )

    assert result == {"sz.000002"}


def test_required_suspension_dates_stop_before_delisting():
    dates = pd.to_datetime([
        "2021-03-18", "2021-03-19", "2021-03-22", "2021-03-23",
    ])

    result = required_suspension_dates(
        dates,
        last_trade_date="2021-03-17",
        end_date="2021-12-31",
        security_end_date="2021-03-22",
    )

    assert result == {
        pd.Timestamp("2021-03-18"),
        pd.Timestamp("2021-03-19"),
    }
