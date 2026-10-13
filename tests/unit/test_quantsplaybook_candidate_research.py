import json

import pandas as pd

from stock_research.research import quantsplaybook_candidate_research as research


class _Repository:
    def load_daily_kline_frames(self, codes, *, start_date, end_date):
        return pd.DataFrame({"code": list(codes)})


def _write_snapshot(directory, date):
    path = directory / f"candidates_{date}.csv"
    pd.DataFrame([{
        "code": "sh.600001",
        "playbook_factor": "factor_a",
        "playbook_factor_rank": 1,
    }]).to_csv(path, index=False, encoding="utf-8-sig")
    return {
        "date": date,
        "file": path.name,
        "sha256": research._sha256(path),
        "candidate_count": 1,
        "signal_eligible_count": 1,
    }


def test_partial_research_output_contains_only_selected_snapshots(
    tmp_path, monkeypatch,
):
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    snapshots = [
        _write_snapshot(source, "2023-12-29"),
        _write_snapshot(source, "2024-01-02"),
    ]
    (source / "manifest.json").write_text(
        json.dumps({
            "snapshot_count": 2,
            "start_date": "2023-12-29",
            "end_date": "2024-01-02",
            "snapshots": snapshots,
        }),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        research,
        "build_candidate_trading_profile_enrichment",
        lambda prices, dates, **kwargs: {
            pd.Timestamp(date).strftime("%Y-%m-%d"): {
                "sh.600001": {"candidate_profile_status": "current"},
            }
            for date in dates
        },
    )

    def outcome_labels(snapshots, prices, **kwargs):
        return pd.DataFrame([{
            "date": date,
            "code": row["code"],
            "playbook_factor_rank": row["playbook_factor_rank"],
            "label_available": True,
        } for date, rows in snapshots.items() for row in rows])

    monkeypatch.setattr(
        research, "build_candidate_outcome_labels", outcome_labels,
    )

    result = research.enrich_frozen_candidate_directory(
        source,
        repository=_Repository(),
        output_directory=output,
        start_date="2023-01-01",
        end_date="2023-12-31",
    )

    manifest = json.loads(
        (output / "manifest.json").read_text(encoding="utf-8"),
    )
    outcomes = pd.read_csv(output / "candidate_outcomes.csv")

    assert result["snapshot_count"] == 1
    assert manifest["snapshot_count"] == 1
    assert manifest["start_date"] == "2023-12-29"
    assert manifest["end_date"] == "2023-12-29"
    assert [row["date"] for row in manifest["snapshots"]] == ["2023-12-29"]
    assert (output / "candidates_2023-12-29.csv").is_file()
    assert not (output / "candidates_2024-01-02.csv").exists()
    assert outcomes["date"].tolist() == ["2023-12-29"]
    assert (source / "candidates_2024-01-02.csv").is_file()


def test_profiles_only_does_not_build_or_write_legacy_outcomes(
    tmp_path, monkeypatch,
):
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    snapshot = _write_snapshot(source, "2024-01-02")
    (source / "manifest.json").write_text(json.dumps({
        "snapshot_count": 1,
        "start_date": "2024-01-02",
        "end_date": "2024-01-02",
        "snapshots": [snapshot],
    }), encoding="utf-8")
    monkeypatch.setattr(
        research,
        "build_candidate_trading_profile_enrichment",
        lambda prices, dates, **kwargs: {
            "2024-01-02": {
                "sh.600001": {"candidate_profile_status": "current"},
            },
        },
    )
    monkeypatch.setattr(
        research,
        "build_candidate_outcome_labels",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("profiles-only must not build outcome labels")
        ),
    )

    result = research.enrich_frozen_candidate_directory(
        source,
        repository=_Repository(),
        output_directory=output,
        generate_outcome_labels=False,
    )

    manifest = json.loads(
        (output / "manifest.json").read_text(encoding="utf-8"),
    )
    assert result["outcome_row_count"] == 0
    assert manifest["candidate_profiles_only"] is True
    assert "candidate_outcome_labels" not in manifest
    assert not (output / "candidate_outcomes.csv").exists()
