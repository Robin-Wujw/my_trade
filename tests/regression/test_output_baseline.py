import csv
import json
import subprocess
import sys
from pathlib import Path

from my_trade.regression.output_baseline import build_entry, compare_entry


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def test_semantic_hash_is_order_independent_but_detects_value_drift(tmp_path):
    path = tmp_path / "formula33.csv"
    rows = [
        {
            "date": "2026-06-30",
            "base_count": "239",
            "count": "65",
            "change": "21",
            "up_streak": "1",
            "down_streak": "0",
            "signal": "观察",
        },
        {
            "date": "2026-06-29",
            "base_count": "200",
            "count": "44",
            "change": "-2",
            "up_streak": "0",
            "down_streak": "1",
            "signal": "观察",
        },
    ]
    write_csv(path, rows)
    baseline = build_entry(path, kind="formula33", relative_to=tmp_path)

    write_csv(path, list(reversed(rows)))
    reordered = build_entry(path, kind="formula33", relative_to=tmp_path)
    assert reordered["semantic_sha256"] == baseline["semantic_sha256"]
    assert reordered["file_sha256"] != baseline["file_sha256"]

    rows[0]["count"] = "66"
    write_csv(path, rows)
    differences = compare_entry(baseline, root=tmp_path)
    assert "file_sha256 changed" in differences
    assert "semantic_sha256 changed" in differences


def test_legacy_manifest_tracks_six_real_outputs():
    manifest_path = Path(__file__).with_name("legacy-output-v1.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    entries = manifest["entries"]
    assert len(entries) == 6
    assert [entry["kind"] for entry in entries].count("formula33") == 3
    assert [entry["kind"] for entry in entries].count("daily_selection") == 2
    assert [entry["kind"] for entry in entries].count("factor_selection") == 1
    assert all(entry["row_count"] > 0 for entry in entries)
    assert all(len(entry["semantic_sha256"]) == 64 for entry in entries)


def test_module_cli_verifies_without_import_warning(tmp_path):
    csv_path = tmp_path / "formula33.csv"
    write_csv(
        csv_path,
        [
            {
                "date": "2026-06-30",
                "base_count": "239",
                "count": "65",
                "change": "21",
                "up_streak": "1",
                "down_streak": "0",
                "signal": "观察",
            }
        ],
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "root": ".",
                "entries": [build_entry(csv_path, kind="formula33", relative_to=tmp_path)],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "my_trade.regression.output_baseline",
            "verify",
            str(manifest_path),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert result.stdout.strip() == "1 baselines verified"
    assert result.stderr == ""
