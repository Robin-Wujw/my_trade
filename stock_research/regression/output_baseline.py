"""Build and verify deterministic baselines for strategy CSV outputs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Iterable


KIND_COLUMNS = {
    "formula33": (
        "date",
        "base_count",
        "count",
        "change",
        "up_streak",
        "down_streak",
        "signal",
    ),
    "daily_selection": (
        "strategy_part",
        "code",
        "name",
        "report_period",
        "value_line",
        "price_to_value",
        "quality_score",
        "earnings_yoy",
        "mktcap",
        "eps_excl",
        "mainline_boards",
        "date",
        "close",
        "wave_high",
        "wave_low",
        "wave_pct",
        "wave_zone",
    ),
    "factor_selection": (
        "date",
        "code",
        "name",
        "selection_bucket",
        "method",
        "total_score",
        "close",
        "ma20",
        "ma60",
        "return_20d",
        "return_60d",
        "volume_ratio_5_20",
        "deduct_periods",
        "wave_low",
        "wave_high",
        "wave_pct",
        "wave_zone",
    ),
}

TEXT_COLUMNS = {
    "date",
    "strategy_part",
    "code",
    "name",
    "report_period",
    "mainline_boards",
    "wave_zone",
    "selection_bucket",
    "method",
    "deduct_periods",
    "signal",
}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_value(column: str, value: str) -> str:
    text = str(value or "").strip()
    if not text or column in TEXT_COLUMNS:
        return text
    try:
        number = Decimal(text)
    except InvalidOperation:
        return text
    if not number.is_finite():
        return text
    if number == 0:
        return "0"
    return format(number.normalize(), "f")


def _canonical_rows(path: Path, kind: str) -> list[dict[str, str]]:
    if kind not in KIND_COLUMNS:
        raise ValueError(f"unknown baseline kind: {kind}")
    columns = KIND_COLUMNS[kind]
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [column for column in columns if column not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")
        rows = [
            {column: _normalize_value(column, row.get(column, "")) for column in columns}
            for row in reader
        ]
    return sorted(rows, key=lambda row: tuple(row[column] for column in columns))


def _semantic_sha256(kind: str, rows: Iterable[dict[str, str]]) -> str:
    payload = {
        "kind": kind,
        "columns": KIND_COLUMNS[kind],
        "rows": list(rows),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_entry(path, *, kind: str, relative_to=None) -> dict:
    """Create a raw and semantic fingerprint for one historical output."""
    csv_path = Path(path)
    root = Path(relative_to) if relative_to is not None else csv_path.parent
    rows = _canonical_rows(csv_path, kind)
    return {
        "path": csv_path.resolve().relative_to(root.resolve()).as_posix(),
        "kind": kind,
        "row_count": len(rows),
        "file_sha256": _file_sha256(csv_path),
        "semantic_sha256": _semantic_sha256(kind, rows),
    }


def compare_entry(expected: dict, *, root) -> list[str]:
    """Return human-readable drift categories for one expected entry."""
    path = Path(root) / expected["path"]
    if not path.is_file():
        return [f"missing file: {expected['path']}"]
    actual = build_entry(path, kind=expected["kind"], relative_to=root)
    differences = []
    for field in ("row_count", "file_sha256", "semantic_sha256"):
        if actual[field] != expected[field]:
            differences.append(f"{field} changed")
    return differences


def verify_manifest(path) -> tuple[int, list[str]]:
    """Verify every output in a checked-in baseline manifest."""
    manifest_path = Path(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = (manifest_path.parent / manifest.get("root", ".")).resolve()
    issues = []
    for entry in manifest["entries"]:
        differences = compare_entry(entry, root=root)
        issues.extend(f"{entry['path']}: {difference}" for difference in differences)
    return len(manifest["entries"]), issues


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Audit historical strategy output baselines")
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify_parser = subparsers.add_parser("verify", help="verify a baseline manifest")
    verify_parser.add_argument("manifest")
    args = parser.parse_args(argv)

    count, issues = verify_manifest(args.manifest)
    if issues:
        for issue in issues:
            print(issue)
        return 1
    print(f"{count} baselines verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
