"""Previous-result persistence and HTML difference rendering."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class SelectionDiff:
    added: tuple[dict, ...]
    removed: tuple[dict, ...]
    moved: tuple[dict, ...]


@dataclass(frozen=True)
class SelectionHistory:
    snapshots: dict[str, list[dict]]

    def snapshot_for(self, date_str):
        snapshot = self.snapshots.get(str(date_str))
        return None if snapshot is None else list(snapshot)

    def previous_before(self, date_str):
        earlier = sorted(date for date in self.snapshots if date < str(date_str))
        if not earlier:
            return None
        return list(self.snapshots[earlier[-1]])


def _normalize_snapshot(rows):
    by_code = {}
    for row in rows or []:
        code = str(row.get("code", "")).strip()
        if not code:
            continue
        by_code[code] = {
            "code": code,
            "name": str(row.get("name", code)).strip() or code,
            "strategy_part": str(row.get("strategy_part", "")).strip(),
        }
    return [by_code[code] for code in sorted(by_code)]


def compare_snapshots(previous_rows, current_rows):
    previous = {row["code"]: row for row in _normalize_snapshot(previous_rows)}
    current = {row["code"]: row for row in _normalize_snapshot(current_rows)}
    added = tuple(current[code] for code in sorted(current.keys() - previous.keys()))
    removed = tuple(
        previous[code] for code in sorted(previous.keys() - current.keys())
    )
    moved = []
    for code in sorted(previous.keys() & current.keys()):
        before = previous[code]
        after = current[code]
        if before["strategy_part"] != after["strategy_part"]:
            moved.append(
                {
                    **after,
                    "from_part": before["strategy_part"],
                    "to_part": after["strategy_part"],
                }
            )
    return SelectionDiff(added=added, removed=removed, moved=tuple(moved))


def load_history(path):
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        snapshots = {
            str(item["date"]): _normalize_snapshot(item.get("stocks", []))
            for item in data.get("snapshots", [])
            if item.get("date")
        }
    except (OSError, json.JSONDecodeError, KeyError, TypeError, AttributeError):
        snapshots = {}
    return SelectionHistory(snapshots=snapshots)


def save_snapshot(path, date_str, rows, retain=45):
    target = Path(path)
    history = load_history(target)
    snapshots = dict(history.snapshots)
    snapshots[str(date_str)] = _normalize_snapshot(rows)
    dates = sorted(snapshots)[-max(2, int(retain)) :]
    payload = {
        "snapshots": [
            {"date": date, "stocks": snapshots[date]}
            for date in dates
        ]
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(target.suffix + ".tmp")
    temp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temp.replace(target)
    return SelectionHistory({date: snapshots[date] for date in dates})


def load_last_result(path):
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return {item["code"]: item["name"] for item in data.get("stocks", [])}
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return {}


def save_current_result(path, date_str, rows):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "date": date_str,
        "stocks": [{"code": row["code"], "name": row["name"]} for row in rows],
    }
    target.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def build_diff_html(last_dict, current_rows):
    current_dict = {
        row["code"]: row["name"] for row in current_rows if row.get("code")
    }
    added = {code: name for code, name in current_dict.items() if code not in last_dict}
    removed = {code: name for code, name in last_dict.items() if code not in current_dict}
    if not added and not removed:
        return "<p>与上一交易日相比：无变化</p>"
    parts = []
    if added:
        items = "、".join(f"{name}({code})" for code, name in added.items())
        parts.append(f"<p style='color:red'>🔴 新增({len(added)}): {items}</p>")
    if removed:
        items = "、".join(f"{name}({code})" for code, name in removed.items())
        parts.append(f"<p style='color:green'>🟢 移除({len(removed)}): {items}</p>")
    return "".join(parts)
