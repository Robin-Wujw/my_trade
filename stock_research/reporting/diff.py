"""Previous-result persistence and HTML difference rendering."""
from __future__ import annotations

import json
from pathlib import Path


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
