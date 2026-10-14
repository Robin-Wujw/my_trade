"""Verify and render immutable local paper artifacts, without any delivery API."""
from __future__ import annotations

import html
import json
from pathlib import Path

import pandas as pd

from stock_research.backtesting.baseline import BASELINE_PATH, load_baseline
from stock_research.research.frozen_forward import sha256


def load_forward_report(manifest_path, observation_date, *, now=None):
    path = Path(manifest_path).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (manifest.get("version") != "frozen-forward-v1"
            or manifest.get("model_id") != load_baseline()["model_id"]
            or manifest.get("configuration_sha256") != sha256(BASELINE_PATH)
            or manifest.get("research_only") is not True
            or manifest.get("strict_point_in_time") is not False):
        raise ValueError("forward manifest model/configuration/provenance mismatch")
    entries = [item for item in manifest.get("snapshots", []) if item.get("date") == observation_date]
    if len(entries) != 1:
        raise ValueError("forward report requires exactly one same-day capture")
    entry = entries[0]
    captured = pd.Timestamp(entry["captured_at"])
    current = pd.Timestamp.now(tz="Asia/Shanghai") if now is None else pd.Timestamp(now)
    if captured.tzinfo is None or current.tzinfo is None:
        raise ValueError("forward capture must have an explicit timezone")
    local = captured.tz_convert("Asia/Shanghai")
    if (local.strftime("%Y-%m-%d") != observation_date
            or current.tz_convert("Asia/Shanghai").strftime("%Y-%m-%d") != observation_date
            or (local.hour, local.minute) < (16, 30) or captured > current):
        raise ValueError("forward report refuses stale, future or unfinished captures")
    candidate_path = (path.parent / entry["file"]).resolve()
    if not candidate_path.is_relative_to(path.parent):
        raise ValueError("forward candidate path escapes manifest directory")
    files = {"candidates": (candidate_path, "sha256"),
             "orders": (candidate_path.parent / "paper_orders.json", "orders_sha256"),
             "replay": (candidate_path.parent / "paper_replay.json", "paper_replay_sha256")}
    for file, key in files.values():
        if not file.is_file() or sha256(file) != entry.get(key):
            raise ValueError("forward artifact hash mismatch")
    if any(key in entry for key in ("structure_event_receipt", "structure_event_manifest", "structure_event_manifest_sha256")):
        from stock_research.research.structure_event_store import load_structure_session
        structure_path = (path.parent / entry.get("structure_event_manifest", "")).resolve()
        if (not structure_path.is_relative_to(candidate_path.parent) or not structure_path.is_file()
                or sha256(structure_path) != entry.get("structure_event_manifest_sha256")):
            raise ValueError("forward structure manifest path/hash mismatch")
        document, _ = load_structure_session(structure_path.parent)
        if (document != entry.get("structure_event_receipt")
                or document.get("observation_date") != observation_date
                or pd.Timestamp(document["captured_at"]) != captured):
            raise ValueError("forward structure capture receipt mismatch")
    candidates = pd.read_csv(candidate_path, dtype={"code": str})
    required = {"code", "date", "observation_date", "captured_at", "candidate_source", "research_only"}
    if (not required.issubset(candidates) or len(candidates) != entry.get("candidate_count")
            or candidates.empty or candidates["code"].duplicated().any()
            or set(candidates["date"]) != {observation_date}
            or set(candidates["observation_date"]) != {observation_date}
            or set(candidates["captured_at"]) != {entry["captured_at"]}
            or not candidates["research_only"].eq(True).all()):
        raise ValueError("forward candidate date/count/provenance mismatch")
    orders_payload = json.loads(files["orders"][0].read_text(encoding="utf-8"))
    replay = json.loads(files["replay"][0].read_text(encoding="utf-8"))
    if (orders_payload.get("paper_only") is not True or replay.get("paper_only") is not True
            or replay.get("research_only") is not True
            or orders_payload.get("observation_date") != observation_date
            or orders_payload.get("captured_at") != entry["captured_at"]):
        raise ValueError("forward paper metadata mismatch")
    orders = orders_payload.get("orders")
    if not isinstance(orders, list) or any(
        item.get("filled") is not False or item.get("triggered") is not False
        or item.get("observation_date") != observation_date or item.get("research_only") is not True
        or item.get("quantity") is not None
        for item in orders
    ):
        raise ValueError("paper intent cannot claim an execution or trigger")
    return {"manifest_path": str(path), "entry": entry,
            "candidates": candidates, "orders": orders, "replay": replay,
            "model_id": manifest["model_id"]}


def render_forward_report(bundle):
    escape = lambda value: html.escape(str(value if value is not None else "—"))
    parts = ["<h2>冻结策略候选与纸面意图</h2>",
             "<p>研究用途；当前记录为条件意图，尚未触发或成交。下一交易日仍须复核价格、停牌、涨跌停、T+1、整手与成本。</p>",
             f"<p>模型：{escape(bundle['model_id'])}；实际捕获：{escape(bundle['entry']['captured_at'])}。</p>",
             "<table><tr><th>代码</th><th>来源</th><th>右侧排名</th><th>评分</th></tr>"]
    for row in bundle["candidates"].to_dict("records"):
        parts.append("<tr>" + "".join(f"<td>{escape(row.get(key))}</td>" for key in
                                     ("code", "candidate_source", "rank", "score")) + "</tr>")
    parts.append("</table><h3>条件纸面意图</h3><ul>")
    for row in bundle["orders"]:
        parts.append(f"<li>{escape(row.get('code'))} {escape(row.get('side'))}；"
                     f"信号：{escape(row.get('signal'))}；计划价：{escape(row.get('planned_price'))}；"
                     f"止损：{escape(row.get('stop_price'))}；目标：{escape(row.get('target_price'))}；"
                     "未触发，未成交。</li>")
    if not bundle["orders"]:
        parts.append("<li>当日无条件纸面意图。</li>")
    parts.append("</ul>")
    return "".join(parts)
