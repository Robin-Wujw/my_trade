"""Safe discovery and normalization of local backtest artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd


@dataclass(frozen=True)
class BacktestArtifact:
    """A summary and the optional files emitted beside it."""

    directory: Path
    summary_path: Path
    equity_path: Optional[Path]
    trades_path: Optional[Path]
    events_path: Optional[Path]
    summary: dict[str, Any]

    @property
    def label(self) -> str:
        return f"{self.directory.name} · {self.summary_path.name}"

    @property
    def end_date(self) -> str:
        return str(self.summary.get("end_date") or "")


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def load_summary(path: Path) -> dict[str, Any]:
    """Load a JSON summary and reject non-object documents."""

    with path.open("r", encoding="utf-8-sig") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"summary must be a JSON object: {path}")
    return value


def _sibling(directory: Path, summary: Path, suffix: str, root: Path) -> Optional[Path]:
    stem = summary.name[: -len("_summary.json")]
    candidate = directory / f"{stem}{suffix}"
    return candidate if _inside(candidate, root) and candidate.is_file() else None


def discover_backtests(root: Path, *, limit: int = 50) -> list[BacktestArtifact]:
    """Find valid summaries below *root* without following paths elsewhere.

    Corrupt or partially written JSON files are skipped.  A dashboard should
    remain usable when a long-running backtest leaves an incomplete artifact.
    """

    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        return []
    found: list[BacktestArtifact] = []
    candidates: list[tuple[float, Path]] = []
    for summary_path in root.rglob("*_summary.json"):
        if not _inside(summary_path, root):
            continue
        try:
            candidates.append((summary_path.stat().st_mtime, summary_path))
        except OSError:
            continue
    # Large summaries contain a complete trade ledger.  Inspect only the most
    # recent files before parsing JSON so opening the dashboard stays quick.
    candidates.sort(key=lambda item: item[0], reverse=True)
    for _, summary_path in candidates[: max(1, int(limit))]:
        try:
            summary = load_summary(summary_path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        directory = summary_path.parent
        found.append(
            BacktestArtifact(
                directory=directory,
                summary_path=summary_path,
                equity_path=_sibling(directory, summary_path, "_equity.csv", root),
                trades_path=_sibling(directory, summary_path, "_trades.csv", root),
                events_path=_sibling(directory, summary_path, "_events.csv", root),
                summary=summary,
            )
        )
    def sort_key(item: BacktestArtifact) -> tuple[str, float]:
        try:
            mtime = item.summary_path.stat().st_mtime
        except OSError:
            mtime = 0.0
        return str(item.summary.get("end_date") or ""), mtime

    found.sort(key=sort_key, reverse=True)
    return found


def load_table(path: Optional[Path], *, max_rows: Optional[int] = None) -> pd.DataFrame:
    """Read a CSV artifact, returning an empty frame for missing files."""

    if path is None or not path.is_file():
        return pd.DataFrame()
    try:
        frame = pd.read_csv(path, nrows=max_rows)
    except (OSError, UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError):
        return pd.DataFrame()
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    return frame


def drawdown_frame(equity: pd.DataFrame) -> pd.DataFrame:
    """Add normalized equity and drawdown columns for charting."""

    if equity.empty or "equity" not in equity.columns:
        return pd.DataFrame(columns=["date", "equity", "drawdown_pct"])
    result = equity.copy()
    result["equity"] = pd.to_numeric(result["equity"], errors="coerce")
    result = result.dropna(subset=["equity"])
    if result.empty:
        return pd.DataFrame(columns=["date", "equity", "drawdown_pct"])
    peak = result["equity"].cummax()
    result["drawdown_pct"] = (result["equity"] / peak - 1.0) * 100.0
    return result


def filter_trades(
    trades: pd.DataFrame,
    *,
    code: str = "全部",
    action: str = "全部",
    keyword: str = "",
) -> pd.DataFrame:
    """Filter trade rows without mutating the loaded frame."""

    result = trades.copy()
    if code and code != "全部" and "code" in result.columns:
        result = result[result["code"].astype(str) == str(code)]
    if action and action != "全部" and "action" in result.columns:
        result = result[result["action"].astype(str) == str(action)]
    if keyword:
        text = result.astype(str).agg(" ".join, axis=1)
        result = result[text.str.contains(keyword, case=False, na=False, regex=False)]
    return result


def quality_badges(summary: dict[str, Any]) -> list[tuple[str, str]]:
    """Return concise research qualification labels for the UI."""

    badges: list[tuple[str, str]] = []
    if summary.get("research_only"):
        badges.append(("研究回测", "warning"))
    if summary.get("acceptance_status") == "data_limited" or summary.get("data_limited"):
        badges.append(("数据受限", "warning"))
    if summary.get("strict_point_in_time") is True:
        badges.append(("严格时点", "success"))
    elif summary.get("strict_point_in_time") is False:
        badges.append(("非严格时点", "warning"))
    if summary.get("close_confirmed_execution"):
        badges.append((str(summary["close_confirmed_execution"]), "info"))
    if summary.get("coverage_complete") is False:
        badges.append(("候选覆盖不完整", "warning"))
    return badges


def summary_metrics(summary: dict[str, Any]) -> list[tuple[str, Any]]:
    """Stable metric order used by the dashboard and lightweight callers."""

    trade_summary = summary.get("trade_summary") or {}
    trade_count = trade_summary.get("trade_count")
    if trade_count is None:
        buy_count = trade_summary.get("buy_count") or 0
        sell_count = trade_summary.get("sell_count") or 0
        trade_count = buy_count + sell_count or summary.get("event_count")
    return [
        ("最终收益", summary.get("final_return_pct")),
        ("最大回撤", summary.get("maximum_drawdown_pct")),
        ("交易笔数", trade_count),
        ("交易成本", summary.get("transaction_cost_pct")),
        ("初始资金", summary.get("initial_capital")),
        ("最大持仓数", summary.get("max_total_held_symbols", summary.get("maximum_total_held_symbols"))),
    ]


__all__ = [
    "BacktestArtifact",
    "discover_backtests",
    "drawdown_frame",
    "filter_trades",
    "load_summary",
    "load_table",
    "quality_badges",
    "summary_metrics",
]
