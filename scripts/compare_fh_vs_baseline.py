"""对比 A1 factor_horizon 实验结果与基线。

用法: python scripts/compare_fh_vs_baseline.py [--control DIR] [--experiment DIR]
默认读取标准输出目录, 缺失的运行会跳过。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


BASE = Path("var/backtests")
SUMMARY_NAME = "portfolio_2021-01-01_2026-07-21_summary.json"

RUNS = {
    "基线 value_only(原始候选)": BASE / "hybrid_value_only_5y",
    "对照 FH候选(旧代码)": BASE / "hybrid_fh_nofh_5y",
    "实验 FH+2千万(旧代码)": BASE / "hybrid_fh_fh_5y",
    "对照 FH候选(gate修复)": BASE / "hybrid_fh_nofh_gatefix_5y",
    "实验 FH+2千万(gate修复)": BASE / "hybrid_fh_fh_gatefix_5y",
}


def _summary(root: Path) -> dict | None:
    path = root / SUMMARY_NAME
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _row(s: dict | None, keys: list[str]) -> list[str]:
    if s is None:
        return ["(未运行)"] * len(keys)
    out = []
    for k in keys:
        v = s.get(k, "")
        if isinstance(v, float):
            out.append(f"{v:.2f}")
        else:
            out.append(str(v))
    return out


def _pnl(s: dict | None) -> str:
    if s is None:
        return "(未运行)"
    tm = s.get("trade_mix_summary", {})
    net = tm.get("net_pnl_by_account", {})
    return json.dumps(net, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", help="覆盖对照回测目录")
    parser.add_argument("--experiment", help="覆盖实验组回测目录")
    args = parser.parse_args()

    runs = dict(RUNS)
    if args.control:
        runs["对照 FH候选(不开FH通道)"] = Path(args.control)
    if args.experiment:
        runs["实验 FH通道+2千万门槛"] = Path(args.experiment)

    keys = [
        "final_return_pct",
        "realized_return_pct",
        "unrealized_return_pct",
        "maximum_drawdown_pct",
        "factor_horizon_channel_enabled",
        "factor_horizon_buy_count",
        "factor_horizon_min_avg_amount",
    ]
    header = ["运行"] + keys
    print(" | ".join(header))
    print("-" * len(" | ".join(header)))
    for name, root in runs.items():
        s = _summary(root)
        print(" | ".join([name] + _row(s, keys)))

    print("\n=== 交易与盈亏分解 (trade_mix) ===")
    for name, root in runs.items():
        s = _summary(root)
        if s is None:
            print(f"{name}: (未运行)")
            continue
        tm = s.get("trade_mix_summary", {})
        tca = tm.get("trade_count_by_account", {})
        bca = tm.get("buy_count_by_account", {})
        npa = tm.get("net_pnl_by_account", {})
        print(f"{name}: 交易={tca} 买入={bca} 净利={npa}")

    print("\n=== 全局 trade_summary ===")
    for name, root in runs.items():
        s = _summary(root)
        if s is None:
            continue
        ts = s.get("trade_summary", {})
        print(f"{name}: {json.dumps(ts, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
