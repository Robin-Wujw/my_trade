"""解读 hybrid + PE/PB 完整回测结果：对比基线，量化左侧贡献，检查审计红线。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

BASELINE = {
    "total_return_pct": 90.015,
    "max_drawdown_pct": -13.328,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", default="var/backtests/hybrid_pe_pb_5y")
    args = parser.parse_args()

    root = Path(args.directory)
    summaries = sorted(root.glob("*_summary.json"))
    if not summaries:
        raise SystemExit(f"未找到 summary: {root}")
    s = json.loads(summaries[0].read_text(encoding="utf-8"))

    print("=== 完整回测结果（含 PE/PB 低估左侧） ===")
    print(f"  区间: {s.get('actual_start')} .. {s.get('end_date')}")
    print(f"  总收益: {s['final_return_pct']:+.2f}%   (基线 {BASELINE['total_return_pct']:+.2f}%)")
    print(f"    实现收益: {s['realized_return_pct']:+.2f}%  未实现: {s['unrealized_return_pct']:+.2f}%")
    print(f"  最大回撤: {s['maximum_drawdown_pct']:.2f}%   (基线 {BASELINE['max_drawdown_pct']:.2f}%)")
    print(f"  最大毛敞口: {s.get('maximum_gross_exposure_pct')}%")
    print(f"  交易成本占比: {s.get('transaction_cost_pct')}%")

    mix = s.get("trade_mix_summary") or {}
    print("\n=== 左右通道贡献 ===")
    for acc in ("left", "right"):
        stat = mix.get("trade_count_by_account", {}).get(acc, 0)
        pnl = mix.get("net_pnl_by_account", {}).get(acc, 0)
        amount = mix.get("trade_amount_by_account", {}).get(acc, 0)
        print(f"  {acc:>5}: 交易{stat}笔  交易额{amount:,.0f}  净盈亏{pnl:+,.0f}元")
    top = mix.get("left_grid_trade_count_by_symbol_top") or []
    if top:
        print("  左侧网格标的 top:")
        for item in top:
            print(f"    {item['symbol']}: {item['trade_count']}笔")
    bought = mix.get("buy_count_by_account", {})
    sold = mix.get("sell_count_by_account", {})
    print(f"  左侧买{bought.get('left',0)}卖{sold.get('left',0)}；右侧买{bought.get('right',0)}卖{sold.get('right',0)}")

    # 审计红线检查：候选伪造 / 违规事件
    print("\n=== 审计检查 ===")
    blocks = s.get("entry_blocks") or []
    conc = s.get("concentration_blocks") or []
    print(f"  entry_blocks(技术入场被拒): {len(blocks)}  concentration_blocks: {len(conc)}")
    if conc:
        print("  !! 存在集中度违规:", json.dumps(conc, ensure_ascii=False)[:500])
    for field in ("violations", "audit_violations"):
        if field in s:
            print(f"  {field}: {s[field]}")
    events_path = list(root.glob("*_events.csv"))
    if events_path:
        ev = pd.read_csv(events_path[0], dtype={"code": str})
        bad = ev[ev.astype(str).apply(
            lambda r: r.str.contains("falsif|violat|audit", case=False, na=False).any(),
            axis=1,
        )]
        print(f"  events 违规标记行: {len(bad)}")

    print("\n=== 左侧交易来源核对 ===")
    ledger = s.get("trade_ledger") or []
    left_codes = sorted({
        t["code"] for t in ledger
        if t.get("account_mode") == "left" and t.get("position_change_pct", 0) > 0
    })
    print(f"  左侧买入标的({len(left_codes)}): {left_codes}")
    if events_path:
        ev = pd.read_csv(events_path[0], dtype={"code": str})
        left = ev[ev.get("account_mode") == "left"]
        if not left.empty and "selection_reason" in left.columns:
            reasons = left["selection_reason"].value_counts().to_dict()
            print(f"  左侧 selection_reason: {reasons}")


if __name__ == "__main__":
    main()
