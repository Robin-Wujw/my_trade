"""Render the single current baseline reference from its versioned config."""
from stock_research.backtesting.baseline import load_baseline


def render_baseline_document() -> str:
    config = load_baseline()
    p = config["engine_parameters"]
    rows = [
        ("配置版本", config["version"]),
        ("右侧模型", config["model_id"]),
        ("右侧候选数", config["right_top_n"]),
        ("左侧模型", config["left_model_id"]),
        ("主动仓 / 全部持股 / 左侧上限", f'{p["max_positions"]} / {p["max_total_held_symbols"]} / {p["max_left_positions"]}'),
        ("右侧普通首仓及加仓倍数", p["right_entry_size_multiplier"]),
        ("佣金 / 最低佣金", f'{p["commission_rate"]:.4%} / {p["minimum_commission"]:g}元'),
        ("卖出印花税 / 2023-08-28之前", f'{p["sell_stamp_duty_rate"]:.2%} / {p["sell_stamp_duty_rate_before_2023_08_28"]:.2%}'),
        ("估算滑点", f'{p["estimated_slippage_rate"]:.2%}'),
        ("收盘确认成交", p["close_confirmed_execution"]),
        ("初始资金", f'{p["initial_capital"]:,.0f}元'),
        ("ADV20最低成交额", f'{p["factor_only_min_avg_amount"]:,.0f}元'),
        ("买单 / 信号日成交额上限", f'{p["max_entry_signal_amount_participation"]:.0%}'),
        ("买单 / ADV20上限", f'{p["max_entry_avg_amount_participation"]:.0%}'),
        ("审计冻结历史累计收益", f'+{config["historical_result"]["cumulative_return_pct"]:.3f}%'),
        ("历史区间", f'{p["requested_start"]} 至 {p["end_date"]}'),
        ("研究资格", "research_only / data_limited；不是前向验证"),
    ]
    lines = [
        "# 当前冻结基线", "",
        "由 `python -m apps.strategy_baseline` 根据 [策略配置](../config/strategy_baseline.json) 生成。`--check` 核对文档是否过期。", "",
        "| 项目 | 配置或结果 |", "|---|---|",
        *[f"| {label} | {value} |" for label, value in rows], "",
        "历史输入指纹、全部引擎参数和重建限制保存在配置文件内；旧冻结包不改写。其他文档的不同收益均为历史版本。", "",
        "当前公开组合/逐年回测入口从该文件读取策略默认值；低层引擎保留既有研究API，生产适配器须显式传入配置。显式风险预算会取代默认仓位倍数，显式冲突仍报错。结果记录配置哈希、实际参数及偏差，参数一致不等于输入模型一致。", "",
        "行情日期、是否已预处理和成交额单位由实际输入决定，不从历史冻结包盲目继承。普通入口默认成交额为元；原始Tushare数据（千元）必须显式使用 `--price-amount-multiplier 1000`。", "",
        "财务修订版本、历史行业、公司行动派息日和精确涨跌停覆盖仍有缺口；公告日可见不等于修订版本可见。MiniQMT只读，所有订单仅供研究或纸面记录。", "",
        "前向证据从实际捕获日开始。补抓2026-08-11之后的历史数据不能补成此前已经运行的样本外记录。", "",
    ]
    return "\n".join(lines)
