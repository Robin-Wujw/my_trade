"""Local Streamlit research dashboard for my_trade.

Run with ``streamlit run apps/dashboard.py``.  Streamlit is optional and is
never imported by the core package or the ordinary command-line tools.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
from typing import Any, Optional

from stock_research.dashboard.catalog import (
    BacktestArtifact,
    discover_backtests,
    drawdown_frame,
    filter_trades,
    load_table,
    quality_badges,
    summary_metrics,
)
from stock_research.dashboard.runner import BacktestJob, start_backtest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _runtime_root() -> Path:
    configured = os.environ.get("STOCK_RESEARCH_VAR")
    return Path(configured).expanduser().resolve() if configured else PROJECT_ROOT / "var"


def _number(value: Any, suffix: str = "") -> str:
    if value is None or value == "":
        return "—"
    try:
        return f"{float(value):,.2f}{suffix}"
    except (TypeError, ValueError):
        return str(value)


def _render_badges(st: Any, summary: dict[str, Any]) -> None:
    badges = quality_badges(summary)
    if not badges:
        st.info("该摘要没有写入资格标签，请先按回测报告核验口径。")
        return
    columns = st.columns(min(4, len(badges)))
    for column, (label, kind) in zip(columns, badges):
        with column:
            if kind == "success":
                st.success(label)
            elif kind == "warning":
                st.warning(label)
            else:
                st.info(label)


def _render_overview(st: Any, artifact: BacktestArtifact) -> None:
    summary = artifact.summary
    _render_badges(st, summary)
    metrics = summary_metrics(summary)
    columns = st.columns(len(metrics))
    for column, (label, value) in zip(columns, metrics):
        suffix = "%" if label in {"最终收益", "最大回撤", "交易成本"} else ""
        column.metric(label, _number(value, suffix))
    limitations = summary.get("data_limitations") or summary.get("qualification")
    if limitations:
        st.caption("数据资格与限制")
        if isinstance(limitations, (list, tuple)):
            for limitation in limitations:
                st.write(f"• {limitation}")
        else:
            st.write(str(limitations))
    with st.expander("执行与成本参数", expanded=False):
        fields = {
            "执行模式": summary.get("close_confirmed_execution"),
            "佣金率": summary.get("commission_rate"),
            "最低佣金": summary.get("minimum_commission"),
            "卖出印花税率": summary.get("sell_stamp_duty_rate"),
            "估算滑点率": summary.get("estimated_slippage_rate"),
            "严格时点": summary.get("strict_point_in_time"),
            "候选覆盖完整": summary.get("coverage_complete"),
        }
        st.json(fields)


def _render_charts(st: Any, artifact: BacktestArtifact) -> None:
    equity = drawdown_frame(load_table(artifact.equity_path))
    if equity.empty:
        st.info("该回测没有可读取的净值文件。")
        return
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        st.dataframe(equity.tail(100), use_container_width=True)
        st.warning("安装 dashboard 可选依赖后可启用交互图表：python -m pip install -e \".[dashboard]\"")
        return
    figure = make_subplots(specs=[[{"secondary_y": True}]])
    figure.add_trace(go.Scatter(x=equity["date"], y=equity["equity"], name="净值"), secondary_y=False)
    figure.add_trace(go.Scatter(x=equity["date"], y=equity["drawdown_pct"], name="回撤 %", fill="tozeroy"), secondary_y=True)
    figure.update_yaxes(title_text="账户净值", secondary_y=False)
    figure.update_yaxes(title_text="回撤 %", secondary_y=True)
    figure.update_layout(height=460, hovermode="x unified", margin={"l": 10, "r": 10, "t": 30, "b": 10})
    st.plotly_chart(figure, use_container_width=True)


def _render_trades(st: Any, artifact: BacktestArtifact) -> None:
    trades = load_table(artifact.trades_path)
    if trades.empty:
        st.info("该回测没有交易流水。")
        return
    left, middle, right = st.columns([1, 1, 2])
    codes = ["全部"] + sorted(trades["code"].dropna().astype(str).unique().tolist()) if "code" in trades else ["全部"]
    actions = ["全部"] + sorted(trades["action"].dropna().astype(str).unique().tolist()) if "action" in trades else ["全部"]
    code = left.selectbox("代码", codes)
    action = middle.selectbox("动作", actions)
    keyword = right.text_input("原因或字段搜索")
    filtered = filter_trades(trades, code=code, action=action, keyword=keyword)
    st.caption(f"显示 {len(filtered):,} / {len(trades):,} 条流水")
    preferred = ["date", "code", "name", "action", "execution_price", "quantity", "profit_loss_pct", "reason", "execution_timing"]
    columns = [column for column in preferred if column in filtered.columns]
    st.dataframe(filtered[columns] if columns else filtered, use_container_width=True, hide_index=True)


def _render_job(st: Any) -> None:
    job = st.session_state.get("dashboard_job")
    if not isinstance(job, BacktestJob):
        return
    job.poll()
    st.subheader("回测任务")
    st.code(" ".join(job.command), language="powershell")
    if job.running:
        st.info("任务运行中。刷新页面可查看最新日志。")
    elif job.returncode == 0:
        st.success("回测完成，请重新扫描产物。")
    else:
        st.error(f"回测退出码：{job.returncode}")
    st.text_area("运行日志", "\n".join(job.lines[-300:]), height=260)


def _render_runner(st: Any, runtime_root: Path) -> None:
    st.subheader("启动研究回测")
    st.caption("此处只启动本地回测 CLI，不连接券商，也不会自动下单。")
    with st.form("backtest_form"):
        first, second = st.columns(2)
        start_date = first.text_input("开始日期", value="2021-01-01")
        end_date = second.text_input("结束日期（可空）", value="")
        output_name = st.text_input("输出目录名", value="dashboard_run")
        frozen = st.checkbox(
            "允许使用已冻结本地输入（研究回测）",
            value=False,
            help="勾选后会传 --no-refresh-inputs --allow-frozen-inputs；结果仍按研究口径显示。",
        )
        execution = st.selectbox("收盘确认后的执行", ["next_open", "close_proxy"], index=0)
        with st.expander("成本参数"):
            initial_capital = st.number_input("初始资金", min_value=1.0, value=1_000_000.0, step=10_000.0)
            commission = st.number_input("佣金率", min_value=0.0, value=0.000085, format="%.7f")
            minimum_commission = st.number_input("最低佣金", min_value=0.0, value=5.0)
            stamp = st.number_input("卖出印花税率", min_value=0.0, value=0.0005, format="%.7f")
            slippage = st.number_input("估算滑点率", min_value=0.0, value=0.0005, format="%.7f")
        submitted = st.form_submit_button("启动回测", type="primary")
    if submitted:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", output_name.strip()):
            st.error("输出目录名只能包含字母、数字、下划线和短横线，长度不超过 64。")
        elif isinstance(st.session_state.get("dashboard_job"), BacktestJob) and st.session_state["dashboard_job"].running:
            st.error("已有回测任务运行中，请等待它结束后再启动新的任务。")
        else:
            output_directory = runtime_root / "backtests" / output_name.strip()
            st.session_state["dashboard_job"] = start_backtest(
                PROJECT_ROOT,
                start_date=start_date,
                end_date=end_date,
                output_directory=output_directory,
                allow_frozen_inputs=frozen,
                close_confirmed_execution=execution,
                initial_capital=initial_capital,
                commission_rate=commission,
                minimum_commission=minimum_commission,
                sell_stamp_duty_rate=stamp,
                estimated_slippage_rate=slippage,
            )
            st.success("回测已启动。")
    _render_job(st)


def main() -> None:
    try:
        import streamlit as st
    except ImportError as exc:  # pragma: no cover - exercised by CLI smoke checks
        raise SystemExit("缺少 Streamlit，请运行 python -m pip install -e \".[dashboard]\"") from exc
    st.set_page_config(page_title="my_trade 量化研究工作台", page_icon="📈", layout="wide")
    st.title("my_trade 量化研究工作台")
    st.caption("参考 QuantsPlaybook 的研究结果组织方式与 OSkhQuant 的任务/日志体验；当前界面只服务本地研究回测。")
    runtime_root = _runtime_root()
    with st.sidebar:
        st.header("产物目录")
        configured = st.text_input("运行根目录", value=str(runtime_root))
        runtime_root = Path(configured).expanduser().resolve()
        if st.button("重新扫描"):
            st.rerun()
    artifacts = discover_backtests(runtime_root / "backtests", limit=50)
    if not artifacts:
        st.warning(f"没有发现回测摘要：{runtime_root / 'backtests'}")
    selected: Optional[BacktestArtifact] = None
    if artifacts:
        labels = [artifact.label for artifact in artifacts]
        index = st.sidebar.selectbox("选择回测", range(len(labels)), format_func=lambda item: labels[item])
        selected = artifacts[index]
        st.caption(f"摘要：{selected.summary_path}")
        _render_overview(st, selected)
        overview, trades, raw = st.tabs(["净值与回撤", "交易流水", "原始摘要"])
        with overview:
            _render_charts(st, selected)
        with trades:
            _render_trades(st, selected)
        with raw:
            st.json(selected.summary)
    st.divider()
    _render_runner(st, runtime_root)


if __name__ == "__main__":
    main()
