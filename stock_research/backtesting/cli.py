"""Argument parsing for the portfolio CLI; execution is delegated to workflow."""
from __future__ import annotations
import argparse
from stock_research.core.paths import PATHS
from stock_research.strategies.historical_candidates import SNAPSHOT_VERSION
from stock_research.backtesting import workflow
from stock_research.backtesting.baseline import BaselineArgumentParser, configure_parser_defaults
from stock_research.backtesting.defaults import (
    DEFAULT_CLOSE_CONFIRMED_EXECUTION,
    DEFAULT_COMMISSION_RATE,
    DEFAULT_ESTIMATED_SLIPPAGE_RATE,
    DEFAULT_MINIMUM_COMMISSION,
    DEFAULT_SELL_STAMP_DUTY_RATE,
    DEFAULT_SELL_STAMP_DUTY_RATE_BEFORE_2023_08_28,
)
from stock_research.backtesting.workflow import (
    DEFAULT_FINANCIAL_CHUNK_SIZE, DEFAULT_FINANCIAL_TARGET_COVERAGE, DEFAULT_FINANCIAL_TIMEOUT,
)


def build_parser():
    parser = BaselineArgumentParser(description="候选池、Formula33和五持仓组合回测", allow_abbrev=False)
    parser.add_argument("--start-date", default="2026-01-01")
    parser.add_argument(
        "--end-date", default="",
        help="default: latest session whose daily bar should be available",
    )
    parser.add_argument(
        "--max-positions", type=int, default=5,
        help="main capacity symbols; default 5, hard-capped at 5",
    )
    parser.add_argument(
        "--max-daily-candidates",
        type=int,
        default=60,
        help="每日进入执行引擎的候选上限；生产默认60，多通道研究需显式放宽",
    )
    parser.add_argument(
        "--max-total-held-symbols", type=int, default=5,
        help=(
            "全部实际持股标的的安全上限，允许1至10只；主动仓仍由"
            "--max-positions限制，合格趋势尾仓不占主动仓名额"
        ),
    )
    parser.add_argument(
        "--max-left-positions", type=int, default=1,
        help="maximum simultaneous left-side symbols; audited baseline defaults to 2",
    )
    parser.add_argument(
        "--max-same-industry", type=int, default=2,
        help="maximum simultaneously held symbols sharing a dated industry/board tag",
    )
    parser.add_argument(
        "--same-theme-correlation", type=float, default=0.60,
        help="60-session return correlation used to group related exposure when tags are missing",
    )
    parser.add_argument(
        "--min-entry-evidence-score", type=float, default=0.0,
        help="Deprecated explanation-only score floor; semantic high R/R gate controls entries.",
    )
    parser.add_argument(
        "--right-entry-priority-mode",
        choices=(
            "legacy_weighted",
            "structure_lexicographic",
        ),
        default="legacy_weighted",
        help=(
            "研究模式：同日合格右侧入口的资金优先级；默认保持原加权排序，"
            "structure_lexicographic依次比较信号层级、证据分、结构接近度和因子分"
        ),
    )
    parser.add_argument(
        "--right-entry-size-multiplier",
        type=float,
        default=1.0,
        help=(
            "研究模式：仅放大既有普通右侧首仓和加仓仓位；"
            "不改变信号、交易日和退出规则，审计基线默认3.0"
        ),
    )
    parser.add_argument(
        "--right-entry-risk-budget-pct",
        type=float,
        default=None,
        help=(
            "研究模式：普通标准右侧新campaign首仓的账户初始风险预算；"
            "例如0.008表示0.8%%，默认关闭，且不能与仓位倍数同时启用"
        ),
    )
    parser.add_argument(
        "--enable-right-inverse-volatility-sizing",
        action="store_true",
        help=(
            "研究模式：按候选中预先计算的252日逆波动乘数，仅调整普通标准"
            "右侧新campaign首仓；不生成额外买卖，默认关闭"
        ),
    )
    parser.add_argument(
        "--enable-right-market-volatility-sizing",
        action="store_true",
        help=(
            "研究模式：按候选中预先计算的20日市场实现方差乘数缩小普通"
            "右侧新campaign首仓；不放大、不再平衡，默认关闭"
        ),
    )
    parser.add_argument(
        "--price-amount-multiplier",
        type=float,
        default=1.0,
        help=(
            "行情amount转换为人民币元的显式倍数；"
            "Tushare日线amount单位为千元时使用1000"
        ),
    )
    parser.add_argument(
        "--factor-only-min-avg-amount",
        type=float,
        default=None,
        help=(
            "研究模式：纯因子右侧入口按信号日可见行情重算的"
            "最低20日平均成交额（元）；默认关闭以兼容旧回测"
        ),
    )
    parser.add_argument(
        "--max-entry-signal-amount-participation",
        type=float,
        default=None,
        help=(
            "研究模式：新右侧买单最多占信号日成交额的比例；"
            "例如0.01表示1%%，次日开盘仍只使用前一信号日成交额"
        ),
    )
    parser.add_argument(
        "--max-entry-avg-amount-participation",
        type=float,
        default=None,
        help=(
            "研究模式：新右侧买单最多占信号日可见20日均成交额的比例；"
            "例如0.05表示5%%"
        ),
    )
    parser.add_argument(
        "--profit-tail-min-return", type=float, default=0.50,
        help="最后一份趋势尾仓释放主动仓名额所需的历史最大浮盈",
    )
    parser.add_argument(
        "--profit-tranches", type=int, choices=(2, 3, 4, 5), default=5,
        help="number of profit-taking tranches; the last is reserved for maximum-profit-half",
    )
    parser.add_argument(
        "--left-grid-unit", type=float, default=0.02,
        help="single left-grid unit as account fraction; use 0 to disable new left-grid buys",
    )
    parser.add_argument(
        "--left-grid-step", type=float, default=0.05,
        help="left-grid price spacing; minimum enforced by strategy is 0.05",
    )
    parser.add_argument(
        "--left-grid-max-exposure", type=float, default=0.10,
        help="maximum left-grid exposure per symbol as account fraction",
    )
    parser.add_argument(
        "--pe-pb-left-max-positions", type=int, default=1,
        help="PE/PB研究左侧最多同时持有标的数；value_model始终优先",
    )
    parser.add_argument(
        "--pe-pb-left-max-exposure", type=float, default=0.05,
        help="PE/PB研究左侧单股最大账户仓位；默认5%%",
    )
    parser.add_argument(
        "--codes",
        help="comma-separated explicit candidate universe; overrides snapshot members",
    )
    parser.add_argument(
        "--exclude-codes",
        default="",
        help=(
            "comma-separated codes to remove from candidate snapshots for "
            "robustness review; daily executable ranks are recomputed after removal"
        ),
    )
    parser.add_argument(
        "--exit-tail-on-candidate-removal", action="store_true",
        help="exit right-side tails below 10%% at the next open after candidate removal",
    )
    parser.add_argument(
        "--candidate-mode",
        choices=("rolling", "fixed-first"),
        default="rolling",
        help="rolling逐日更新候选；fixed-first冻结首个交易日候选",
    )
    parser.add_argument(
        "--candidate-directory",
        default=str(
            PATHS.runtime_root / "backtests" / "candidate_snapshots"
            / SNAPSHOT_VERSION
        ),
    )
    parser.add_argument(
        "--formula-history",
        default=str(PATHS.runtime_root / "backtests" / "formula33_phase_research_v1.csv"),
    )
    parser.add_argument(
        "--trade-plans",
        default=str(PATHS.project_root / "config" / "trade_plans.json"),
    )
    parser.add_argument(
        "--output-directory",
        default=str(PATHS.runtime_root / "backtests" / "portfolio"),
    )
    parser.add_argument(
        "--price-kline-directory",
        default="",
        help="组合回测成交价和持仓估值使用的K线目录；严格历史回测建议使用不复权缓存",
    )
    parser.add_argument(
        "--price-frames-prepared",
        action="store_true",
        help="研究模式：输入行情已包含组合引擎所需均线、背离和复权映射特征",
    )
    parser.add_argument(
        "--require-daily-price-limits",
        action="store_true",
        help=(
            "strict research mode: require exact local Tushare stk_limit rows "
            "for every tradable execution bar"
        ),
    )
    parser.add_argument(
        "--no-price-database", action="store_true",
        help="只从 price-kline-directory 读取CSV，不使用SQLite前复权K线",
    )
    parser.add_argument(
        "--price-database-source",
        choices=("auto", "miniqmt", "akshare", "tushare"),
        default="auto",
        help="SQLite price source for backtest bars; auto uses the requested cache first, then Tushare.",
    )
    parser.add_argument(
        "--no-refresh-inputs", action="store_true",
        help=(
            "research only: skip automatic input refresh; requires --allow-frozen-inputs"
        ),
    )
    parser.add_argument(
        "--allow-frozen-inputs", action="store_true",
        help=(
            "explicitly acknowledge that this run uses already-frozen local inputs "
            "instead of first refreshing and completing source data"
        ),
    )
    parser.add_argument(
        "--refresh-price-source",
        choices=("akshare", "miniqmt", "miniqmt-akshare"),
        default="miniqmt",
        help="K-line source used by the strict pre-backtest refresh chain",
    )
    parser.add_argument(
        "--refresh-metadata-source",
        choices=("akshare", "baostock", "auto"),
        default="auto",
        help="metadata source used by Formula33 trade calendar/universe/list-date refresh",
    )
    parser.add_argument(
        "--refresh-market-cap-source",
        choices=("auto", "tushare", "akshare", "akshare-capital", "none"),
        default="auto",
        help="market-cap source used by Formula33; none is degraded research only",
    )
    parser.add_argument(
        "--no-auto-price-structure", action="store_true",
        help="disable automatic structure anchors for a sensitivity comparison",
    )
    parser.add_argument(
        "--no-structure-pullback", action="store_true",
        help="allow structure breakout orders but disable structure support pullback orders",
    )
    parser.add_argument(
        "--allow-pullback-pilot", action="store_true",
        help="兼容旧参数；支撑拉回小仓首仓现在默认允许",
    )
    parser.add_argument(
        "--disable-pullback-pilot", action="store_true",
        help="关闭领先族群支撑拉回小仓首仓，用于敏感性对照",
    )
    parser.add_argument(
        "--factor-only-breadth-reversal-only", action="store_true",
        help=(
            "研究模式：纯因子候选只启用低广度连阳一半拉回通道，"
            "其他趋势买点关闭"
        ),
    )
    parser.add_argument(
        "--enable-breadth-reversal-channel",
        action="store_true",
        help="研究模式：仅对带低广度反转通道资格标签的候选启用连阳拉回买点",
    )
    parser.add_argument(
        "--breadth-reversal-max-unique-count", type=int, default=50,
        help="信号日Formula33最近21日去重数必须严格低于该值",
    )
    parser.add_argument(
        "--breadth-reversal-risk-budget", type=float, default=0.006,
        help="每笔低广度反转交易的账户计划风险比例",
    )
    parser.add_argument(
        "--breadth-reversal-max-position", type=float, default=0.20,
        help="每只低广度反转标的的最大账户仓位比例",
    )
    parser.add_argument(
        "--breadth-reversal-max-positions", type=int, default=5,
        help="低广度反转通道允许同时持有的最大标的数",
    )
    parser.add_argument(
        "--breadth-reversal-time-limit-days", type=int, default=20,
        help="低广度反转批次的时间退出交易日数",
    )
    parser.add_argument(
        "--enable-factor-horizon-channel", action="store_true",
        help="研究模式：启用因子周期匹配通道，候选次日开盘买入并固定周期退出",
    )
    parser.add_argument(
        "--factor-horizon-risk-budget", type=float, default=0.015,
        help="每笔因子周期匹配交易的账户计划风险比例",
    )
    parser.add_argument(
        "--factor-horizon-stop-loss-pct", type=float, default=0.08,
        help="因子周期匹配交易的固定失效止损距离",
    )
    parser.add_argument(
        "--factor-horizon-max-position", type=float, default=0.20,
        help="因子周期匹配交易的单股最大账户仓位比例",
    )
    parser.add_argument(
        "--factor-horizon-max-positions", type=int, default=5,
        help="因子周期通道允许同时持有的最大标的数",
    )
    parser.add_argument(
        "--factor-horizon-holding-days", type=int, default=20,
        help="因子周期匹配交易的固定持有交易日数",
    )
    parser.add_argument(
        "--factor-horizon-trend-management-return",
        type=float,
        default=None,
        help=(
            "研究模式：因子周期仓历史最大浮盈达到该值后取消固定到期退出，"
            "转入统一分仓止盈和趋势尾仓管理，例如0.20；默认关闭"
        ),
    )
    parser.add_argument(
        "--factor-horizon-max-return-60d", type=float, default=None,
        help="研究模式：因子周期候选允许的60日收益上限，例如-0.10",
    )
    parser.add_argument(
        "--factor-horizon-require-ma20-not-rising", action="store_true",
        help="研究模式：因子周期候选必须处于MA20尚未上扬的深回撤反转阶段",
    )
    parser.add_argument(
        "--factor-horizon-exclude-watch-phase", action="store_true",
        help="研究模式：Formula33处于观察阶段时不启动因子周期交易",
    )
    parser.add_argument(
        "--factor-horizon-min-avg-amount", type=float, default=None,
        help="研究模式：因子周期候选的最低20日均成交额（元）",
    )
    parser.add_argument(
        "--factor-only-compressed-structure-only", action="store_true",
        help=(
            "研究模式：纯因子候选只启用紧结构平台延续通道，"
            "其他趋势买点关闭"
        ),
    )
    parser.add_argument(
        "--enable-compressed-structure-channel",
        action="store_true",
        help="研究模式：仅对带紧结构通道资格标签的候选启用平台延续买点",
    )
    parser.add_argument(
        "--compressed-structure-risk-budget", type=float, default=0.015,
        help="每笔紧结构平台延续交易的账户计划风险比例",
    )
    parser.add_argument(
        "--compressed-structure-stop-loss-pct", type=float, default=0.08,
        help="紧结构平台延续交易的固定保护止损距离",
    )
    parser.add_argument(
        "--compressed-structure-max-position", type=float, default=0.20,
        help="紧结构平台延续交易的单股最大账户仓位比例",
    )
    parser.add_argument(
        "--compressed-structure-max-positions", type=int, default=5,
        help="紧结构通道允许同时持有的最大标的数",
    )
    parser.add_argument(
        "--compressed-structure-holding-days", type=int, default=20,
        help="紧结构平台延续交易的固定持有交易日数",
    )
    parser.add_argument(
        "--compressed-structure-min-entry-evidence-score",
        type=float,
        default=9.0,
        help="紧结构平台延续信号的最低技术证据分",
    )
    parser.add_argument(
        "--compressed-structure-min-proximity-score",
        type=float,
        default=40.0,
        help="候选结构接近分必须严格高于该值",
    )
    parser.add_argument(
        "--compressed-structure-min-avg-amount",
        type=float,
        default=0.0,
        help="紧结构通道观察日前20日平均成交额下限；0保持旧回测兼容",
    )
    parser.add_argument(
        "--enable-intraday-right-protective-stop",
        action="store_true",
        help="研究模式：普通右侧仓在T+1后启用预设结构盘中保护止损",
    )
    parser.add_argument(
        "--enable-diversified-factor-sleeve",
        action="store_true",
        help="研究模式：纯因子右侧候选按风险预算并行持有",
    )
    parser.add_argument(
        "--diversified-factor-risk-budget", type=float, default=0.01,
        help="纯因子分散仓每笔账户计划风险比例",
    )
    parser.add_argument(
        "--diversified-factor-max-position", type=float, default=0.20,
        help="纯因子分散仓单股最大账户仓位比例",
    )
    parser.add_argument(
        "--allow-unsafe-financial", action="store_true",
        help="research only: allow candidate manifests marked financial_point_in_time=false",
    )
    parser.add_argument(
        "--allow-unsafe-industry", action="store_true",
        help="research only: allow candidate manifests marked industry_point_in_time=false",
    )
    parser.add_argument(
        "--financial-target-coverage",
        type=float,
        default=DEFAULT_FINANCIAL_TARGET_COVERAGE,
        help="minimum strict financial point-in-time coverage required before backtest",
    )
    parser.add_argument(
        "--financial-chunk-size",
        type=int,
        default=DEFAULT_FINANCIAL_CHUNK_SIZE,
        help="MiniQMT financial auto-fetch chunk size",
    )
    parser.add_argument(
        "--financial-timeout",
        type=int,
        default=DEFAULT_FINANCIAL_TIMEOUT,
        help="MiniQMT financial auto-fetch timeout per chunk in seconds",
    )
    parser.add_argument(
        "--close-confirmed-execution", choices=("close_proxy", "next_open"),
        default=DEFAULT_CLOSE_CONFIRMED_EXECUTION,
        help=(
            "daily close-confirmed signals execute at the next open by default; "
            "close_proxy is an explicit research sensitivity"
        ),
    )
    parser.add_argument("--commission-rate", type=float, default=DEFAULT_COMMISSION_RATE)
    parser.add_argument("--minimum-commission", type=float, default=DEFAULT_MINIMUM_COMMISSION)
    parser.add_argument("--initial-capital", type=float, default=1_000_000.0)
    parser.add_argument("--sell-stamp-duty-rate", type=float, default=DEFAULT_SELL_STAMP_DUTY_RATE)
    parser.add_argument(
        "--sell-stamp-duty-rate-before-2023-08-28",
        type=float,
        default=DEFAULT_SELL_STAMP_DUTY_RATE_BEFORE_2023_08_28,
        help=(
            "optional historical sell stamp duty before the 2023-08-28 "
            "reduction; use 0.001 for the statutory A-share schedule"
        ),
    )
    parser.add_argument(
        "--estimated-slippage-rate",
        type=float,
        default=DEFAULT_ESTIMATED_SLIPPAGE_RATE,
    )
    parser.add_argument(
        "--vectorbt-cross-check", action="store_true",
        help="replay emitted fills with vectorbt shared-cash accounting",
    )
    parser.add_argument("--max-symbol-exposure", type=float, default=0.70)
    parser.add_argument("--risk-free-annual", type=float, default=0.0,
                        help="研究诊断的年化无风险利率，不改变买卖逻辑")
    parser.add_argument("--trial-count", type=int,
                        help="DSR使用的已登记有效独立试验次数；缺失则不计算")
    parser.add_argument("--trial-sharpe-std-annual", type=float,
                        help="DSR使用的跨试验年化Sharpe标准差；不得由单次回测猜测")
    return configure_parser_defaults(parser)


def main(argv=None):
    return workflow.run_job(build_parser().parse_args(argv))
