"""Zer0share-inspired Tushare dataset manifest for the unified SQLite cache."""
from __future__ import annotations

from dataclasses import dataclass, field


INDEX_DAILY_CODES = (
    "000001.SH",
    "399001.SZ",
    "000016.SH",
    "000300.SH",
    "000905.SH",
    "000852.SH",
    "000985.SH",
    "399006.SZ",
    "000688.SH",
    "399005.SZ",
    "000922.SH",
)

INDEX_WEIGHT_CODES = ("399300.SZ", "000905.SH", "000852.SH")
TRADE_CAL_EXCHANGES = ("SSE", "SZSE", "CFFEX", "DCE", "SHFE", "CZCE", "INE", "GFEX")
FUTURES_EXCHANGES = ("CZCE", "SHFE", "DCE", "CFFEX", "INE", "GFEX")
FUT_INDEX_CODES = ("NHCI.NH", "NHAI.NH", "NHMI.NH")
OPTIONS_EXCHANGES = ("SSE", "SZSE", "CFFEX", "DCE", "SHFE", "CZCE")
ETF_SHARE_EXCHANGES = ("SSE", "SZSE")
SW_VERSIONS = ("SW2014", "SW2021")
SW_LEVELS = ("L1", "L2", "L3")


@dataclass(frozen=True)
class Zer0shareDataset:
    """A single dataset sync recipe from zer0share's public data catalog."""

    name: str
    api_name: str
    group: str
    fields: tuple[str, ...]
    mode: str
    first_date: str | None = None
    date_param: str = "trade_date"
    params: dict[str, str] = field(default_factory=dict)
    loops: dict[str, tuple[str, ...]] = field(default_factory=dict)
    pagination: bool = False
    calendar: str = "trading"
    enabled: bool = True
    note: str = ""

    @property
    def fields_arg(self) -> str:
        return ",".join(self.fields)


BASIC_COLS = (
    "ts_code", "symbol", "name", "area", "industry", "fullname", "enname",
    "cnspell", "market", "exchange", "curr_type", "list_status", "list_date",
    "delist_date", "is_hs", "act_name", "act_ent_type",
)
DAILY_COLS = (
    "ts_code", "trade_date", "open", "high", "low", "close", "pre_close",
    "change", "pct_chg", "vol", "amount",
)
TRADE_CAL_COLS = ("exchange", "cal_date", "is_open", "pretrade_date")
ADJ_FACTOR_COLS = ("ts_code", "trade_date", "adj_factor")
DAILY_BASIC_COLS = (
    "ts_code", "trade_date", "close", "turnover_rate", "turnover_rate_f",
    "volume_ratio", "pe", "pe_ttm", "pb", "ps", "ps_ttm", "dv_ratio",
    "dv_ttm", "total_share", "float_share", "free_share", "total_mv",
    "circ_mv", "limit_status",
)
STOCK_ST_COLS = ("ts_code", "name", "trade_date", "type", "type_name")
SUSPEND_D_COLS = ("ts_code", "trade_date", "suspend_timing", "suspend_type")
STK_LIMIT_COLS = ("trade_date", "ts_code", "pre_close", "up_limit", "down_limit")
INDEX_WEIGHT_COLS = ("index_code", "con_code", "trade_date", "weight")
INDEX_DAILY_COLS = DAILY_COLS
IDX_ANNS_COLS = ("ann_date", "title", "url", "source", "type")
ETF_BASIC_COLS = (
    "ts_code", "csname", "extname", "cname", "index_code", "index_name",
    "setup_date", "list_date", "list_status", "exchange", "mgr_name",
    "custod_name", "mgt_fee", "etf_type",
)
ETF_INDEX_COLS = (
    "ts_code", "indx_name", "indx_csname", "pub_party_name", "pub_date",
    "base_date", "bp", "adj_circle",
)
FUND_DAILY_COLS = DAILY_COLS
FUND_ADJ_COLS = ("ts_code", "trade_date", "adj_factor", "discount_rate")
ETF_SHARE_SIZE_COLS = (
    "trade_date", "ts_code", "etf_name", "total_share", "total_size", "nav",
    "close", "exchange",
)
ETF_SH_CONS_COLS = (
    "trade_date", "ts_code", "con_code", "con_name", "qty", "sub_flag", "cpr",
    "rdr", "sca", "exchange",
)
SW_CLASSIFY_COLS = (
    "index_code", "industry_name", "level", "parent_code", "industry_code",
    "is_pub", "src",
)
SW_MEMBER_COLS = (
    "l1_code", "l1_name", "l2_code", "l2_name", "l3_code", "l3_name",
    "ts_code", "name", "in_date", "out_date", "is_new",
)
CI_MEMBER_COLS = SW_MEMBER_COLS
SW_DAILY_COLS = (
    "ts_code", "trade_date", "name", "open", "high", "low", "close", "change",
    "pct_change", "vol", "amount", "pe", "pb", "float_mv", "total_mv",
)
FUT_BASIC_COLS = (
    "ts_code", "symbol", "exchange", "name", "fut_code", "multiplier",
    "trade_unit", "per_unit", "quote_unit", "quote_unit_desc", "d_mode_desc",
    "list_date", "delist_date", "d_month", "last_ddate", "trade_time_desc",
)
FUT_DAILY_COLS = (
    "ts_code", "trade_date", "pre_close", "pre_settle", "open", "high", "low",
    "close", "settle", "change1", "change2", "vol", "amount", "oi", "oi_chg",
    "delv_settle",
)
FUT_HOLDING_COLS = (
    "trade_date", "symbol", "broker", "vol", "vol_chg", "long_hld",
    "long_chg", "short_hld", "short_chg", "exchange",
)
FUT_WSR_COLS = (
    "trade_date", "symbol", "fut_name", "warehouse", "wh_id", "pre_vol", "vol",
    "vol_chg", "area", "year", "grade", "brand", "place", "pd", "is_ct",
    "unit", "exchange",
)
FUT_SETTLE_COLS = (
    "ts_code", "trade_date", "settle", "trading_fee_rate", "trading_fee",
    "delivery_fee", "b_hedging_margin_rate", "s_hedging_margin_rate",
    "long_margin_rate", "short_margin_rate", "offset_today_fee", "exchange",
)
FUT_MAPPING_COLS = ("ts_code", "trade_date", "mapping_ts_code")
FT_LIMIT_COLS = (
    "trade_date", "ts_code", "name", "up_limit", "down_limit", "m_ratio",
    "cont", "exchange",
)
FUT_WEEKLY_COLS = (
    "ts_code", "trade_date", "freq", "open", "high", "low", "close",
    "pre_close", "settle", "pre_settle", "vol", "amount", "oi", "oi_chg",
    "exchange", "change1", "change2",
)
FUT_INDEX_DAILY_COLS = (
    "ts_code", "trade_date", "close", "open", "high", "low", "pre_close",
    "change", "pct_chg", "vol", "amount",
)
FUT_WEEKLY_DETAIL_COLS = (
    "exchange", "prd", "name", "vol", "vol_yoy", "amount", "amout_yoy",
    "cumvol", "cumvol_yoy", "cumamt", "cumamt_yoy", "open_interest",
    "interest_wow", "mc_close", "close_wow", "week", "week_date",
)
OPT_BASIC_COLS = (
    "ts_code", "symbol", "exchange", "name", "per_unit", "opt_code",
    "opt_type", "call_put", "exercise_type", "exercise_price", "s_month",
    "maturity_date", "list_price", "list_date", "delist_date", "last_edate",
    "last_ddate", "quote_unit", "min_price_chg",
)
OPT_DAILY_COLS = (
    "ts_code", "trade_date", "exchange", "pre_settle", "pre_close", "open",
    "high", "low", "close", "settle", "vol", "amount", "oi",
)


ZER0SHARE_DATASETS = (
    Zer0shareDataset("trade_cal", "trade_cal", "stock", TRADE_CAL_COLS, "range", first_date="19900101", date_param="cal_date", loops={"exchange": TRADE_CAL_EXCHANGES}, calendar="calendar"),
    Zer0shareDataset("basic", "stock_basic", "stock", BASIC_COLS, "snapshot", params={"exchange": "", "list_status": "L,D,P,G"}),
    Zer0shareDataset("daily_kline", "daily", "stock", DAILY_COLS, "daily", first_date="19901219"),
    Zer0shareDataset("adj_factor", "adj_factor", "stock", ADJ_FACTOR_COLS, "daily", first_date="19901219"),
    Zer0shareDataset("daily_basic", "daily_basic", "stock", DAILY_BASIC_COLS, "daily", first_date="19901219", params={"ts_code": ""}),
    Zer0shareDataset("stock_st", "stock_st", "stock", STOCK_ST_COLS, "daily", first_date="20000104"),
    Zer0shareDataset("suspend_d", "suspend_d", "stock", SUSPEND_D_COLS, "daily", first_date="20000104", params={"suspend_type": "S"}),
    Zer0shareDataset("stk_limit", "stk_limit", "stock", STK_LIMIT_COLS, "daily", first_date="20070104"),
    Zer0shareDataset("index_weight", "index_weight", "stock", INDEX_WEIGHT_COLS, "monthly", first_date="20050101", loops={"index_code": INDEX_WEIGHT_CODES}),
    Zer0shareDataset("index_daily", "index_daily", "stock", INDEX_DAILY_COLS, "range", first_date="19901219", loops={"ts_code": INDEX_DAILY_CODES}),
    Zer0shareDataset("idx_anns", "idx_anns", "stock", IDX_ANNS_COLS, "daily", first_date="20040101", date_param="ann_date", calendar="natural", pagination=True),
    Zer0shareDataset("etf_basic", "etf_basic", "etf", ETF_BASIC_COLS, "snapshot"),
    Zer0shareDataset("etf_index", "etf_index", "etf", ETF_INDEX_COLS, "snapshot"),
    Zer0shareDataset("fund_daily", "fund_daily", "etf", FUND_DAILY_COLS, "daily", first_date="20050223"),
    Zer0shareDataset("fund_adj", "fund_adj", "etf", FUND_ADJ_COLS, "daily", first_date="20100101"),
    Zer0shareDataset("etf_share_size", "etf_share_size", "etf", ETF_SHARE_SIZE_COLS, "daily", first_date="20100101", loops={"exchange": ETF_SHARE_EXCHANGES}),
    Zer0shareDataset("etf_sh_cons", "etf_sh_cons", "etf", ETF_SH_CONS_COLS, "daily", first_date="20100101", pagination=True),
    Zer0shareDataset("sw_classify", "index_classify", "industry", SW_CLASSIFY_COLS, "snapshot", loops={"src": SW_VERSIONS, "level": SW_LEVELS}),
    Zer0shareDataset("sw_member", "index_member_all", "industry", SW_MEMBER_COLS, "derived_snapshot"),
    Zer0shareDataset("ci_member", "ci_index_member", "industry", CI_MEMBER_COLS, "derived_snapshot"),
    Zer0shareDataset("sw_daily", "sw_daily", "industry", SW_DAILY_COLS, "range", first_date="20000104", params={"ts_code": ""}),
    Zer0shareDataset("fut_basic", "fut_basic", "futures", FUT_BASIC_COLS, "snapshot", loops={"exchange": FUTURES_EXCHANGES, "fut_type": ("1", "2")}),
    Zer0shareDataset("fut_daily", "fut_daily", "futures", FUT_DAILY_COLS, "daily", first_date="19950417"),
    Zer0shareDataset("fut_holding", "fut_holding", "futures", FUT_HOLDING_COLS, "daily", first_date="20020101"),
    Zer0shareDataset("fut_wsr", "fut_wsr", "futures", FUT_WSR_COLS, "daily", first_date="20070101"),
    Zer0shareDataset("fut_settle", "fut_settle", "futures", FUT_SETTLE_COLS, "daily", first_date="20120101"),
    Zer0shareDataset("fut_mapping", "fut_mapping", "futures", FUT_MAPPING_COLS, "daily", first_date="19950417"),
    Zer0shareDataset("ft_limit", "ft_limit", "futures", FT_LIMIT_COLS, "daily", first_date="20050101"),
    Zer0shareDataset("fut_weekly", "fut_weekly_monthly", "futures", FUT_WEEKLY_COLS, "daily", first_date="19950417", params={"freq": "week"}),
    Zer0shareDataset("fut_monthly", "fut_weekly_monthly", "futures", FUT_WEEKLY_COLS, "daily", first_date="19950417", params={"freq": "month"}),
    Zer0shareDataset("fut_index_daily", "fut_index_daily", "futures", FUT_INDEX_DAILY_COLS, "daily", first_date="20060101", loops={"ts_code": FUT_INDEX_CODES}),
    Zer0shareDataset("fut_weekly_detail", "fut_weekly_detail", "futures", FUT_WEEKLY_DETAIL_COLS, "weekly", first_date="20151201", date_param="week"),
    Zer0shareDataset("opt_basic", "opt_basic", "options", OPT_BASIC_COLS, "snapshot", loops={"exchange": OPTIONS_EXCHANGES}),
    Zer0shareDataset("opt_daily", "opt_daily", "options", OPT_DAILY_COLS, "daily", first_date="20150209", loops={"exchange": OPTIONS_EXCHANGES}),
    Zer0shareDataset("ricequant_basic", "ricequant_basic", "ricequant", (), "external", enabled=False, note="Requires RiceQuant credentials."),
    Zer0shareDataset("ricequant_stock_minute", "ricequant_stock_minute", "ricequant", (), "external", enabled=False, note="Requires RiceQuant credentials."),
    Zer0shareDataset("ricequant_etf_basic", "ricequant_etf_basic", "ricequant", (), "external", enabled=False, note="Requires RiceQuant credentials."),
    Zer0shareDataset("ricequant_etf_minute", "ricequant_etf_minute", "ricequant", (), "external", enabled=False, note="Requires RiceQuant credentials."),
)

ZER0SHARE_DATASET_MAP = {item.name: item for item in ZER0SHARE_DATASETS}


def enabled_datasets() -> tuple[Zer0shareDataset, ...]:
    return tuple(item for item in ZER0SHARE_DATASETS if item.enabled)
