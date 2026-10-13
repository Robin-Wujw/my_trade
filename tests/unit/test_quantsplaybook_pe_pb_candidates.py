import pandas as pd

from stock_research.research.quantsplaybook_pe_pb_candidates import (
    PointInTimeFinancialResolver,
    PointInTimeIndustryResolver,
    _cached_frame,
    _finalize_row,
    deduplicate_market_rows,
    point_in_time_tradable_rows,
)


def test_financial_resolver_hides_results_until_announcement_date():
    history = pd.DataFrame([
        {
            "ts_code": "000001.SZ",
            "announcement_date": pd.Timestamp("2024-04-30"),
            "report_period": pd.Timestamp("2024-03-31"),
            "quality_score": 80.0,
            "earnings_yoy": 0.2,
        },
    ])
    resolver = PointInTimeFinancialResolver(history)

    assert resolver.latest("000001.SZ", "2024-04-29") == {}
    assert resolver.latest("000001.SZ", "2024-04-30")["quality_score"] == 80.0


def test_financial_resolver_old_period_revision_does_not_replace_newer_period():
    history = pd.DataFrame([
        {
            "ts_code": "000001.SZ",
            "announcement_date": pd.Timestamp("2024-04-30"),
            "report_period": pd.Timestamp("2024-03-31"),
            "quality_score": 80.0,
            "earnings_yoy": 0.2,
        },
        {
            "ts_code": "000001.SZ",
            "announcement_date": pd.Timestamp("2024-08-30"),
            "report_period": pd.Timestamp("2024-06-30"),
            "quality_score": 70.0,
            "earnings_yoy": 0.1,
        },
        {
            "ts_code": "000001.SZ",
            "announcement_date": pd.Timestamp("2024-09-10"),
            "report_period": pd.Timestamp("2024-03-31"),
            "quality_score": 10.0,
            "earnings_yoy": -0.5,
        },
    ])
    resolver = PointInTimeFinancialResolver(history)

    visible = resolver.latest("000001.SZ", "2024-09-10")

    assert visible["report_period"] == pd.Timestamp("2024-06-30")
    assert visible["quality_score"] == 70.0


def test_industry_resolver_uses_exclusive_out_date_and_switches_history():
    history = pd.DataFrame([
        {
            "ts_code": "000001.SZ",
            "industry": "银行",
            "in_date": pd.Timestamp("2020-01-01"),
            "out_date": pd.Timestamp("2024-01-03"),
        },
        {
            "ts_code": "000001.SZ",
            "industry": "资本市场服务",
            "in_date": pd.Timestamp("2024-01-03"),
            "out_date": pd.NaT,
        },
    ])
    resolver = PointInTimeIndustryResolver(history)

    resolver.advance("2024-01-02")
    assert resolver.industry("000001.SZ") == "银行"
    resolver.advance("2024-01-03")
    assert resolver.industry("000001.SZ") == "资本市场服务"


def test_tradable_rows_enforce_listing_bar_delisting_and_dated_st():
    frame = pd.DataFrame([
        {
            "ts_code": code,
            "open": 10.0,
            "high": 10.2,
            "low": 9.8,
            "kline_close": 10.0,
            "volume": volume,
            "amount": volume * 10,
        }
        for code, volume in (
            ("OLD.SZ", 1000),
            ("NEW.SZ", 1000),
            ("DELISTED.SZ", 1000),
            ("SUSPENDED.SZ", 0),
            ("ST.SZ", 1000),
        )
    ])
    basic = pd.DataFrame([
        {
            "ts_code": code,
            "list_date": list_date,
            "delist_date": delist_date,
        }
        for code, list_date, delist_date in (
            ("OLD.SZ", pd.Timestamp("2020-01-01"), pd.NaT),
            ("NEW.SZ", pd.Timestamp("2024-01-01"), pd.NaT),
            ("DELISTED.SZ", pd.Timestamp("2020-01-01"), pd.Timestamp("2024-01-01")),
            ("SUSPENDED.SZ", pd.Timestamp("2020-01-01"), pd.NaT),
            ("ST.SZ", pd.Timestamp("2020-01-01"), pd.NaT),
        )
    ])

    result = point_in_time_tradable_rows(
        frame,
        basic,
        observation_date="2024-01-03",
        minimum_listing_days=300,
        st_set={("ST.SZ", "2024-01-03")},
    )

    assert result["ts_code"].tolist() == ["OLD.SZ"]


def test_pe_pb_value_line_is_historical_multiple_price_anchor():
    row = _finalize_row({
        "method": "PE",
        "date": "2024-01-03",
        "code": "000001.SZ",
        "name": "测试",
        "industry": "银行",
        "close": 10.0,
        "ratio": 0.5,
        "low_avg": 8.0,
        "quality_score": 80.0,
        "earnings_yoy": 0.2,
        "mktcap": 200.0,
        "candidate_score": 90.0,
    })

    assert row["value_line"] == 20.0
    assert row["price_to_value"] == 0.5


def test_cached_frame_reuses_only_matching_database_fingerprint(tmp_path):
    database = tmp_path / "source.sqlite3"
    database.write_bytes(b"version-one")
    calls = []

    def builder(_):
        calls.append(True)
        return pd.DataFrame({"value": [len(calls)]})

    first, first_evidence = _cached_frame(
        db=database,
        cache_directory=tmp_path / "cache",
        name="test",
        builder=builder,
    )
    second, second_evidence = _cached_frame(
        db=database,
        cache_directory=tmp_path / "cache",
        name="test",
        builder=builder,
    )
    database.write_bytes(b"version-two-is-different")
    third, third_evidence = _cached_frame(
        db=database,
        cache_directory=tmp_path / "cache",
        name="test",
        builder=builder,
    )

    assert first["value"].tolist() == [1]
    assert second["value"].tolist() == [1]
    assert third["value"].tolist() == [2]
    assert first_evidence["cache_hit"] is False
    assert second_evidence["cache_hit"] is True
    assert third_evidence["cache_hit"] is False
    assert len(calls) == 2


def test_market_row_deduplication_is_deterministic_and_reports_conflicts():
    frame = pd.DataFrame([
        {
            "ts_code": "000001.SZ",
            "trade_date": "2024-01-03",
            "row_key": "old",
            "updated_at": "2024-01-03T18:00:00Z",
            "close": 10.0,
        },
        {
            "ts_code": "000001.SZ",
            "trade_date": "2024-01-03",
            "row_key": "new",
            "updated_at": "2024-01-04T18:00:00Z",
            "close": 10.5,
        },
        {
            "ts_code": "000002.SZ",
            "trade_date": "2024-01-03",
            "row_key": "only",
            "updated_at": "2024-01-03T18:00:00Z",
            "close": 8.0,
        },
    ])

    result, audit = deduplicate_market_rows(
        frame,
        value_columns=("close",),
    )

    assert result.set_index("ts_code").loc["000001.SZ", "close"] == 10.5
    assert audit["extra_rows"] == 1
    assert audit["conflicting_keys"] == 1
    assert audit["resolution"] == "latest_updated_at_then_row_key"
