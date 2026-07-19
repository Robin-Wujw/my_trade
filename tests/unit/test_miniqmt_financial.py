import json

import pytest

from stock_research.market.miniqmt_financial import (
    build_miniqmt_financial_cache,
    build_value_metrics_from_miniqmt_tables,
    financial_cache_path,
    point_in_time_financial_cache_complete,
    point_in_time_financial_cache_coverage,
)


def _summary(rows):
    return {"rows": len(rows), "columns": sorted({key for row in rows for key in row}), "sample": rows}


def test_build_value_metrics_from_miniqmt_tables_uses_announce_time_fields():
    table_map = {
        "PershareIndex": _summary([
            {
                "m_timetag": "20221231",
                "m_anntime": "20230424",
                "adjusted_earnings_per_share": 1.31,
                "adjusted_net_profit_rate": 42.5394,
                "s_fa_bps": 14.9136,
            },
            {
                "m_timetag": "20231231",
                "m_anntime": "20240422",
                "adjusted_earnings_per_share": 2.74,
                "adjusted_net_profit_rate": 104.7079,
                "s_fa_bps": 17.7635,
            },
            {
                "m_timetag": "20240331",
                "m_anntime": "20240422",
                "adjusted_earnings_per_share": float("nan"),
                "adjusted_net_profit_rate": 325.7286,
                "s_fa_bps": 19.5617,
            },
        ]),
        "Income": _summary([
            {
                "m_timetag": "20221231",
                "m_anntime": "20230424",
                "net_profit_incl_min_int_inc_after": 1_037_414_632.71,
            },
            {
                "m_timetag": "20231231",
                "m_anntime": "20240422",
                "net_profit_incl_min_int_inc_after": 2_123_669_234.59,
            },
        ]),
        "Capital": _summary([
            {
                "m_timetag": "20240331",
                "m_anntime": "20240422",
                "total_capital": 802_826_238,
            }
        ]),
    }

    metrics = build_value_metrics_from_miniqmt_tables(
        "sz.300308",
        table_map,
        report_period="2024-03-31",
        as_of_date="2024-04-30",
    )

    assert metrics["data_source"] == "miniqmt/xtdata_financial"
    assert metrics["financial_point_in_time_source"] == "announce_time"
    assert metrics["eps_excl"] == pytest.approx(2.74)
    assert metrics["yoy"] == pytest.approx(3.257286)
    assert metrics["value_line"] == pytest.approx(19.5617 + 2.74 * (1 + 3.257286) * 10)
    assert metrics["total_share"] == pytest.approx(802_826_238)
    assert metrics["announcement_date"] == "2024-04-22"


def test_build_value_metrics_keeps_loss_makers_point_in_time_complete(tmp_path):
    table_map = {
        "PershareIndex": _summary([
            {
                "m_timetag": "20211231",
                "m_anntime": "20220430",
                "adjusted_earnings_per_share": -0.42,
                "adjusted_net_profit_rate": -180.0,
                "s_fa_bps": 4.8,
            },
            {
                "m_timetag": "20221231",
                "m_anntime": "20230430",
                "adjusted_earnings_per_share": -1.20,
                "adjusted_net_profit_rate": -240.0,
                "s_fa_bps": 3.7,
            },
            {
                "m_timetag": "20240331",
                "m_anntime": "20240429",
                "adjusted_earnings_per_share": -0.32,
                "adjusted_net_profit_rate": -80.0,
                "s_fa_bps": 3.2,
            },
        ]),
        "Income": _summary([
            {
                "m_timetag": "20211231",
                "m_anntime": "20220430",
                "net_profit_incl_min_int_inc_after": -4_200_000_000,
            },
            {
                "m_timetag": "20221231",
                "m_anntime": "20230430",
                "net_profit_incl_min_int_inc_after": -12_000_000_000,
            },
        ]),
        "Capital": _summary([
            {
                "m_timetag": "20240331",
                "m_anntime": "20240429",
                "total_capital": 10_000_000_000,
            }
        ]),
    }

    metrics = build_value_metrics_from_miniqmt_tables(
        "sh.600029",
        table_map,
        report_period="2024-03-31",
        as_of_date="2024-04-30",
    )

    assert metrics is not None
    assert metrics["financial_point_in_time_source"] == "announce_time"
    assert metrics["eps_excl"] == pytest.approx(-1.20)
    assert metrics["modeled_value_line"] == pytest.approx(0.8)
    assert metrics["value_line"] == pytest.approx(3.2)
    assert metrics["value_line_policy"] == "book_value_floor_for_loss_maker"
    assert metrics["loss_maker"] is True
    assert metrics["quality_score"] < 70

    path = financial_cache_path("sh.600029", "2024-03-31", tmp_path)
    path.write_text(json.dumps(metrics), encoding="utf-8")
    assert point_in_time_financial_cache_complete(
        "sh.600029",
        "2024-03-31",
        as_of_date="2024-04-30",
        output_directory=tmp_path,
    )


def test_build_value_metrics_skips_rows_announced_after_as_of_date():
    table_map = {
        "PershareIndex": _summary([
            {
                "m_timetag": "20231231",
                "m_anntime": "20240422",
                "adjusted_earnings_per_share": 2.74,
                "adjusted_net_profit_rate": 104.7079,
                "s_fa_bps": 17.7635,
            },
            {
                "m_timetag": "20240331",
                "m_anntime": "20240501",
                "adjusted_net_profit_rate": 325.7286,
                "s_fa_bps": 19.5617,
            },
        ]),
        "Income": _summary([]),
        "Capital": _summary([
            {"m_timetag": "20240331", "m_anntime": "20240422", "total_capital": 802_826_238}
        ]),
    }

    assert build_value_metrics_from_miniqmt_tables(
        "sz.300308",
        table_map,
        report_period="2024-03-31",
        as_of_date="2024-04-30",
    ) is None


def test_build_miniqmt_financial_cache_persists_q1_value_json(monkeypatch, tmp_path):
    def fake_query(codes, tables, **kwargs):
        assert codes == ["sz.300308"]
        assert kwargs["report_type"] == "announce_time"
        assert kwargs["row_limit"] == 0
        return {
            "ok": True,
            "data": {
                "300308.SZ": {
                    "PershareIndex": _summary([
                        {
                            "m_timetag": "20231231",
                            "m_anntime": "20240422",
                            "adjusted_earnings_per_share": 2.74,
                            "adjusted_net_profit_rate": 104.7079,
                            "s_fa_bps": 17.7635,
                        },
                        {
                            "m_timetag": "20240331",
                            "m_anntime": "20240422",
                            "adjusted_net_profit_rate": 325.7286,
                            "s_fa_bps": 19.5617,
                        },
                    ]),
                    "Income": _summary([]),
                    "Capital": _summary([
                        {"m_timetag": "20240331", "m_anntime": "20240422", "total_capital": 802_826_238}
                    ]),
                }
            },
        }

    monkeypatch.setattr("stock_research.market.miniqmt_financial.query_financial_data_via_qmt_python", fake_query)

    result = build_miniqmt_financial_cache(
        ["300308"],
        report_period="2024-03-31",
        as_of_date="2024-04-30",
        output_directory=tmp_path,
    )

    path = financial_cache_path("sz.300308", "2024-03-31", tmp_path)
    assert result["saved_count"] == 1
    assert path.is_file()
    assert "miniqmt/xtdata_financial" in path.read_text(encoding="utf-8")


def test_miniqmt_financial_cache_rechecks_non_point_in_time_files(monkeypatch, tmp_path):
    good_path = financial_cache_path("sz.300308", "2024-03-31", tmp_path)
    good_path.write_text(
        """
        {
          "financial_point_in_time_source": "announce_time",
          "announcement_date": "2024-04-22",
          "value_line": 30.0,
          "quality_score": 90.0,
          "eps_excl": 2.0,
          "yoy": 0.5,
          "total_share": 800000000
        }
        """,
        encoding="utf-8",
    )
    bad_path = financial_cache_path("sz.000001", "2024-03-31", tmp_path)
    bad_path.write_text('{"value_line": 10.0}', encoding="utf-8")

    def fake_query(codes, tables, **kwargs):
        assert codes == ["sz.000001"]
        return {
            "ok": True,
            "data": {
                "000001.SZ": {
                    "PershareIndex": _summary([
                        {
                            "m_timetag": "20231231",
                            "m_anntime": "20240410",
                            "adjusted_earnings_per_share": 1.0,
                            "adjusted_net_profit_rate": 20.0,
                            "s_fa_bps": 10.0,
                        },
                        {
                            "m_timetag": "20240331",
                            "m_anntime": "20240420",
                            "adjusted_earnings_per_share": 1.1,
                            "adjusted_net_profit_rate": 30.0,
                            "s_fa_bps": 11.0,
                        },
                    ]),
                    "Income": _summary([]),
                    "Capital": _summary([
                        {"m_timetag": "20240331", "m_anntime": "20240420", "total_capital": 1_000_000_000}
                    ]),
                }
            },
        }

    monkeypatch.setattr("stock_research.market.miniqmt_financial.query_financial_data_via_qmt_python", fake_query)

    result = build_miniqmt_financial_cache(
        ["300308", "000001"],
        report_period="2024-03-31",
        as_of_date="2024-04-30",
        output_directory=tmp_path,
        missing_point_in_time_only=True,
    )

    assert result["saved_count"] == 1
    assert result["skipped_existing_count"] == 1
    assert point_in_time_financial_cache_complete(
        "sz.000001",
        "2024-03-31",
        as_of_date="2024-04-30",
        output_directory=tmp_path,
    )
    assert point_in_time_financial_cache_coverage(
        ["300308", "000001"],
        report_period="2024-03-31",
        as_of_date="2024-04-30",
        output_directory=tmp_path,
    )["coverage"] == pytest.approx(1.0)
