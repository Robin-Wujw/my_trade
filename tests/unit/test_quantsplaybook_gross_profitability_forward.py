import math

import pandas as pd
import pytest

from stock_research.research.quantsplaybook_gross_profitability_forward import (
    FINANCIAL_SOURCE_DATASETS,
    GROSS_PROFITABILITY_FACTOR,
    SOURCE_FACTOR,
    add_gross_profitability_combination,
    build_parser,
    calculate_forward_gross_profitability_factor,
    fixed_gross_profitability_combination_spec,
)


def _index(codes, dates):
    return pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp(date), code)
            for date in dates
            for code in codes
        ],
        names=["date", "code"],
    )


def _income(
    code,
    *,
    period,
    captured,
    announced=None,
    revenue=100.0,
    cost=60.0,
    suffix="a",
):
    return {
        "ts_code": code,
        "ann_date": pd.Timestamp(announced or captured).strftime("%Y%m%d"),
        "end_date": pd.Timestamp(period).strftime("%Y%m%d"),
        "report_type": "1",
        "total_revenue": revenue,
        "oper_cost": cost,
        "_captured_at": captured,
        "_payload_sha256": f"income-{code}-{period}-{suffix}",
    }


def _balance(code, *, period, captured, assets=100.0, suffix="a"):
    return {
        "ts_code": code,
        "ann_date": pd.Timestamp(captured).strftime("%Y%m%d"),
        "end_date": pd.Timestamp(period).strftime("%Y%m%d"),
        "report_type": "1",
        "total_assets": assets,
        "_captured_at": captured,
        "_payload_sha256": f"balance-{code}-{period}-{suffix}",
    }


def _industry(
    code,
    *,
    captured="2026-08-11",
    name="医药生物",
    out_date=None,
):
    return {
        "ts_code": code,
        "l1_name": name,
        "l2_name": "",
        "l3_name": "",
        "in_date": "20210101",
        "out_date": out_date,
        "_updated_at": captured,
    }


def _base_panel(index):
    values = range(1, len(index) + 1)
    return pd.DataFrame({
        "network_cc": values,
        "ma_convergence": values,
        "disposition_reversal": values,
        "high_quality_momentum": values,
        "coin_team": values,
        "low_market_residual_volatility_20": values,
    }, index=index)


def test_forward_gpa_defaults_to_isolated_eastmoney_version_datasets():
    args = build_parser().parse_args([
        "--formula-history", "formula.csv",
        "--start-date", "2026-08-11",
        "--end-date", "2026-08-11",
    ])

    assert args.financial_source == "eastmoney"
    assert FINANCIAL_SOURCE_DATASETS[args.financial_source] == (
        "eastmoney_income", "eastmoney_balancesheet",
    )


def test_gpa_waits_until_both_statements_have_been_captured():
    code = "600001.SH"
    index = _index(["sh.600001"], ["2026-08-12", "2026-08-13"])
    income = pd.DataFrame([
        _income(code, period="2025-12-31", captured="2026-08-12 09:00:00"),
    ])
    balance = pd.DataFrame([
        _balance(code, period="2025-12-31", captured="2026-08-13 09:00:00"),
    ])

    factor = calculate_forward_gross_profitability_factor(
        index, income, balance, pd.DataFrame([_industry(code)]),
    )

    assert math.isnan(factor.iloc[0])
    assert factor.iloc[1] == pytest.approx(0.4)


def test_gpa_effective_date_is_not_before_provider_announcement():
    code = "600001.SH"
    index = _index(["sh.600001"], ["2026-08-13", "2026-08-14"])
    income = pd.DataFrame([
        _income(
            code,
            period="2025-12-31",
            captured="2026-08-12 09:00:00",
            announced="2026-08-14",
        ),
    ])
    balance = pd.DataFrame([
        _balance(code, period="2025-12-31", captured="2026-08-12 09:00:00"),
    ])

    factor = calculate_forward_gross_profitability_factor(
        index, income, balance, pd.DataFrame([_industry(code)]),
    )

    assert math.isnan(factor.iloc[0])
    assert factor.iloc[1] == pytest.approx(0.4)


def test_utc_capture_after_shanghai_midnight_belongs_to_next_market_date():
    code = "600001.SH"
    index = _index(["sh.600001"], ["2026-08-11", "2026-08-12"])
    income = pd.DataFrame([
        _income(
            code,
            period="2025-12-31",
            captured="2026-08-11 16:30:00+00:00",
            announced="2026-08-11",
        ),
    ])
    balance = pd.DataFrame([
        _balance(
            code,
            period="2025-12-31",
            captured="2026-08-11 16:30:00+00:00",
        ),
    ])

    factor = calculate_forward_gross_profitability_factor(
        index, income, balance, pd.DataFrame([_industry(code)]),
    )

    assert math.isnan(factor.iloc[0])
    assert factor.iloc[1] == pytest.approx(0.4)


def test_gpa_revision_starts_on_capture_date_without_backfill():
    code = "600001.SH"
    index = _index(["sh.600001"], ["2026-08-13", "2026-08-14"])
    original = _income(
        code,
        period="2025-12-31",
        captured="2026-08-12 09:00:00",
        revenue=100.0,
        cost=60.0,
    )
    revised = _income(
        code,
        period="2025-12-31",
        captured="2026-08-14 09:00:00",
        revenue=100.0,
        cost=80.0,
        suffix="revision",
    )
    balance = pd.DataFrame([
        _balance(code, period="2025-12-31", captured="2026-08-12 09:00:00"),
    ])
    industry = pd.DataFrame([_industry(code)])

    before_revision = calculate_forward_gross_profitability_factor(
        index, pd.DataFrame([original]), balance, industry,
    )
    with_revision = calculate_forward_gross_profitability_factor(
        index, pd.DataFrame([original, revised]), balance, industry,
    )

    assert before_revision.iloc[0] == pytest.approx(0.4)
    assert with_revision.iloc[0] == pytest.approx(0.4)
    assert with_revision.iloc[1] == pytest.approx(0.2)


def test_same_capture_prefers_provider_update_flag_revision():
    code = "600001.SH"
    index = _index(["sh.600001"], ["2026-08-13"])
    original = _income(
        code,
        period="2025-12-31",
        captured="2026-08-12 09:00:00",
        revenue=100.0,
        cost=60.0,
        suffix="original",
    )
    original["update_flag"] = "0"
    revised = _income(
        code,
        period="2025-12-31",
        captured="2026-08-12 09:00:00",
        revenue=100.0,
        cost=80.0,
        suffix="revised",
    )
    revised["update_flag"] = "1"
    balance = pd.DataFrame([
        _balance(code, period="2025-12-31", captured="2026-08-12 09:00:00"),
    ])

    factor = calculate_forward_gross_profitability_factor(
        index,
        pd.DataFrame([revised, original]),
        balance,
        pd.DataFrame([_industry(code)]),
    )

    assert factor.iloc[0] == pytest.approx(0.2)


def test_final_announcement_date_is_a_conservative_availability_bound():
    code = "600001.SH"
    index = _index(["sh.600001"], ["2026-08-13", "2026-08-14"])
    income_row = _income(
        code,
        period="2025-12-31",
        captured="2026-08-12",
        announced="2026-08-12",
    )
    income_row["f_ann_date"] = "20260814"
    balance = pd.DataFrame([
        _balance(code, period="2025-12-31", captured="2026-08-12"),
    ])

    factor = calculate_forward_gross_profitability_factor(
        index,
        pd.DataFrame([income_row]),
        balance,
        pd.DataFrame([_industry(code)]),
    )

    assert math.isnan(factor.iloc[0])
    assert factor.iloc[1] == pytest.approx(0.4)


def test_latest_annual_report_missing_field_fails_closed_instead_of_falling_back():
    code = "600001.SH"
    index = _index(["sh.600001"], ["2026-08-13", "2026-08-14"])
    income = pd.DataFrame([
        _income(
            code,
            period="2024-12-31",
            captured="2026-08-12 08:00:00",
            revenue=100.0,
            cost=60.0,
        ),
        _income(
            code,
            period="2025-12-31",
            captured="2026-08-14 08:00:00",
            revenue=120.0,
            cost=None,
            suffix="missing-cost",
        ),
        {
            **_income(
                code,
                period="2026-09-30",
                captured="2026-08-12 08:00:00",
                revenue=999.0,
                cost=1.0,
                suffix="quarterly",
            ),
            "report_type": "1",
        },
    ])
    balance = pd.DataFrame([
        _balance(code, period="2024-12-31", captured="2026-08-12 08:00:00"),
        _balance(code, period="2025-12-31", captured="2026-08-14 08:00:00"),
    ])

    factor = calculate_forward_gross_profitability_factor(
        index, income, balance, pd.DataFrame([_industry(code)]),
    )

    assert factor.iloc[0] == pytest.approx(0.4)
    assert math.isnan(factor.iloc[1])


def test_new_annual_report_single_statement_blocks_old_report_fallback():
    code = "600001.SH"
    index = _index(["sh.600001"], ["2026-08-13", "2026-08-14"])
    income = pd.DataFrame([
        _income(
            code,
            period="2024-12-31",
            captured="2026-08-12",
        ),
        _income(
            code,
            period="2025-12-31",
            captured="2026-08-14",
            revenue=120.0,
            cost=60.0,
            suffix="new-report-income-only",
        ),
    ])
    balance = pd.DataFrame([
        _balance(code, period="2024-12-31", captured="2026-08-12"),
    ])

    factor = calculate_forward_gross_profitability_factor(
        index, income, balance, pd.DataFrame([_industry(code)]),
    )

    assert factor.iloc[0] == pytest.approx(0.4)
    assert math.isnan(factor.iloc[1])


def test_industry_out_date_revision_replaces_old_open_membership():
    code = "600001.SH"
    index = _index(["sh.600001"], ["2026-08-13", "2026-08-14"])
    income = pd.DataFrame([
        _income(code, period="2025-12-31", captured="2026-08-12"),
    ])
    balance = pd.DataFrame([
        _balance(code, period="2025-12-31", captured="2026-08-12"),
    ])
    industry = pd.DataFrame([
        _industry(code, captured="2026-08-12"),
        _industry(
            code,
            captured="2026-08-14",
            out_date="20260813",
        ),
    ])

    factor = calculate_forward_gross_profitability_factor(
        index, income, balance, industry,
    )

    assert factor.iloc[0] == pytest.approx(0.4)
    assert math.isnan(factor.iloc[1])


def test_same_capture_conflicting_industry_membership_fails_closed():
    code = "600001.SH"
    index = _index(["sh.600001"], ["2026-08-13"])
    income = pd.DataFrame([
        _income(code, period="2025-12-31", captured="2026-08-12"),
    ])
    balance = pd.DataFrame([
        _balance(code, period="2025-12-31", captured="2026-08-12"),
    ])
    industry = pd.DataFrame([
        _industry(code, captured="2026-08-12", out_date=None),
        _industry(code, captured="2026-08-12", out_date="20260811"),
    ])

    factor = calculate_forward_gross_profitability_factor(
        index, income, balance, industry,
    )

    assert math.isnan(factor.iloc[0])


def test_late_revision_of_older_year_does_not_replace_newer_annual_report():
    code = "600001.SH"
    index = _index(["sh.600001"], ["2026-08-13", "2026-08-14"])
    income = pd.DataFrame([
        _income(
            code,
            period="2024-12-31",
            captured="2026-08-12",
            revenue=100.0,
            cost=60.0,
        ),
        _income(
            code,
            period="2025-12-31",
            captured="2026-08-13",
            revenue=100.0,
            cost=70.0,
        ),
        _income(
            code,
            period="2024-12-31",
            captured="2026-08-14",
            revenue=100.0,
            cost=10.0,
            suffix="late-old-revision",
        ),
    ])
    balance = pd.DataFrame([
        _balance(code, period="2024-12-31", captured="2026-08-12"),
        _balance(code, period="2025-12-31", captured="2026-08-13"),
    ])

    factor = calculate_forward_gross_profitability_factor(
        index, income, balance, pd.DataFrame([_industry(code)]),
    )

    assert factor.iloc[0] == pytest.approx(0.3)
    assert factor.iloc[1] == pytest.approx(0.3)


def test_gpa_excludes_financial_missing_industry_and_nonpositive_assets():
    codes = ["600001.SH", "600002.SH", "600003.SH", "600004.SH"]
    local_codes = ["sh.600001", "sh.600002", "sh.600003", "sh.600004"]
    index = _index(local_codes, ["2026-08-13"])
    income = pd.DataFrame([
        _income(code, period="2025-12-31", captured="2026-08-12")
        for code in codes
    ])
    balance = pd.DataFrame([
        _balance(
            code,
            period="2025-12-31",
            captured="2026-08-12",
            assets=0.0 if code == "600004.SH" else 100.0,
        )
        for code in codes
    ])
    industry = pd.DataFrame([
        _industry("600001.SH"),
        _industry("600002.SH", name="银行"),
        _industry("600004.SH"),
    ])

    factor = calculate_forward_gross_profitability_factor(
        index, income, balance, industry,
    )

    assert factor.loc[(pd.Timestamp("2026-08-13"), "sh.600001")] == pytest.approx(0.4)
    assert factor.drop(index=(pd.Timestamp("2026-08-13"), "sh.600001")).isna().all()


def test_fixed_gpa_combination_is_equal_weight_and_requires_all_seven_factors():
    index = _index(["sh.600001", "sh.600002"], ["2026-08-13"])
    panel = _base_panel(index)
    income = pd.DataFrame([
        _income(code, period="2025-12-31", captured="2026-08-12")
        for code in ("600001.SH", "600002.SH")
    ])
    balance = pd.DataFrame([
        _balance(code, period="2025-12-31", captured="2026-08-12")
        for code in ("600001.SH", "600002.SH")
    ])
    industry = pd.DataFrame([
        _industry("600001.SH"), _industry("600002.SH"),
    ])
    average_amount = pd.Series(100_000_000.0, index=index)

    combined, coverage = add_gross_profitability_combination(
        panel, income, balance, industry, average_amount,
    )
    spec = fixed_gross_profitability_combination_spec()

    assert set(spec["weights"]) == set(spec["selected_factors"])
    assert all(weight == pytest.approx(1 / 7) for weight in spec["weights"].values())
    assert combined[GROSS_PROFITABILITY_FACTOR].notna().all()
    assert combined[SOURCE_FACTOR].notna().all()
    assert coverage["combination_available_rows"] == 2

    panel.loc[index[0], "network_cc"] = None
    incomplete, _ = add_gross_profitability_combination(
        panel, income, balance, industry, average_amount,
    )
    assert math.isnan(incomplete.loc[index[0], SOURCE_FACTOR])


def test_combination_ranks_only_complete_liquid_seven_factor_universe():
    local_codes = [f"sh.60000{number}" for number in range(1, 5)]
    tushare_codes = [f"60000{number}.SH" for number in range(1, 5)]
    index = _index(local_codes, ["2026-08-13"])
    panel = _base_panel(index).astype(float)
    excluded_liquidity = index[2]
    excluded_incomplete = index[3]
    panel.loc[excluded_liquidity, :] = 1000.0
    panel.loc[excluded_incomplete, :] = 2000.0
    panel.loc[excluded_incomplete, "network_cc"] = None
    income = pd.DataFrame([
        _income(code, period="2025-12-31", captured="2026-08-12")
        for code in tushare_codes
    ])
    balance = pd.DataFrame([
        _balance(code, period="2025-12-31", captured="2026-08-12")
        for code in tushare_codes
    ])
    industry = pd.DataFrame([_industry(code) for code in tushare_codes])
    average_amount = pd.Series(100_000_000.0, index=index)
    average_amount.loc[excluded_liquidity] = 10_000_000.0

    combined, coverage = add_gross_profitability_combination(
        panel, income, balance, industry, average_amount,
    )
    eligible_index = index[:2]
    eligible_only, _ = add_gross_profitability_combination(
        panel.loc[eligible_index],
        income,
        balance,
        industry,
        average_amount.loc[eligible_index],
    )

    pd.testing.assert_series_equal(
        combined.loc[eligible_index, SOURCE_FACTOR],
        eligible_only[SOURCE_FACTOR],
    )
    assert combined.loc[[excluded_liquidity, excluded_incomplete], SOURCE_FACTOR].isna().all()
    assert coverage["complete_seven_factor_rows"] == 3
    assert coverage["liquid_complete_seven_factor_rows"] == 2
