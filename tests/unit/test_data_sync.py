from apps.data_sync import TUSHARE_DEFAULT_FIELDS


def test_income_sync_defaults_capture_forward_gross_profitability_fields():
    fields = set(TUSHARE_DEFAULT_FIELDS["income"].split(","))

    assert {
        "ts_code",
        "ann_date",
        "f_ann_date",
        "end_date",
        "total_revenue",
        "oper_cost",
        "comp_type",
        "update_flag",
    }.issubset(fields)
