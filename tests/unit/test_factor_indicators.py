from stock_research.indicators.factors import (
    parse_float,
    parse_pct,
    parse_yi,
    remove_outliers,
    score_direct,
)


def test_factor_parsers_and_scalers_keep_current_behavior():
    assert parse_yi("1.5亿") == 150_000_000
    assert parse_pct("12.5%") == 0.125
    assert parse_float("1,234.5") == 1234.5
    assert remove_outliers([10, 11, 12, 100]) == [10, 11, 12]
    assert score_direct(5, 0, 10) == 50
