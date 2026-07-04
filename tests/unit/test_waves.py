from stock_research.indicators.waves import calc_wave_pct, wave_levels


def test_wave_levels_are_deterministic():
    assert calc_wave_pct(10, 20, 15) == 50.0
    assert wave_levels(10, 20, current=16)["level_625"] == 16.25
