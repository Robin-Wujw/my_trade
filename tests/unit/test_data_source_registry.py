from stock_research.data import DataSourceRole, default_source_registry


def test_default_source_registry_matches_target_architecture():
    registry = default_source_registry()

    assert registry.provider_for(DataSourceRole.PRIMARY_MARKET) == "miniqmt"
    assert registry.provider_for(DataSourceRole.REALTIME_MARKET) == "miniqmt"
    assert registry.provider_for(DataSourceRole.LIVE_TRADING) == "miniqmt"
    assert registry.provider_for(DataSourceRole.FUNDAMENTAL) == "tushare"
    assert registry.provider_for(DataSourceRole.VALUATION) == "tushare"
    assert registry.provider_for(DataSourceRole.SPECIAL_DATA) == "tushare"
    assert registry.provider_for(DataSourceRole.FALLBACK_PATCH) == "akshare"
