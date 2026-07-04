from stock_research.core.paths import ProjectPaths


def test_project_paths_put_all_runtime_files_under_var(tmp_path):
    paths = ProjectPaths(project_root=tmp_path)

    assert paths.runtime_root == tmp_path / "var"
    assert paths.cache == tmp_path / "var" / "cache"
    assert paths.database == tmp_path / "var" / "data" / "my_trade.duckdb"
    assert paths.selection_exports == tmp_path / "var" / "exports" / "selection"
    assert paths.market_exports == tmp_path / "var" / "exports" / "market"
    assert paths.report_exports == tmp_path / "var" / "exports" / "reports"


def test_runtime_root_can_be_overridden_without_moving_project(monkeypatch, tmp_path):
    runtime = tmp_path / "runtime"
    monkeypatch.setenv("STOCK_RESEARCH_VAR", str(runtime))

    paths = ProjectPaths(project_root=tmp_path / "project")

    assert paths.runtime_root == runtime.resolve()
