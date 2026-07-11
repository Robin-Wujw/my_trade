import ast
from pathlib import Path


ROOT = Path(__file__).parents[2]
OLD_MODULES = {
    "dailyFundamentalSelect",
    "dailyReportPush",
    "factorStock",
    "formula33Stats",
    "fullMarketFundamentalUpdate",
    "pipelineAlert",
    "point_in_time",
    "sectorStats",
    "sectorWatch",
    "trade_utils",
    "wave_utils",
}


def imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
    return found


def test_root_contains_no_business_python_files():
    assert [path.name for path in ROOT.glob("*.py")] == []


def test_old_module_names_are_not_imported():
    offenders = {}
    paths = [*ROOT.glob("apps/**/*.py"), *ROOT.glob("stock_research/**/*.py")]
    for path in paths:
        stale = {
            name for name in imports(path) if name.split(".")[0] in OLD_MODULES
        }
        if stale:
            offenders[path.relative_to(ROOT).as_posix()] = sorted(stale)
    assert offenders == {}


def test_core_never_depends_on_outer_layers():
    forbidden = (
        "apps",
        "stock_research.pipelines",
        "stock_research.reporting",
        "stock_research.strategies",
    )
    offenders = {}
    for path in ROOT.glob("stock_research/core/*.py"):
        bad = {name for name in imports(path) if name.startswith(forbidden)}
        if bad:
            offenders[path.name] = sorted(bad)
    assert offenders == {}


def test_api_never_depends_on_strategy_or_pipeline():
    forbidden = ("stock_research.strategies", "stock_research.pipelines")
    offenders = {}
    for path in ROOT.glob("stock_research/api/*.py"):
        bad = {name for name in imports(path) if name.startswith(forbidden)}
        if bad:
            offenders[path.name] = sorted(bad)
    assert offenders == {}


def test_pipelines_use_market_sdks_through_api_adapters():
    offenders = {}
    for path in ROOT.glob("stock_research/pipelines/*.py"):
        direct = imports(path) & {"akshare", "baostock", "tushare"}
        if direct:
            offenders[path.name] = sorted(direct)
    assert offenders == {}
