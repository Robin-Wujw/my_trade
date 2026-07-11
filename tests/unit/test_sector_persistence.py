import pandas as pd
import pytest
from types import SimpleNamespace

from stock_research.core.as_of import read_metadata
from stock_research.core.part_logger import PartLogger
from stock_research.pipelines import formula33, sector_statistics, sector_watch
from stock_research.storage import Database, KlineRepository, SectorRepository


def make_repository(tmp_path):
    database = Database(tmp_path / "my_trade.duckdb", code_version="test")
    database.initialize()
    return SectorRepository(database)


def make_kline_repository(tmp_path):
    database = Database(tmp_path / "kline.duckdb", code_version="test")
    database.initialize()
    return KlineRepository(database, lock_path=tmp_path / "kline.lock")


def test_sector_statistics_uses_duckdb_history_before_network(monkeypatch, tmp_path):
    repository = make_repository(tmp_path)
    logger = PartLogger("sector_stats", repository=repository)
    repository.upsert_board_history(
        "半导体",
        pd.DataFrame(
            [
                {
                    "date": f"2026-07-{day:02d}",
                    "open": float(day),
                    "close": float(day + 1),
                    "high": float(day + 2),
                    "low": float(day - 1),
                    "amount": float(day * 100),
                    "pct_chg": 0.01,
                }
                for day in range(1, 11)
            ]
        ),
        source="ths/history",
    )

    provider = SimpleNamespace(
        load_board_history=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("network called")
        )
    )
    monkeypatch.setattr(
        sector_statistics.time,
        "sleep",
        lambda _seconds: (_ for _ in ()).throw(AssertionError("sleep called")),
    )

    result = sector_statistics.load_board_history(
        "半导体",
        5,
        repository=repository,
        logger=logger,
        as_of_date="2026-07-10",
        provider=provider,
        request_sleep=5.0,
    )

    assert result["date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2026-07-06",
        "2026-07-07",
        "2026-07-08",
        "2026-07-09",
        "2026-07-10",
    ]
    assert result["close"].tolist() == [7.0, 8.0, 9.0, 10.0, 11.0]


def test_sector_watch_uses_duckdb_history_before_network(monkeypatch, tmp_path):
    repository = make_repository(tmp_path)
    logger = PartLogger("sector_watch", repository=repository)
    repository.upsert_board_history(
        "通信设备",
        pd.DataFrame(
            [
                {
                    "date": f"2026-07-{day:02d}",
                    "open": float(day),
                    "close": float(day + 2),
                    "high": float(day + 3),
                    "low": float(day - 1),
                    "amount": float(day * 1000),
                    "volume": float(day * 10),
                    "pct_chg": 1.0,
                }
                for day in range(1, 11)
            ]
        ),
        source="ths/history",
    )

    provider = SimpleNamespace(
        load_board_history=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("network called")
        )
    )
    monkeypatch.setattr(
        sector_watch.time,
        "sleep",
        lambda _seconds: (_ for _ in ()).throw(AssertionError("sleep called")),
    )

    result = sector_watch.load_board_history(
        "通信设备",
        5,
        as_of_date="2026-07-10",
        repository=repository,
        logger=logger,
        provider=provider,
        request_sleep=5.0,
    )

    assert result["日期"].dt.strftime("%Y-%m-%d").tolist() == [
        "2026-07-06",
        "2026-07-07",
        "2026-07-08",
        "2026-07-09",
        "2026-07-10",
    ]
    assert result["收盘"].tolist() == [8.0, 9.0, 10.0, 11.0, 12.0]


def test_sector_history_fetches_only_missing_right_edge(monkeypatch, tmp_path):
    repository = make_repository(tmp_path)
    dates = pd.bdate_range(end="2026-07-10", periods=60)
    repository.upsert_board_history(
        "半导体",
        pd.DataFrame(
            {
                "date": dates,
                "open": 10.0,
                "high": 11.0,
                "low": 9.0,
                "close": 10.5,
                "volume": 1000.0,
                "amount": 10000.0,
                "pct_chg": 0.01,
            }
        ),
        source="ths/history",
    )
    calls = []

    def fetch(code, *, start_date, end_date):
        calls.append((code, start_date, end_date))
        return pd.DataFrame(
            [
                {
                    "date": pd.Timestamp("2026-07-13"),
                    "open": 10.5,
                    "high": 11.5,
                    "low": 10.0,
                    "close": 11.0,
                    "volume": 1100.0,
                    "amount": 11000.0,
                    "pct_chg": 0.0476,
                }
            ]
        )

    result = sector_statistics.load_board_history(
        "半导体",
        60,
        repository=repository,
        as_of_date="2026-07-13",
        board_code="881121",
        provider=SimpleNamespace(load_board_history=fetch),
    )

    assert calls == [("881121", "20260711", "20260713")]
    assert len(result) == 60
    assert result["date"].max() == pd.Timestamp("2026-07-13")


def test_formula33_uses_duckdb_kline_before_network(monkeypatch, tmp_path):
    repository = make_kline_repository(tmp_path)
    monkeypatch.setattr(formula33, "KLINE_CACHE_DIR", str(tmp_path / "csv-cache"))
    repository.upsert_stock_kline(
        "akshare",
        "000001",
        pd.DataFrame(
            [
                {
                    "date": f"2026-07-{day:02d}",
                    "code": "000001",
                    "open": float(day),
                    "high": float(day + 1),
                    "low": float(day - 1),
                    "close": float(day + 0.5),
                    "volume": float(day * 100),
                    "tradestatus": "1",
                }
                for day in range(1, 10)
            ]
        ),
    )
    monkeypatch.setattr(
        formula33,
        "load_kline_akshare",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("network called")),
    )

    result = formula33.load_kline_with_cache(
        "akshare",
        "000001",
        "2026-07-01",
        "2026-07-09",
        repository=repository,
    )

    assert result["date"].tolist() == [f"2026-07-{day:02d}" for day in range(1, 10)]
    assert result["close"].iloc[-1] == 9.5


def test_formula33_persists_fresh_kline_to_duckdb(monkeypatch, tmp_path):
    repository = make_kline_repository(tmp_path)
    monkeypatch.setattr(formula33, "KLINE_CACHE_DIR", str(tmp_path / "csv-cache"))

    def fresh(*args, **kwargs):
        return pd.DataFrame(
            [
                {
                    "date": "2026-07-09",
                    "code": "000002",
                    "open": 20.0,
                    "high": 21.0,
                    "low": 19.0,
                    "close": 20.5,
                    "volume": 2000.0,
                    "tradestatus": "1",
                }
            ]
        )

    monkeypatch.setattr(formula33, "load_kline_akshare", fresh)

    result = formula33.load_kline_with_cache(
        "akshare",
        "000002",
        "2026-07-09",
        "2026-07-09",
        repository=repository,
    )
    loaded = repository.load_stock_kline(
        "akshare",
        "000002",
        start_date="2026-07-09",
        end_date="2026-07-09",
    )

    assert result["close"].tolist() == [20.5]
    assert loaded["close"].tolist() == [20.5]


def test_sector_watch_restores_api_percentage_units_from_duckdb():
    stored = pd.DataFrame(
        [{"date": "2026-07-09", "close": 10.5, "pct_chg": 0.012}]
    )

    converted = sector_watch._to_akshare_board_history(stored)

    assert converted["涨跌幅"].tolist() == pytest.approx([1.2])


@pytest.mark.parametrize(
    ("module", "name_column"),
    [
        (sector_statistics, "board"),
        (sector_watch, "board_name"),
    ],
)
def test_stale_board_list_is_refreshed_from_network(
    module,
    name_column,
    monkeypatch,
    tmp_path,
):
    database = Database(
        tmp_path / f"{module.__name__.rsplit('.', 1)[-1]}.duckdb",
        code_version="test",
    )
    database.initialize()
    repository = SectorRepository(database)
    repository.replace_boards(
        pd.DataFrame(
            [{"board": "旧板块", "group": "旧分类", "board_code": "881000"}]
        ),
        source="ths/board_list",
    )
    with database.connect() as connection:
        connection.execute(
            "UPDATE raw.sector_boards SET updated_at = CURRENT_TIMESTAMP - INTERVAL '2 days'"
        )

    provider = SimpleNamespace(
        load_board_list=lambda: pd.DataFrame(
            [{"code": "881001", "name": "新板块"}]
        )
    )
    monkeypatch.setattr(module, "write_cache", lambda *args, **kwargs: None)

    loaded = module.load_board_names(
        retries=1,
        retry_delay=0,
        repository=repository,
        provider=provider,
    )

    assert loaded[name_column].tolist() == ["新板块"]
    assert loaded["board_code"].tolist() == ["881001"]


def test_sector_watch_coverage_gate_stops_before_export(monkeypatch, tmp_path):
    repository = make_repository(tmp_path)
    boards = pd.DataFrame(
        [
            {"board_name": f"板块{index:02d}", "board_code": f"881{index:03d}"}
            for index in range(90)
        ]
    )
    fresh_history = pd.DataFrame(
        {
            "日期": pd.bdate_range(end="2026-07-10", periods=60),
            "开盘": 10.0,
            "收盘": 10.5,
            "最高": 11.0,
            "最低": 9.5,
            "成交量": 1000.0,
            "成交额": 10000.0,
            "涨跌幅": 1.0,
        }
    )
    monkeypatch.setattr(sector_watch, "load_board_names", lambda **kwargs: boards)
    monkeypatch.setattr(
        sector_watch,
        "load_benchmark",
        lambda *args, **kwargs: pd.DataFrame(
            {
                "date": pd.bdate_range(end="2026-07-10", periods=30),
                "close": range(30),
            }
        ),
    )
    monkeypatch.setattr(sector_watch, "load_limit_up_counts", lambda *args, **kwargs: {})

    def history(board_name, *args, **kwargs):
        index = int(board_name[-2:])
        return fresh_history.copy() if index < 85 else pd.DataFrame()

    monkeypatch.setattr(sector_watch, "load_board_history", history)
    monkeypatch.setattr(sector_watch, "OUTPUT_DIR", str(tmp_path / "exports"))

    with pytest.raises(SystemExit) as error:
        sector_watch.main(
            [
                "--as-of-date", "2026-07-10",
                "--workers", "1",
                "--days", "25",
                "--sleep", "0",
                "--retries", "1",
                "--retry-delay", "0",
            ],
            repository=repository,
        )

    assert error.value.code == 2
    assert not list((tmp_path / "exports").glob("sector_watch_*.csv"))


def test_sector_statistics_stops_after_fifth_definite_failure(monkeypatch, tmp_path):
    repository = make_repository(tmp_path)
    boards = pd.DataFrame(
        [
            {
                "board": f"板块{index:02d}",
                "group": "测试",
                "board_code": f"881{index:03d}",
            }
            for index in range(90)
        ]
    )
    calls = []
    monkeypatch.setattr(
        sector_statistics,
        "load_benchmark_frame",
        lambda: pd.DataFrame(
            [{"date": "2026-07-10", "open": 3500.0, "close": 3510.0}]
        ),
    )
    monkeypatch.setattr(sector_statistics, "load_board_names", lambda **kwargs: boards)

    def truncated_history(board, *args, **kwargs):
        calls.append(board)
        return pd.DataFrame([{"date": "2026-07-10", "close": 10.0}])

    monkeypatch.setattr(sector_statistics, "load_board_history", truncated_history)
    monkeypatch.setattr(
        sector_statistics,
        "save_outputs",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("export called")),
    )

    with pytest.raises(SystemExit) as error:
        sector_statistics.main(
            [
                "--history-days", "25",
                "--sleep", "0",
                "--retries", "1",
                "--retry-delay", "0",
            ],
            repository=repository,
        )

    assert error.value.code == 2
    assert len(calls) == 5


def test_sector_watch_uses_bounded_window_and_stops_after_gate_is_impossible(
    monkeypatch,
    tmp_path,
):
    repository = make_repository(tmp_path)
    boards = pd.DataFrame(
        [
            {"board_name": f"板块{index:02d}", "board_code": f"881{index:03d}"}
            for index in range(90)
        ]
    )
    calls = []
    monkeypatch.setattr(sector_watch, "load_board_names", lambda **kwargs: boards)
    monkeypatch.setattr(
        sector_watch,
        "load_benchmark",
        lambda *args, **kwargs: pd.DataFrame(
            {
                "date": pd.bdate_range(end="2026-07-10", periods=30),
                "close": range(30),
            }
        ),
    )
    monkeypatch.setattr(sector_watch, "load_limit_up_counts", lambda *args, **kwargs: {})

    def truncated_history(board_name, *args, **kwargs):
        calls.append(board_name)
        return pd.DataFrame([{"日期": "2026-07-10", "收盘": 10.0}])

    monkeypatch.setattr(sector_watch, "load_board_history", truncated_history)
    monkeypatch.setattr(sector_watch, "OUTPUT_DIR", str(tmp_path / "exports"))

    with pytest.raises(SystemExit) as error:
        sector_watch.main(
            [
                "--as-of-date", "2026-07-10",
                "--workers", "4",
                "--days", "25",
                "--sleep", "0",
                "--retries", "1",
                "--retry-delay", "0",
            ],
            repository=repository,
        )

    assert error.value.code == 2
    assert 5 <= len(calls) <= 8
    assert not list((tmp_path / "exports").glob("sector_watch_*.csv"))


def test_sector_benchmarks_use_sina_index(monkeypatch):
    calls = []
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-07-09", "2026-07-10"]),
            "open": [3500.0, 3510.0],
            "close": [3510.0, 3520.0],
        }
    )

    def load(symbol):
        calls.append(symbol)
        return frame.copy()

    monkeypatch.setattr(sector_watch.ak, "stock_zh_index_daily", load)
    monkeypatch.setattr(sector_statistics.ak, "stock_zh_index_daily", load)

    watch = sector_watch.load_benchmark(2, "2026-07-10")
    stats = sector_statistics.load_benchmark_daily(["2026-07-09", "2026-07-10"])

    assert calls == ["sh000001", "sh000001"]
    assert len(watch) == 2
    assert len(stats) == 2


def test_sector_statistics_reuses_first_successful_benchmark_response(
    monkeypatch,
    tmp_path,
):
    repository = make_repository(tmp_path)
    dates = pd.bdate_range(end="2026-07-10", periods=60)
    benchmark_source = pd.DataFrame(
        {
            "date": dates,
            "open": [3500.0 + index for index in range(60)],
            "close": [3501.0 + index for index in range(60)],
        }
    )
    history = pd.DataFrame(
        {
            "date": dates,
            "open": 10.0,
            "high": 11.0,
            "low": 9.0,
            "close": [10.0 + index / 100 for index in range(60)],
            "amount": 10000.0,
            "pct_chg": 0.001,
        }
    )
    calls = []

    def load_index(symbol):
        calls.append(symbol)
        if len(calls) > 1:
            raise OSError("transient second-call failure")
        return benchmark_source.copy()

    monkeypatch.setattr(sector_statistics.ak, "stock_zh_index_daily", load_index)
    monkeypatch.setattr(
        sector_statistics,
        "load_board_names",
        lambda **kwargs: pd.DataFrame(
            [{"board": "半导体", "group": "半导体", "board_code": "881121"}]
        ),
    )
    monkeypatch.setattr(
        sector_statistics,
        "load_board_history",
        lambda *args, **kwargs: history.copy(),
    )
    monkeypatch.setattr(
        sector_statistics,
        "load_limit_up_by_date",
        lambda date_keys: pd.DataFrame(),
    )
    captured = {}

    def save_outputs(board_daily, limit_up, benchmark, top_amount):
        captured["date_keys"] = sorted(board_daily["date_key"].unique())
        captured["benchmark"] = benchmark.copy()
        return "sector.xlsx", "sector.md", []

    monkeypatch.setattr(sector_statistics, "save_outputs", save_outputs)

    sector_statistics.main(
        [
            "--history-days", "60",
            "--sleep", "0",
            "--retries", "1",
            "--retry-delay", "0",
        ],
        repository=repository,
    )

    assert calls == ["sh000001"]
    assert not captured["benchmark"].empty
    assert captured["benchmark"]["date_key"].tolist() == captured["date_keys"]


def test_mainline_constituents_use_ths_board_code(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(sector_watch, "CACHE_DIR", str(tmp_path / "cache"))
    provider = SimpleNamespace(
        load_board_constituents=lambda code: calls.append(code) or pd.DataFrame(
            [
                {"code": "000001", "name": "平安银行"},
                {"code": "600000", "name": "浦发银行"},
            ]
        )
    )
    target = tmp_path / "constituents.csv"
    boards = pd.DataFrame(
        [
            {
                "board": "银行",
                "board_code": "881155",
                "final_score": 88.0,
                "date": "2026-07-10",
            }
        ]
    )

    sector_watch.save_mainline_constituents(
        boards,
        top=1,
        retries=1,
        retry_delay=0,
        sleep=0,
        output_path=str(target),
        provider=provider,
    )

    saved = pd.read_csv(target, dtype={"code": str})
    assert calls == ["881155"]
    assert saved["code"].tolist() == ["sz.000001", "sh.600000"]
    metadata = read_metadata(str(target))
    assert metadata["board_coverage_expected"] == 1
    assert metadata["board_coverage_completed"] == 1
    assert metadata["board_coverage"] == 1.0


def test_mainline_constituent_failure_invalidates_old_output(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(sector_watch, "CACHE_DIR", str(tmp_path / "cache"))

    def load_constituents(code):
        calls.append(code)
        if code == "881002":
            raise OSError("network unavailable")
        return pd.DataFrame([{"code": "000001", "name": "平安银行"}])

    provider = SimpleNamespace(load_board_constituents=load_constituents)
    target = tmp_path / "constituents.csv"
    target.write_text("stale-output", encoding="utf-8")
    (tmp_path / "constituents.csv.meta.json").write_text(
        '{"kind":"sector_mainline_constituents"}',
        encoding="utf-8",
    )
    boards = pd.DataFrame(
        [
            {"board": "银行", "board_code": "881001", "final_score": 88.0},
            {"board": "半导体", "board_code": "881002", "final_score": 87.0},
        ]
    )

    with pytest.raises(RuntimeError, match="半导体.*1/2"):
        sector_watch.save_mainline_constituents(
            boards,
            top=2,
            retries=1,
            retry_delay=0,
            sleep=0,
            output_path=str(target),
            provider=provider,
        )

    assert calls == ["881001", "881002"]
    assert not target.exists()
    assert not (tmp_path / "constituents.csv.meta.json").exists()
    assert not list(tmp_path.glob("constituents.csv.*.tmp"))


def test_sector_watch_main_fails_closed_when_top_constituents_fail(
    monkeypatch,
    tmp_path,
):
    repository = make_repository(tmp_path)
    target = tmp_path / "sector_mainline_constituents.csv"
    target.write_text("stale-output", encoding="utf-8")
    boards = pd.DataFrame(
        [{"board_name": "半导体", "board_code": "881121"}]
    )
    fresh_history = pd.DataFrame(
        {
            "日期": pd.bdate_range(end="2026-07-10", periods=60),
            "收盘": 10.0,
        }
    )
    monkeypatch.setattr(sector_watch, "BOARD_CONSTITUENT_FILE", str(target))
    monkeypatch.setattr(sector_watch, "CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(sector_watch, "OUTPUT_DIR", str(tmp_path / "exports"))
    monkeypatch.setattr(sector_watch, "load_board_names", lambda **kwargs: boards)
    monkeypatch.setattr(
        sector_watch,
        "load_benchmark",
        lambda *args, **kwargs: pd.DataFrame(
            {
                "date": pd.bdate_range(end="2026-07-10", periods=30),
                "close": range(30),
            }
        ),
    )
    monkeypatch.setattr(sector_watch, "load_limit_up_counts", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        sector_watch,
        "load_board_history",
        lambda *args, **kwargs: fresh_history.copy(),
    )
    monkeypatch.setattr(
        sector_watch,
        "calc_board_metrics",
        lambda board, history, benchmark: {
            "board": board,
            "date": "2026-07-10",
            "mainline_score": 50.0,
            "ret5": 0.1,
        },
    )
    provider = SimpleNamespace(
        load_board_constituents=lambda code: (_ for _ in ()).throw(
            OSError("offline")
        )
    )

    with pytest.raises(SystemExit) as error:
        sector_watch.main(
            [
                "--top", "1",
                "--workers", "1",
                "--days", "25",
                "--sleep", "0",
                "--retries", "1",
                "--retry-delay", "0",
            ],
            repository=repository,
            provider=provider,
        )

    assert error.value.code == 2
    assert not target.exists()
    assert not list((tmp_path / "exports").glob("sector_watch_*.csv"))


def test_ths_internal_retry_is_not_repeated_by_sector_pipeline():
    calls = []

    def fail(*args, **kwargs):
        calls.append((args, kwargs))
        raise OSError("offline")

    provider = SimpleNamespace(
        REQUEST_ATTEMPTS=2,
        load_board_history=fail,
    )

    with pytest.raises(OSError, match="offline"):
        sector_statistics.load_board_history(
            "半导体",
            60,
            as_of_date="2026-07-10",
            board_code="881121",
            provider=provider,
            retries=4,
            retry_delay=0,
        )

    assert len(calls) == 1


def test_limit_up_pool_is_cached_by_date(monkeypatch, tmp_path):
    calls = []

    def fetch(date):
        calls.append(date)
        return pd.DataFrame(
            [{"代码": "000001", "名称": "平安银行", "所属行业": "银行"}]
        )

    monkeypatch.setattr(sector_statistics, "CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(sector_statistics.ak, "stock_zt_pool_em", fetch)

    first = sector_statistics.load_limit_up_by_date(["2026-07-10"])
    second = sector_statistics.load_limit_up_by_date(["2026-07-10"])

    assert calls == ["20260710"]
    assert first[["date_key", "code", "board"]].equals(
        second[["date_key", "code", "board"]]
    )


def test_limit_up_pool_missing_required_date_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(sector_statistics, "CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(
        sector_statistics.ak,
        "stock_zt_pool_em",
        lambda date: pd.DataFrame(),
    )

    with pytest.raises(RuntimeError, match="2026-07-10"):
        sector_statistics.load_limit_up_by_date(["2026-07-10"])


def test_top_board_limit_up_count_is_confirmed_by_constituent_codes(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(sector_watch, "CACHE_DIR", str(tmp_path))
    sector_watch.write_cache(
        "limit_up_pool",
        "2026-07-10",
        pd.DataFrame(
            [
                {"code": "000001", "board": "银行"},
                {"code": "000002", "board": "银行"},
                {"code": "600000", "board": "房地产"},
            ]
        ),
    )
    boards = pd.DataFrame(
        [{"board": "银行", "mainline_score": 50.0, "limit_up_count": 99, "final_score": 99.0}]
    )
    constituents = pd.DataFrame(
        [
            {"board": "银行", "code": "sz.000001"},
            {"board": "银行", "code": "sh.600000"},
        ]
    )

    result = sector_watch.apply_constituent_limit_up_counts(
        boards,
        constituents,
        ["2026-07-10"],
    )

    assert result.iloc[0]["limit_up_count"] == 2
    assert result.iloc[0]["final_score"] < 99.0
