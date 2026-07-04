from stock_research.core.config import load_pipeline_config


def test_environment_overrides_toml(monkeypatch, tmp_path):
    path = tmp_path / "pipeline.toml"
    path.write_text(
        "[factor]\nworkers = 1\n"
        "[formula33]\nworkers = 2\nsleep = 0.2\nretries = 5\n"
        "[sector]\nsleep = 0.3\nretries = 5\n"
        "[fundamental]\nmax_updates = 100\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("FORMULA33_WORKERS", "4")

    config = load_pipeline_config(path)

    assert config.formula33_workers == 4
