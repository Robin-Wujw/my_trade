from stock_research.api.pushplus import get_pushplus_token
from stock_research.core.paths import ProjectPaths


def test_pushplus_token_prefers_environment_then_local_secret(monkeypatch, tmp_path):
    paths = ProjectPaths(tmp_path)
    paths.secrets.mkdir(parents=True)
    (paths.secrets / "pushplus_token").write_text("file-token", encoding="utf-8")
    monkeypatch.setenv("PUSHPLUS_TOKEN", "env-token")

    assert get_pushplus_token(paths) == "env-token"

    monkeypatch.delenv("PUSHPLUS_TOKEN")
    assert get_pushplus_token(paths) == "file-token"
