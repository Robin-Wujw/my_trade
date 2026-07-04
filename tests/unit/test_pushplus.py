from stock_research.api.pushplus import get_pushplus_token, send_pushplus
from stock_research.core.paths import ProjectPaths


def test_pushplus_token_prefers_environment_then_local_secret(monkeypatch, tmp_path):
    paths = ProjectPaths(tmp_path)
    paths.secrets.mkdir(parents=True)
    (paths.secrets / "pushplus_token").write_text("file-token", encoding="utf-8")
    monkeypatch.setenv("PUSHPLUS_TOKEN", "env-token")

    assert get_pushplus_token(paths) == "env-token"

    monkeypatch.delenv("PUSHPLUS_TOKEN")
    assert get_pushplus_token(paths) == "file-token"


def test_pushplus_api_guard_sends_at_most_18000_characters(monkeypatch, tmp_path):
    paths = ProjectPaths(tmp_path)
    captured = {}

    class Response:
        status_code = 200
        text = "ok"

        @staticmethod
        def json():
            return {"code": 200}

    def post(**kwargs):
        captured.update(kwargs["json"])
        return Response()

    monkeypatch.setenv("PUSHPLUS_TOKEN", "token")
    monkeypatch.delenv("PUSHPLUS_MAX_CONTENT_CHARS", raising=False)
    monkeypatch.setattr("stock_research.api.pushplus.requests.post", post)

    assert send_pushplus("title", "x" * 19000, paths=paths) is True
    assert len(captured["content"]) == 18000
