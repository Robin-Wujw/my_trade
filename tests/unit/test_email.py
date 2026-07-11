from stock_research.api.email import load_smtp_config, send_html_email
from stock_research.core.paths import ProjectPaths


def configure(monkeypatch):
    monkeypatch.setenv("SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("SMTP_PORT", "465")
    monkeypatch.setenv("SMTP_USERNAME", "sender@example.com")
    monkeypatch.setenv("SMTP_PASSWORD", "app-password")
    monkeypatch.setenv("REPORT_EMAIL_TO", "one@example.com; two@example.com")


def test_smtp_config_parses_recipients_and_defaults_sender(monkeypatch, tmp_path):
    configure(monkeypatch)

    config = load_smtp_config(paths=ProjectPaths(tmp_path))

    assert config.sender == "sender@example.com"
    assert config.recipients == ("one@example.com", "two@example.com")
    assert config.security == "ssl"


def test_send_html_email_uses_authenticated_ssl(monkeypatch, tmp_path):
    configure(monkeypatch)
    captured = {}

    class Client:
        def __init__(self, host, port, timeout):
            captured.update(host=host, port=port, timeout=timeout)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def login(self, username, password):
            captured.update(username=username, password=password)

        def send_message(self, message):
            captured["message"] = message

    monkeypatch.setattr("stock_research.api.email.smtplib.SMTP_SSL", Client)

    assert send_html_email(
        "Daily report", "<h1>Complete report</h1>", paths=ProjectPaths(tmp_path)
    ) is True
    assert captured["host"] == "smtp.example.com"
    assert captured["message"].get_content_type() == "multipart/alternative"
    assert "Complete report" in str(captured["message"])
