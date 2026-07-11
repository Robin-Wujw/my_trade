"""SMTP delivery for the complete HTML daily report."""
from __future__ import annotations

from dataclasses import dataclass
from email.message import EmailMessage
import os
from pathlib import Path
import smtplib

from stock_research.core.paths import PATHS, ProjectPaths


@dataclass(frozen=True)
class SmtpConfig:
    host: str
    port: int
    username: str
    password: str
    sender: str
    recipients: tuple[str, ...]
    security: str


def _secret(name: str, filename: str, paths: ProjectPaths) -> str:
    value = os.environ.get(name, "").strip()
    if value:
        return value
    try:
        return (paths.secrets / filename).read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def load_smtp_config(
    *, recipients: str = "", paths: ProjectPaths = PATHS
) -> SmtpConfig:
    host = os.environ.get("SMTP_HOST", "").strip()
    username = os.environ.get("SMTP_USERNAME", "").strip()
    password = _secret("SMTP_PASSWORD", "smtp_password", paths)
    sender = os.environ.get("SMTP_FROM", "").strip() or username
    recipient_text = recipients.strip() or os.environ.get("REPORT_EMAIL_TO", "").strip()
    recipient_list = tuple(
        item.strip() for item in recipient_text.replace(";", ",").split(",")
        if item.strip()
    )
    security = os.environ.get("SMTP_SECURITY", "ssl").strip().lower()
    if security not in {"ssl", "starttls", "plain"}:
        raise ValueError("SMTP_SECURITY must be ssl, starttls, or plain")
    default_port = 465 if security == "ssl" else 587
    try:
        port = int(os.environ.get("SMTP_PORT", str(default_port)))
    except ValueError as exc:
        raise ValueError("SMTP_PORT must be an integer") from exc
    missing = [
        name for name, value in (
            ("SMTP_HOST", host),
            ("SMTP_USERNAME", username),
            ("SMTP_PASSWORD", password),
            ("SMTP_FROM", sender),
            ("REPORT_EMAIL_TO", recipient_list),
        )
        if not value
    ]
    if missing:
        raise RuntimeError("HTML email configuration missing: " + ", ".join(missing))
    return SmtpConfig(host, port, username, password, sender, recipient_list, security)


def send_html_email(
    subject: str,
    html_content: str,
    *,
    recipients: str = "",
    paths: ProjectPaths = PATHS,
    timeout: float = 30,
) -> bool:
    config = load_smtp_config(recipients=recipients, paths=paths)
    message = EmailMessage()
    message["Subject"] = str(subject)
    message["From"] = config.sender
    message["To"] = ", ".join(config.recipients)
    message.set_content("This report requires an HTML-capable email client.")
    message.add_alternative(str(html_content), subtype="html")
    try:
        if config.security == "ssl":
            client = smtplib.SMTP_SSL(config.host, config.port, timeout=timeout)
        else:
            client = smtplib.SMTP(config.host, config.port, timeout=timeout)
        with client:
            if config.security == "starttls":
                client.starttls()
            client.login(config.username, config.password)
            client.send_message(message)
        return True
    except (OSError, smtplib.SMTPException) as exc:
        print(f"HTML email delivery failed: {exc}")
        return False
