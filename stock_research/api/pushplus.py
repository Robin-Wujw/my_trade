"""PushPlus token discovery and message delivery."""
from __future__ import annotations

import os
import time

import requests

from stock_research.core.paths import PATHS, ProjectPaths


DEFAULT_URL = "https://www.pushplus.plus/send"


def get_pushplus_token(paths: ProjectPaths = PATHS) -> str:
    token = os.environ.get("PUSHPLUS_TOKEN", "").strip()
    if token:
        return token
    token_file = os.environ.get(
        "PUSHPLUS_TOKEN_FILE", str(paths.secrets / "pushplus_token")
    )
    try:
        with open(token_file, "r", encoding="utf-8") as handle:
            return handle.read().strip()
    except OSError:
        return ""


def send_pushplus(
    title, content, retries=3, timeout=10, paths: ProjectPaths = PATHS
):
    token = get_pushplus_token(paths)
    if not token:
        print("PushPlus token 未配置，跳过推送。")
        return False
    max_chars = int(os.environ.get("PUSHPLUS_MAX_CONTENT_CHARS", "18000"))
    title = str(title)[:100]
    content = str(content)
    if len(content) > max_chars:
        suffix = "<hr><p>内容超过推送长度限制，已自动截断；完整报告请查看本地报告文件。</p>"
        content = content[: max(0, max_chars - len(suffix))] + suffix
    payload = {"token": token, "title": title, "content": content, "template": "html"}
    headers = {"User-Agent": "Mozilla/5.0", "Content-Type": "application/json"}
    url = os.environ.get("PUSHPLUS_URL", DEFAULT_URL)
    for attempt in range(1, retries + 1):
        try:
            response = requests.post(
                url=url, headers=headers, json=payload, timeout=timeout
            )
            try:
                body = response.json()
            except ValueError:
                body = {}
            if body.get("code") == 200:
                return True
            message = body.get("msg") or response.text[:120]
            print(
                f"PushPlus推送失败: HTTP {response.status_code} | {message} | "
                f"第 {attempt}/{retries} 次"
            )
        except requests.RequestException as exc:
            print(f"PushPlus推送网络错误: {exc} | 第 {attempt}/{retries} 次")
        if attempt < retries:
            time.sleep(3 * attempt)
    return False
