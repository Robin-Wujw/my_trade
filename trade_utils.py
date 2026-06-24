# -*- coding: utf-8 -*-
import json
import os
import time

import requests


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PUSHPLUS_URL = os.environ.get("PUSHPLUS_URL", "https://www.pushplus.plus/send")


def get_project_path(filename):
    return os.path.join(BASE_DIR, filename)


def get_pushplus_token():
    token = os.environ.get("PUSHPLUS_TOKEN", "").strip()
    if token:
        return token

    token_file = os.environ.get("PUSHPLUS_TOKEN_FILE", os.path.join(BASE_DIR, ".pushplus_token"))
    try:
        with open(token_file, "r", encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return ""


def load_last_result(path):
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return {item["code"]: item["name"] for item in data.get("stocks", [])}
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        print(f"读取历史结果失败: {path} | {exc}")
    return {}


def save_current_result(path, date_str, rows):
    try:
        data = {"date": date_str, "stocks": [{"code": r["code"], "name": r["name"]} for r in rows]}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except (OSError, KeyError, TypeError) as exc:
        print(f"保存历史结果失败: {path} | {exc}")


def build_diff_html(last_dict, current_rows):
    current_dict = {r["code"]: r["name"] for r in current_rows if r.get("code")}
    added = {code: name for code, name in current_dict.items() if code not in last_dict}
    removed = {code: name for code, name in last_dict.items() if code not in current_dict}
    if not added and not removed:
        return "<p>与上一交易日相比：无变化</p>"

    parts = []
    if added:
        items = "、".join([f"{name}({code})" for code, name in added.items()])
        parts.append(f"<p style='color:red'>🔴 新增({len(added)}): {items}</p>")
    if removed:
        items = "、".join([f"{name}({code})" for code, name in removed.items()])
        parts.append(f"<p style='color:green'>🟢 移除({len(removed)}): {items}</p>")
    return "".join(parts)


def send_pushplus(title, content, retries=3, timeout=10):
    token = get_pushplus_token()
    if not token:
        print("PushPlus token 未配置，跳过推送。请设置 PUSHPLUS_TOKEN 或 .pushplus_token。")
        return False

    payload = {"token": token, "title": title, "content": content, "template": "html"}
    headers = {"User-Agent": "Mozilla/5.0", "Content-Type": "application/json"}

    for attempt in range(1, retries + 1):
        try:
            res = requests.post(url=PUSHPLUS_URL, headers=headers, json=payload, timeout=timeout)
            try:
                body = res.json()
            except ValueError:
                body = {}
            if body.get("code") == 200:
                print("PushPlus推送成功")
                return True
            msg = body.get("msg") or res.text[:120]
            print(f"PushPlus推送失败: HTTP {res.status_code} | {msg} | 第 {attempt}/{retries} 次")
        except requests.RequestException as exc:
            print(f"PushPlus推送网络错误: {exc} | 第 {attempt}/{retries} 次")

        if attempt < retries:
            time.sleep(3 * attempt)

    print("PushPlus推送多次重试后仍失败")
    return False


def call_with_retry(func, *args, retries=3, delay=2, label=None, **kwargs):
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            prefix = f"{label} " if label else ""
            print(f"{prefix}请求失败: {exc} | 第 {attempt}/{retries} 次")
            if attempt < retries:
                time.sleep(delay * attempt)
    raise last_exc
