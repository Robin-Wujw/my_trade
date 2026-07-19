"""Read-only MiniQMT adapter.

The production research pipeline does not place broker orders.  This module
keeps MiniQMT optional, lazy-imported, and read-only so local account checks can
coexist with the existing after-close workflow.
"""
from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
import importlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import textwrap
from typing import Any

from stock_research.core.paths import PATHS


DEFAULT_PROCESS_NAMES = ("XtMiniQmt.exe", "miniquote.exe", "XtuService.exe")
DEFAULT_CONFIG_PATH = PATHS.secrets / "miniqmt.json"


class MiniQmtError(RuntimeError):
    """Base MiniQMT integration error."""


class MiniQmtSdkNotFound(MiniQmtError):
    """Raised when the xtquant SDK cannot be found or imported."""


class MiniQmtConnectionError(MiniQmtError):
    """Raised when MiniQMT does not accept a trader connection."""


class LiveTradingDisabled(MiniQmtError):
    """Raised for order APIs until live trading is deliberately enabled."""


@dataclass(frozen=True)
class MiniQmtConfig:
    qmt_root: Path | None = None
    userdata_path: Path | None = None
    python_executable: Path | None = None
    accounts: tuple[str, ...] = ()
    session_id: int | None = None

    @property
    def resolved_qmt_root(self) -> Path:
        if self.qmt_root is not None:
            return Path(self.qmt_root).resolve()
        discovered = discover_qmt_root()
        if discovered is None:
            raise MiniQmtSdkNotFound(
                "MiniQMT root not configured. Set MINIQMT_ROOT or create "
                "var/secrets/miniqmt.json."
            )
        return discovered

    @property
    def bin_dir(self) -> Path:
        return self.resolved_qmt_root / "bin.x64"

    @property
    def site_packages(self) -> Path:
        return self.bin_dir / "Lib" / "site-packages"

    @property
    def resolved_userdata_path(self) -> Path:
        if self.userdata_path is not None:
            return Path(self.userdata_path).resolve()
        return self.resolved_qmt_root / "userdata_mini"

    @property
    def resolved_python_executable(self) -> Path:
        if self.python_executable is not None:
            return Path(self.python_executable).resolve()
        pythonw = self.bin_dir / "pythonw.exe"
        python = self.bin_dir / "python.exe"
        if pythonw.is_file():
            return pythonw
        return python


def discover_qmt_root(candidates: tuple[Path, ...] | None = None) -> Path | None:
    """Find a local QMT install without hard-coding a broker-specific folder."""
    roots = candidates or (
        Path("D:/Softwares/zhzqqmt"),
        Path("C:/Program Files"),
        Path("C:/Program Files (x86)"),
    )
    for root in roots:
        if not root.exists():
            continue
        with suppress(OSError):
            for child in root.iterdir():
                if _looks_like_qmt_root(child):
                    return child.resolve()
        if _looks_like_qmt_root(root):
            return root.resolve()
    return None


def _looks_like_qmt_root(path: Path) -> bool:
    return (
        (path / "bin.x64" / "Lib" / "site-packages" / "xtquant").is_dir()
        and (path / "userdata_mini").is_dir()
    )


def load_miniqmt_config(path: str | Path | None = None) -> MiniQmtConfig:
    """Load MiniQMT config from JSON plus environment overrides."""
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    data: dict[str, Any] = {}
    if config_path.is_file():
        data = json.loads(config_path.read_text(encoding="utf-8"))

    qmt_root = os.environ.get("MINIQMT_ROOT") or data.get("qmt_root")
    userdata_path = os.environ.get("MINIQMT_USERDATA_PATH") or data.get("userdata_path")
    python_executable = os.environ.get("MINIQMT_PYTHON") or data.get("python_executable")
    accounts_value = os.environ.get("MINIQMT_ACCOUNTS")
    accounts = _split_accounts(accounts_value) if accounts_value else tuple(data.get("accounts") or ())
    session_value = os.environ.get("MINIQMT_SESSION_ID") or data.get("session_id")

    return MiniQmtConfig(
        qmt_root=Path(qmt_root) if qmt_root else None,
        userdata_path=Path(userdata_path) if userdata_path else None,
        python_executable=Path(python_executable) if python_executable else None,
        accounts=tuple(str(item).strip() for item in accounts if str(item).strip()),
        session_id=int(session_value) if session_value else None,
    )


def _split_accounts(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.replace(";", ",").split(",") if item.strip())


def mask_account_id(account_id: Any) -> str:
    text = "" if account_id is None else str(account_id)
    if len(text) <= 7:
        return "*" * len(text)
    return f"{text[:3]}****{text[-4:]}"


def detect_running_processes(process_names: tuple[str, ...] = DEFAULT_PROCESS_NAMES) -> dict[str, bool]:
    """Return whether common MiniQMT processes are visible in Windows tasklist."""
    wanted = {name.lower(): name for name in process_names}
    found = {name: False for name in process_names}
    if os.name != "nt":
        return found
    try:
        completed = subprocess.run(
            ["tasklist", "/FO", "CSV", "/NH"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return found
    for line in completed.stdout.splitlines():
        process = line.split(",", 1)[0].strip().strip('"').lower()
        if process in wanted:
            found[wanted[process]] = True
    return found


def _add_sdk_path(config: MiniQmtConfig) -> None:
    site_packages = config.site_packages
    bin_dir = config.bin_dir
    if not site_packages.is_dir():
        raise MiniQmtSdkNotFound(f"xtquant site-packages not found: {site_packages}")
    if str(site_packages) not in sys.path:
        sys.path.insert(0, str(site_packages))
    os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")
    add_dll_directory = getattr(os, "add_dll_directory", None)
    if add_dll_directory and bin_dir.is_dir():
        with suppress(OSError):
            add_dll_directory(str(bin_dir))


def check_sdk(config: MiniQmtConfig | None = None) -> dict[str, Any]:
    cfg = config or load_miniqmt_config()
    status = {
        "qmt_root": str(cfg.qmt_root or discover_qmt_root() or ""),
        "userdata_path": "",
        "site_packages": "",
        "site_packages_exists": False,
        "userdata_exists": False,
        "xtquant_importable": False,
        "bridge_python": "",
        "bridge_python_exists": False,
        "error": None,
    }
    try:
        status["userdata_path"] = str(cfg.resolved_userdata_path)
        status["site_packages"] = str(cfg.site_packages)
        status["site_packages_exists"] = cfg.site_packages.is_dir()
        status["userdata_exists"] = cfg.resolved_userdata_path.is_dir()
        status["bridge_python"] = str(cfg.resolved_python_executable)
        status["bridge_python_exists"] = cfg.resolved_python_executable.is_file()
        _add_sdk_path(cfg)
        importlib.import_module("xtquant.xttrader")
        importlib.import_module("xtquant.xttype")
        status["xtquant_importable"] = True
    except Exception as exc:  # noqa: BLE001 - status object is for diagnostics.
        status["error"] = str(exc)
    return status


def query_accounts_via_qmt_python(
    config: MiniQmtConfig | None = None,
    account_ids: tuple[str, ...] | list[str] | None = None,
    *,
    positions_sample_size: int = 5,
    timeout: int = 30,
) -> dict[str, Any]:
    """Run a Python-3.6-compatible xtquant bridge through QMT's bundled Python."""
    cfg = config or load_miniqmt_config()
    accounts = tuple(account_ids or cfg.accounts)
    if not accounts:
        raise MiniQmtError("No MiniQMT accounts configured.")
    python_executable = cfg.resolved_python_executable
    if not python_executable.is_file():
        raise MiniQmtSdkNotFound(f"MiniQMT Python executable not found: {python_executable}")

    PATHS.tmp.mkdir(parents=True, exist_ok=True)
    suffix = f"{random.randint(100000, 999999)}"
    script_path = PATHS.tmp / f"miniqmt_bridge_{suffix}.py"
    result_path = PATHS.tmp / f"miniqmt_bridge_{suffix}.json"
    script_path.write_text(
        _bridge_script(
            qmt_root=str(cfg.resolved_qmt_root),
            userdata_path=str(cfg.resolved_userdata_path),
            accounts=list(accounts),
            positions_sample_size=max(0, int(positions_sample_size)),
            result_path=str(result_path),
            session_id=cfg.session_id or random.randint(100000, 999999),
        ),
        encoding="utf-8",
    )
    try:
        completed = subprocess.run(
            [str(python_executable), str(script_path)],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if completed.returncode != 0 and not result_path.is_file():
            raise MiniQmtError(
                "MiniQMT bridge failed before writing a result: "
                f"returncode={completed.returncode} stderr={completed.stderr}"
            )
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        payload["bridge_python"] = str(python_executable)
        payload["read_only"] = True
        payload["live_trading_enabled"] = False
        return payload
    finally:
        with suppress(OSError):
            script_path.unlink()
        with suppress(OSError):
            result_path.unlink()


def probe_data_capabilities_via_qmt_python(
    config: MiniQmtConfig | None = None,
    *,
    timeout: int = 30,
) -> dict[str, Any]:
    """Inspect xtdata functions through QMT's bundled Python.

    This is intentionally read-only.  It does not assume every MiniQMT build
    exposes the same data APIs, especially for financial/fundamental fields.
    """
    cfg = config or load_miniqmt_config()
    python_executable = cfg.resolved_python_executable
    if not python_executable.is_file():
        raise MiniQmtSdkNotFound(f"MiniQMT Python executable not found: {python_executable}")

    PATHS.tmp.mkdir(parents=True, exist_ok=True)
    suffix = f"{random.randint(100000, 999999)}"
    script_path = PATHS.tmp / f"miniqmt_probe_{suffix}.py"
    result_path = PATHS.tmp / f"miniqmt_probe_{suffix}.json"
    script_path.write_text(
        _data_probe_script(
            qmt_root=str(cfg.resolved_qmt_root),
            result_path=str(result_path),
        ),
        encoding="utf-8",
    )
    try:
        completed = subprocess.run(
            [str(python_executable), str(script_path)],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if completed.returncode != 0 and not result_path.is_file():
            raise MiniQmtError(
                "MiniQMT data probe failed before writing a result: "
                f"returncode={completed.returncode} stderr={completed.stderr}"
            )
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        payload["bridge_python"] = str(python_executable)
        payload["read_only"] = True
        return payload
    finally:
        with suppress(OSError):
            script_path.unlink()
        with suppress(OSError):
            result_path.unlink()


def query_financial_data_via_qmt_python(
    codes: list[str] | tuple[str, ...],
    tables: list[str] | tuple[str, ...],
    *,
    start_time: str,
    end_time: str,
    report_type: str = "announce_time",
    config: MiniQmtConfig | None = None,
    row_limit: int = 5,
    timeout: int = 120,
) -> dict[str, Any]:
    """Fetch a small read-only MiniQMT financial data sample."""
    cfg = config or load_miniqmt_config()
    python_executable = cfg.resolved_python_executable
    if not python_executable.is_file():
        raise MiniQmtSdkNotFound(f"MiniQMT Python executable not found: {python_executable}")

    PATHS.tmp.mkdir(parents=True, exist_ok=True)
    suffix = f"{random.randint(100000, 999999)}"
    script_path = PATHS.tmp / f"miniqmt_financial_{suffix}.py"
    result_path = PATHS.tmp / f"miniqmt_financial_{suffix}.json"
    script_path.write_text(
        _financial_query_script(
            qmt_root=str(cfg.resolved_qmt_root),
            result_path=str(result_path),
            codes=list(codes),
            tables=list(tables),
            start_time=start_time,
            end_time=end_time,
            report_type=report_type,
            row_limit=int(row_limit),
        ),
        encoding="utf-8",
    )
    try:
        completed = subprocess.run(
            [str(python_executable), str(script_path)],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if completed.returncode != 0 and not result_path.is_file():
            raise MiniQmtError(
                "MiniQMT financial bridge failed before writing a result: "
                f"returncode={completed.returncode} stderr={completed.stderr}"
            )
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        payload["bridge_python"] = str(python_executable)
        payload["read_only"] = True
        return payload
    finally:
        with suppress(OSError):
            script_path.unlink()
        with suppress(OSError):
            result_path.unlink()


def _financial_query_script(
    *,
    qmt_root: str,
    result_path: str,
    codes: list[str],
    tables: list[str],
    start_time: str,
    end_time: str,
    report_type: str,
    row_limit: int,
) -> str:
    payload = {
        "qmt_root": qmt_root,
        "result_path": result_path,
        "codes": codes,
        "tables": tables,
        "start_time": start_time,
        "end_time": end_time,
        "report_type": report_type,
        "row_limit": row_limit,
    }
    return (
        "# coding: utf-8\n"
        "CONFIG = "
        + repr(payload)
        + "\n"
        + textwrap.dedent(
            r'''
            import json
            import os
            import sys
            import traceback


            def project_to_provider(code):
                text = str(code).strip()
                if "." in text:
                    left, right = text.split(".", 1)
                    if left.lower() in ("sh", "sz"):
                        return right.zfill(6).upper() + "." + left.upper()
                    return left.zfill(6).upper() + "." + right.upper()
                symbol = text.zfill(6)
                market = "SH" if symbol.startswith(("6", "9")) else "SZ"
                return symbol + "." + market


            def frame_summary(frame, row_limit):
                if frame is None:
                    return {"rows": 0, "columns": [], "sample": []}
                data = frame.copy()
                rows = []
                sample = data if row_limit <= 0 else data.head(row_limit)
                for item in sample.to_dict("records"):
                    clean = {}
                    for key, value in item.items():
                        try:
                            json.dumps(value)
                            clean[str(key)] = value
                        except TypeError:
                            clean[str(key)] = str(value)
                    rows.append(clean)
                return {
                    "rows": int(len(data)),
                    "columns": [str(column) for column in data.columns],
                    "sample": rows,
                }


            def main():
                qmt_root = CONFIG["qmt_root"]
                bin_dir = os.path.join(qmt_root, "bin.x64")
                sys.path.insert(0, os.path.join(bin_dir, "Lib", "site-packages"))
                os.chdir(bin_dir)
                from xtquant import xtdata

                provider_codes = [project_to_provider(code) for code in CONFIG["codes"]]
                xtdata.download_financial_data(
                    provider_codes,
                    CONFIG["tables"],
                    CONFIG["start_time"],
                    CONFIG["end_time"],
                )
                data = xtdata.get_financial_data(
                    provider_codes,
                    CONFIG["tables"],
                    CONFIG["start_time"],
                    CONFIG["end_time"],
                    CONFIG["report_type"],
                )
                result = {
                    "ok": True,
                    "requested_codes": provider_codes,
                    "tables": CONFIG["tables"],
                    "start_time": CONFIG["start_time"],
                    "end_time": CONFIG["end_time"],
                    "report_type": CONFIG["report_type"],
                    "data": {},
                }
                for code, table_map in data.items():
                    result["data"][code] = {}
                    for table, frame in table_map.items():
                        result["data"][code][table] = frame_summary(frame, CONFIG["row_limit"])
                return result


            try:
                payload = main()
            except Exception:
                payload = {"ok": False, "error": traceback.format_exc()}
            with open(CONFIG["result_path"], "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
            '''
        )
    )


def _data_probe_script(*, qmt_root: str, result_path: str) -> str:
    payload = {
        "qmt_root": qmt_root,
        "result_path": result_path,
    }
    return (
        "# coding: utf-8\n"
        "CONFIG = "
        + repr(payload)
        + "\n"
        + textwrap.dedent(
            r'''
            import inspect
            import json
            import os
            import sys
            import traceback


            def safe_signature(value):
                try:
                    return str(inspect.signature(value))
                except Exception:
                    return ""


            def main():
                qmt_root = CONFIG["qmt_root"]
                bin_dir = os.path.join(qmt_root, "bin.x64")
                sys.path.insert(0, os.path.join(bin_dir, "Lib", "site-packages"))
                os.chdir(bin_dir)
                from xtquant import xtdata

                names = sorted(name for name in dir(xtdata) if not name.startswith("_"))
                keywords = ("fin", "fund", "finance", "financial", "income", "balance", "cash",
                            "holder", "dividend", "factor", "instrument", "sector", "stock")
                matched = [
                    {
                        "name": name,
                        "signature": safe_signature(getattr(xtdata, name)),
                        "callable": callable(getattr(xtdata, name)),
                    }
                    for name in names
                    if any(keyword in name.lower() for keyword in keywords)
                ]
                explicit = {}
                for name in (
                    "get_financial_data",
                    "download_financial_data",
                    "download_financial_data2",
                    "get_instrument_detail",
                    "get_stock_list_in_sector",
                    "get_sector_list",
                    "get_market_data_ex",
                    "download_history_data",
                ):
                    value = getattr(xtdata, name, None)
                    explicit[name] = {
                        "exists": value is not None,
                        "signature": safe_signature(value) if value is not None else "",
                        "callable": callable(value),
                    }
                return {
                    "ok": True,
                    "xtdata_file": getattr(xtdata, "__file__", ""),
                    "matched_functions": matched,
                    "explicit_functions": explicit,
                    "function_count": len(names),
                }


            try:
                payload = main()
            except Exception:
                payload = {"ok": False, "error": traceback.format_exc()}
            with open(CONFIG["result_path"], "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
            '''
        )
    )


def _bridge_script(
    *,
    qmt_root: str,
    userdata_path: str,
    accounts: list[str],
    positions_sample_size: int,
    result_path: str,
    session_id: int,
) -> str:
    payload = {
        "qmt_root": qmt_root,
        "userdata_path": userdata_path,
        "accounts": accounts,
        "positions_sample_size": positions_sample_size,
        "result_path": result_path,
        "session_id": session_id,
    }
    return (
        "# coding: utf-8\n"
        "CONFIG = "
        + repr(payload)
        + "\n"
        + textwrap.dedent(
            r'''
            import json
            import os
            import sys
            import time
            import traceback


            def mask_account_id(account_id):
                text = "" if account_id is None else str(account_id)
                if len(text) <= 7:
                    return "*" * len(text)
                return text[:3] + "****" + text[-4:]


            def obj_to_dict(obj):
                if obj is None:
                    return None
                if isinstance(obj, (str, int, float, bool)):
                    return obj
                if isinstance(obj, (list, tuple)):
                    return [obj_to_dict(item) for item in obj]
                if isinstance(obj, dict):
                    return {str(key): obj_to_dict(value) for key, value in obj.items()}
                data = {}
                for name in dir(obj):
                    if name.startswith("_"):
                        continue
                    try:
                        value = getattr(obj, name)
                    except Exception:
                        continue
                    if callable(value):
                        continue
                    try:
                        json.dumps(value)
                        data[name] = value
                    except TypeError:
                        data[name] = repr(value)
                return data


            def mask_account_fields(value):
                if isinstance(value, list):
                    return [mask_account_fields(item) for item in value]
                if isinstance(value, dict):
                    result = {}
                    for key, item in value.items():
                        if str(key).lower() in ("account_id", "m_straccountid"):
                            result[key] = mask_account_id(item)
                        else:
                            result[key] = mask_account_fields(item)
                    return result
                return value


            def main():
                qmt_root = CONFIG["qmt_root"]
                bin_dir = os.path.join(qmt_root, "bin.x64")
                sys.path.insert(0, os.path.join(bin_dir, "Lib", "site-packages"))
                os.chdir(bin_dir)

                from xtquant.xttrader import XtQuantTrader
                from xtquant.xttype import StockAccount

                trader = XtQuantTrader(CONFIG["userdata_path"], int(CONFIG["session_id"]))
                trader.start()
                connect_result = trader.connect()
                result = {
                    "ok": connect_result == 0,
                    "session_id": int(CONFIG["session_id"]),
                    "connect_result": connect_result,
                    "accounts": [],
                }
                if connect_result == 0:
                    for account_id in CONFIG["accounts"]:
                        account = StockAccount(str(account_id))
                        subscribe_result = trader.subscribe(account)
                        time.sleep(0.2)
                        asset = obj_to_dict(trader.query_stock_asset(account))
                        positions = obj_to_dict(trader.query_stock_positions(account) or [])
                        result["accounts"].append({
                            "account_id_masked": mask_account_id(account_id),
                            "account_type": getattr(account, "account_type", None),
                            "subscribe_result": subscribe_result,
                            "asset": mask_account_fields(asset),
                            "positions_count": len(positions) if isinstance(positions, list) else 0,
                            "positions_sample": mask_account_fields(
                                positions[:int(CONFIG["positions_sample_size"])]
                                if isinstance(positions, list) else []
                            ),
                        })
                trader.stop()
                return result


            try:
                payload = main()
            except Exception:
                payload = {"ok": False, "error": traceback.format_exc()}
            with open(CONFIG["result_path"], "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
            '''
        )
    )


def object_to_dict(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [object_to_dict(item) for item in value]
    if isinstance(value, dict):
        return {_sanitize_key(key): object_to_dict(item) for key, item in value.items()}
    result = {}
    for name in dir(value):
        if name.startswith("_"):
            continue
        try:
            item = getattr(value, name)
        except Exception:  # noqa: BLE001 - broker objects may expose fragile attrs.
            continue
        if callable(item):
            continue
        result[_sanitize_key(name)] = object_to_dict(item)
    return result


def _sanitize_key(key: Any) -> str:
    return str(key)


def mask_account_fields(value: Any) -> Any:
    if isinstance(value, list):
        return [mask_account_fields(item) for item in value]
    if isinstance(value, dict):
        masked = {}
        for key, item in value.items():
            if str(key).lower() in {"account_id", "m_straccountid", "m_straccountid"}:
                masked[key] = mask_account_id(item)
            else:
                masked[key] = mask_account_fields(item)
        return masked
    return value


class MiniQmtClient:
    """Small read-only facade over xtquant.xttrader.XtQuantTrader."""

    def __init__(self, config: MiniQmtConfig | None = None):
        self.config = config or load_miniqmt_config()
        self._trader = None
        self.session_id = self.config.session_id or random.randint(100000, 999999)
        self._previous_cwd = None

    def __enter__(self) -> "MiniQmtClient":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def connect(self) -> int:
        _add_sdk_path(self.config)
        self._previous_cwd = Path.cwd()
        os.chdir(self.config.bin_dir)
        xttrader = importlib.import_module("xtquant.xttrader")
        trader = xttrader.XtQuantTrader(
            str(self.config.resolved_userdata_path),
            int(self.session_id),
        )
        trader.start()
        result = trader.connect()
        if result != 0:
            with suppress(Exception):
                trader.stop()
            if self._previous_cwd is not None:
                with suppress(OSError):
                    os.chdir(self._previous_cwd)
            raise MiniQmtConnectionError(f"MiniQMT connect failed: {result}")
        self._trader = trader
        return int(result)

    def close(self) -> None:
        if self._trader is not None:
            with suppress(Exception):
                self._trader.stop()
        self._trader = None
        if self._previous_cwd is not None:
            with suppress(OSError):
                os.chdir(self._previous_cwd)
        self._previous_cwd = None

    @property
    def trader(self):
        if self._trader is None:
            raise MiniQmtConnectionError("MiniQMT client is not connected.")
        return self._trader

    def stock_account(self, account_id: str):
        xttype = importlib.import_module("xtquant.xttype")
        return xttype.StockAccount(str(account_id))

    def subscribe(self, account_id: str) -> dict[str, Any]:
        account = self.stock_account(account_id)
        result = self.trader.subscribe(account)
        return {
            "account_id_masked": mask_account_id(account_id),
            "account_type": getattr(account, "account_type", None),
            "subscribe_result": int(result) if isinstance(result, int) else result,
        }

    def query_account(self, account_id: str, *, positions_sample_size: int = 5) -> dict[str, Any]:
        account = self.stock_account(account_id)
        subscribe_result = self.trader.subscribe(account)
        asset = object_to_dict(self.trader.query_stock_asset(account))
        positions = object_to_dict(self.trader.query_stock_positions(account) or [])
        sample = positions[:positions_sample_size] if isinstance(positions, list) else []
        return {
            "account_id_masked": mask_account_id(account_id),
            "account_type": getattr(account, "account_type", None),
            "subscribe_result": subscribe_result,
            "asset": mask_account_fields(asset),
            "positions_count": len(positions) if isinstance(positions, list) else 0,
            "positions_sample": mask_account_fields(sample),
        }

    def query_accounts(
        self,
        account_ids: tuple[str, ...] | list[str] | None = None,
        *,
        positions_sample_size: int = 5,
    ) -> dict[str, Any]:
        accounts = tuple(account_ids or self.config.accounts)
        if not accounts:
            raise MiniQmtError("No MiniQMT accounts configured.")
        return {
            "ok": True,
            "session_id": self.session_id,
            "connect_result": 0,
            "accounts": [
                self.query_account(account_id, positions_sample_size=positions_sample_size)
                for account_id in accounts
            ],
        }

    def place_order(self, *args, **kwargs):
        raise LiveTradingDisabled(
            "Live MiniQMT order placement is disabled in this research system."
        )

    def cancel_order(self, *args, **kwargs):
        raise LiveTradingDisabled(
            "Live MiniQMT order cancellation is disabled in this research system."
        )
