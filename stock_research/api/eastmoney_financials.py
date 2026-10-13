"""Forward-only Eastmoney financial statement snapshots for GP/A research."""
from __future__ import annotations

from dataclasses import dataclass
import math
import os
import ssl

import pandas as pd
import requests
from requests.adapters import HTTPAdapter

from stock_research.core.paths import PATHS

from .retry import FileRateLimiter, call_with_backoff, is_transient_error


API_URL = "https://datacenter-web.eastmoney.com/api/data/v1/get"
PAGE_SIZE = 500
HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://data.eastmoney.com/",
}
COMMON_FIELDS = (
    "SECUCODE",
    "SECURITY_CODE",
    "NOTICE_DATE",
    "REPORT_DATE",
    "REPORT_TYPE_CODE",
    "DATA_STATE",
)
REPORT_FIELDS = {
    "RPT_DMSK_FN_INCOME": (
        *COMMON_FIELDS,
        "TOTAL_OPERATE_INCOME",
        "OPERATE_COST",
    ),
    "RPT_DMSK_FN_BALANCE": (
        *COMMON_FIELDS,
        "TOTAL_ASSETS",
    ),
}
_RATE_LIMITER = FileRateLimiter(
    float(os.environ.get("EASTMONEY_MIN_INTERVAL", "0.15")),
    PATHS.tmp / "eastmoney_financial_rate_limit.lock",
)


class EastmoneyFinancialError(RuntimeError):
    """Eastmoney returned an invalid or internally inconsistent snapshot."""


def _create_system_trust_context() -> ssl.SSLContext:
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.check_hostname = True
    context.verify_mode = ssl.CERT_REQUIRED
    context.load_default_certs(purpose=ssl.Purpose.SERVER_AUTH)
    return context


class _SystemTrustAdapter(HTTPAdapter):
    def __init__(self, ssl_context: ssl.SSLContext, *args, **kwargs):
        self._ssl_context = ssl_context
        super().__init__(*args, **kwargs)

    def init_poolmanager(self, *args, **kwargs):
        kwargs.setdefault("ssl_context", self._ssl_context)
        return super().init_poolmanager(*args, **kwargs)

    def proxy_manager_for(self, proxy, **proxy_kwargs):
        proxy_kwargs.setdefault("ssl_context", self._ssl_context)
        return super().proxy_manager_for(proxy, **proxy_kwargs)


def _build_http_session() -> requests.Session:
    session = requests.Session()
    if os.name == "nt":
        session.mount("https://", _SystemTrustAdapter(_create_system_trust_context()))
    return session


@dataclass(frozen=True)
class PageMetadata:
    report_name: str
    report_date: str
    row_count: int
    page_count: int
    page_size: int = PAGE_SIZE


def _report_date(value: str) -> str:
    parsed = pd.to_datetime(str(value), errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"invalid financial report date: {value}")
    return parsed.strftime("%Y-%m-%d")


def _page_result(payload: dict, *, report_name: str, page: int) -> dict:
    if not isinstance(payload, dict):
        raise EastmoneyFinancialError(
            f"invalid Eastmoney payload: report={report_name} page={page}"
        )
    if payload.get("success") is False:
        raise EastmoneyFinancialError(
            "Eastmoney request failed: "
            f"report={report_name} page={page} "
            f"code={payload.get('code')} message={payload.get('message')}"
        )
    result = payload.get("result")
    if not isinstance(result, dict):
        raise EastmoneyFinancialError(
            f"Eastmoney result is missing: report={report_name} page={page}"
        )
    return result


def fetch_report_snapshot(
    report_name: str,
    report_date: str,
    *,
    session: requests.Session | None = None,
    retries: int = 4,
    retry_delay: float = 1.0,
    timeout: float = 20.0,
) -> tuple[pd.DataFrame, PageMetadata]:
    """Fetch one full report period while checking pagination consistency."""
    if report_name not in REPORT_FIELDS:
        raise ValueError(f"unsupported Eastmoney report: {report_name}")
    period = _report_date(report_date)
    owned_session = session is None
    http = session or _build_http_session()
    rows = []
    declared_count = declared_pages = None
    try:
        page = 1
        while declared_pages is None or page <= declared_pages:
            params = {
                "sortColumns": "SECURITY_CODE",
                "sortTypes": "1",
                "pageSize": str(PAGE_SIZE),
                "pageNumber": str(page),
                "reportName": report_name,
                "columns": ",".join(REPORT_FIELDS[report_name]),
                "filter": f"(REPORT_DATE='{period}')",
            }

            def request_once():
                _RATE_LIMITER.wait()
                response = http.get(
                    API_URL,
                    params=params,
                    headers=HEADERS,
                    timeout=timeout,
                )
                response.raise_for_status()
                return response.json()

            payload = call_with_backoff(
                request_once,
                f"Eastmoney {report_name} page {page}",
                retries=retries,
                retry_delay=retry_delay,
                retry_if=is_transient_error,
            )
            result = _page_result(payload, report_name=report_name, page=page)
            try:
                current_count = int(result["count"])
                current_pages = int(result["pages"])
            except (KeyError, TypeError, ValueError) as exc:
                raise EastmoneyFinancialError(
                    "Eastmoney pagination metadata is invalid: "
                    f"report={report_name} page={page}"
                ) from exc
            if current_count <= 0 or current_pages <= 0:
                raise EastmoneyFinancialError(
                    f"Eastmoney returned an empty report: {report_name} {period}"
                )
            if declared_count is None:
                declared_count = current_count
                declared_pages = current_pages
                expected_pages = math.ceil(declared_count / PAGE_SIZE)
                if declared_pages != expected_pages:
                    raise EastmoneyFinancialError(
                        "Eastmoney page count is inconsistent with row count: "
                        f"report={report_name} count={declared_count} "
                        f"pages={declared_pages} expected={expected_pages}"
                    )
            elif (
                current_count != declared_count
                or current_pages != declared_pages
            ):
                raise EastmoneyFinancialError(
                    "Eastmoney snapshot changed during pagination: "
                    f"report={report_name} page={page}"
                )
            page_rows = result.get("data")
            if not isinstance(page_rows, list) or not page_rows:
                raise EastmoneyFinancialError(
                    f"Eastmoney page data is empty: report={report_name} page={page}"
                )
            rows.extend(page_rows)
            page += 1
    finally:
        if owned_session:
            http.close()

    frame = pd.DataFrame(rows)
    missing = sorted(set(REPORT_FIELDS[report_name]) - set(frame.columns))
    if missing:
        raise EastmoneyFinancialError(
            f"Eastmoney report fields are missing: {report_name} {missing}"
        )
    if len(frame) != declared_count:
        raise EastmoneyFinancialError(
            "Eastmoney row count changed during pagination: "
            f"report={report_name} expected={declared_count} actual={len(frame)}"
        )
    identity = [
        "SECUCODE", "REPORT_DATE", "REPORT_TYPE_CODE", "DATA_STATE",
    ]
    duplicates = frame.duplicated(identity, keep=False)
    if duplicates.any():
        sample = frame.loc[duplicates, identity].head(10).to_dict("records")
        raise EastmoneyFinancialError(
            f"Eastmoney report contains duplicate identities: sample={sample}"
        )
    frame = frame.sort_values(identity, kind="mergesort").reset_index(drop=True)
    return frame, PageMetadata(
        report_name=report_name,
        report_date=period,
        row_count=int(declared_count),
        page_count=int(declared_pages),
    )


def normalize_report_snapshot(
    frame: pd.DataFrame,
    *,
    report_name: str,
    report_date: str,
) -> pd.DataFrame:
    """Map explicit provider fields to the repository financial schema."""
    if report_name not in REPORT_FIELDS:
        raise ValueError(f"unsupported Eastmoney report: {report_name}")
    missing = sorted(set(REPORT_FIELDS[report_name]) - set(frame.columns))
    if missing:
        raise EastmoneyFinancialError(
            f"Eastmoney report fields are missing: {report_name} {missing}"
        )
    period = _report_date(report_date)
    data = frame.copy()
    observed_period = pd.to_datetime(
        data["REPORT_DATE"], errors="coerce",
    ).dt.strftime("%Y-%m-%d")
    if observed_period.isna().any() or not observed_period.eq(period).all():
        raise EastmoneyFinancialError(
            f"Eastmoney report period mismatch: report={report_name} expected={period}"
        )
    codes = data["SECUCODE"].fillna("").astype(str).str.strip().str.upper()
    valid_codes = codes.str.fullmatch(r"\d{6}\.(SH|SZ|BJ)")
    if not valid_codes.all():
        raise EastmoneyFinancialError(
            "Eastmoney report contains invalid A-share codes: "
            f"{codes[~valid_codes].head(10).tolist()}"
        )
    report_type = data["REPORT_TYPE_CODE"].fillna("").astype(str).str.strip()
    if not report_type.isin({"001", "1"}).all():
        raise EastmoneyFinancialError(
            "Eastmoney report contains unsupported report types: "
            f"{sorted(report_type.unique())}"
        )
    announcement = pd.to_datetime(data["NOTICE_DATE"], errors="coerce")
    if announcement.isna().any():
        raise EastmoneyFinancialError(
            f"Eastmoney report has missing announcement dates: {report_name}"
        )
    normalized = pd.DataFrame({
        "ts_code": codes,
        "ann_date": announcement.dt.strftime("%Y%m%d"),
        "end_date": observed_period.str.replace("-", "", regex=False),
        "report_type": "1",
        "data_state": data["DATA_STATE"].astype(str),
        "provider_security_code": data["SECURITY_CODE"].astype(str),
        "provider_report_type_code": report_type,
    })
    if report_name == "RPT_DMSK_FN_INCOME":
        normalized["total_revenue"] = pd.to_numeric(
            data["TOTAL_OPERATE_INCOME"], errors="coerce",
        )
        normalized["oper_cost"] = pd.to_numeric(
            data["OPERATE_COST"], errors="coerce",
        )
    else:
        normalized["total_assets"] = pd.to_numeric(
            data["TOTAL_ASSETS"], errors="coerce",
        )
    return normalized.sort_values("ts_code", kind="mergesort").reset_index(drop=True)


def validate_gross_profitability_pair(
    income: pd.DataFrame,
    balance: pd.DataFrame,
    *,
    minimum_rows: int = 5000,
    minimum_complete_ratio: float = 0.97,
) -> dict:
    """Fail closed unless both annual statements form a full-market snapshot."""
    for name, frame in (("income", income), ("balancesheet", balance)):
        if len(frame) < int(minimum_rows):
            raise EastmoneyFinancialError(
                f"Eastmoney {name} coverage is below minimum: "
                f"rows={len(frame)} minimum={minimum_rows}"
            )
        duplicates = frame["ts_code"].duplicated(keep=False)
        if duplicates.any():
            raise EastmoneyFinancialError(
                f"Eastmoney {name} contains duplicate stock codes"
            )
    income_codes = set(income["ts_code"])
    balance_codes = set(balance["ts_code"])
    if income_codes != balance_codes:
        raise EastmoneyFinancialError(
            "Eastmoney income/balance universes differ: "
            f"income_only={len(income_codes - balance_codes)} "
            f"balance_only={len(balance_codes - income_codes)}"
        )
    positive_assets = pd.to_numeric(
        balance.set_index("ts_code")["total_assets"], errors="coerce",
    ).gt(0)
    income_metrics = income.set_index("ts_code")[[
        "total_revenue", "oper_cost",
    ]].apply(pd.to_numeric, errors="coerce").notna().all(axis=1)
    complete = income_metrics & positive_assets.reindex(income_metrics.index).fillna(False)
    complete_ratio = float(complete.mean()) if len(complete) else 0.0
    if complete_ratio < float(minimum_complete_ratio):
        raise EastmoneyFinancialError(
            "Eastmoney GP/A field coverage is below minimum: "
            f"ratio={complete_ratio:.6f} minimum={minimum_complete_ratio:.6f}"
        )
    announcement = pd.concat([
        pd.to_datetime(income["ann_date"], format="%Y%m%d", errors="coerce"),
        pd.to_datetime(balance["ann_date"], format="%Y%m%d", errors="coerce"),
    ], ignore_index=True)
    return {
        "income_rows": int(len(income)),
        "balancesheet_rows": int(len(balance)),
        "common_codes": int(len(income_codes)),
        "beijing_codes": int(sum(code.endswith(".BJ") for code in income_codes)),
        "income_missing_total_revenue": int(income["total_revenue"].isna().sum()),
        "income_missing_oper_cost": int(income["oper_cost"].isna().sum()),
        "balance_missing_or_nonpositive_assets": int((~positive_assets).sum()),
        "complete_gpa_codes": int(complete.sum()),
        "complete_gpa_ratio": complete_ratio,
        "latest_announcement_date": announcement.max().strftime("%Y-%m-%d"),
        "minimum_rows_gate": int(minimum_rows),
        "minimum_complete_ratio_gate": float(minimum_complete_ratio),
    }
