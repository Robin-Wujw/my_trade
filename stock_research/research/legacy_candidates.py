"""Legacy research candidate exceptions, excluded by frozen value/factor inputs."""
import numpy as np
import pandas as pd
from stock_research.backtesting.signals import _enabled

def candidate_is_daily_quota_diagnostic(candidate):
    reason = str((candidate or {}).get("candidate_failure_reason") or "").strip()
    return reason.startswith(
        "not_selected_for_trading: factor_rank_not_in_observation_pool"
    ) or reason.startswith(
        "not_selected_for_trading: daily_top10_quota_or_core_reservation"
    )


def candidate_has_exceptional_right_profile(candidate):
    """Keep the over-quota escape hatch only for clear right-side leaders."""
    if not candidate:
        return False
    sources = set(str(candidate.get("candidate_source") or "").split("+"))
    if not sources & {
        "standard_mainline", "growth_leadership",
        "quant_right", "factor_quant",
    }:
        return False
    right_quant_score = pd.to_numeric(
        candidate.get("right_quant_score"), errors="coerce",
    )
    if pd.notna(right_quant_score) and float(right_quant_score) >= 95.0:
        return True
    leadership = pd.to_numeric(candidate.get("leadership_score"), errors="coerce")
    return_20 = pd.to_numeric(candidate.get("return_20d"), errors="coerce")
    return_60 = pd.to_numeric(candidate.get("return_60d"), errors="coerce")
    drawdown_60 = pd.to_numeric(candidate.get("drawdown_60"), errors="coerce")
    avg_amount_20 = pd.to_numeric(candidate.get("avg_amount_20"), errors="coerce")
    visible_right_strength = (
        pd.notna(leadership)
        and float(leadership) >= 23.0
    ) or (
        pd.notna(right_quant_score)
        and float(right_quant_score) >= 75.0
    )
    return bool(
        visible_right_strength
        and pd.notna(return_20)
        and pd.notna(return_60)
        and pd.notna(drawdown_60)
        and pd.notna(avg_amount_20)
        and 0.05 <= float(return_20) <= 0.60
        and float(return_60) >= 0.35
        and float(drawdown_60) >= -0.12
        and float(avg_amount_20) >= 5_000_000_000.0
    )


def pe_pb_candidate_can_add(candidate, *, falsification_reason=""):
    quality = pd.to_numeric(candidate.get("quality_score"), errors="coerce")
    market_cap = pd.to_numeric(candidate.get("mktcap"), errors="coerce")
    # PE/PB 低估通道：低估比率 = 当前估值/历史年度低估均值 <= 1.0，
    # 质量/市值门槛放宽（quality>=60、mktcap>=100亿）。
    pe_ratio = pd.to_numeric(candidate.get("pe_ratio"), errors="coerce")
    pb_ratio = pd.to_numeric(candidate.get("pb_ratio"), errors="coerce")
    ratios = [v for v in (pe_ratio, pb_ratio) if pd.notna(v)]
    low_ratio = min(ratios) if ratios else np.nan
    return (
        _enabled(candidate.get("selected_for_trading"), default=True)
        and _enabled(candidate.get("signal_eligible"), default=True)
        and _enabled(candidate.get("allow_left"), default=False)
        and pd.notna(low_ratio)
        and 0.0 < float(low_ratio) <= 1.0
        and pd.notna(quality)
        and float(quality) >= 60.0
        and pd.notna(market_cap)
        and float(market_cap) >= 100.0
        and not falsification_reason
    )

