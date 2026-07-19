from stock_research.strategies.candidate_interface import (
    MAX_DAILY_CANDIDATES,
    normalize_candidate_snapshots,
)


def _candidate(index, source, score):
    row = {
        "date": "2026-01-02",
        "code": f"sz.{index:06d}",
        "name": f"候选{index}",
        "candidate_source": source,
        "candidate_score": score,
        "quality_score": 80,
        "earnings_yoy": 0.20,
        "mktcap": 200,
        "price_to_value": 1.0,
        "signal_eligible": True,
        "selected_for_trading": True,
        "data_status": "traded",
        "valid_price_bar": True,
    }
    if source == "factor_quant":
        row["right_quant_score"] = score
    return row


def test_daily_candidate_pool_uses_factor_rank_without_core_reservation():
    rows = []
    for index in range(12):
        rows.append(_candidate(index, "value_model", 100 - index))
    for index in range(12, 80):
        rows.append(_candidate(index, "factor_quant", 200 - index))

    selected = normalize_candidate_snapshots({"2026-01-02": rows})["2026-01-02"]

    assert len(selected) == MAX_DAILY_CANDIDATES
    old_value_count = sum(
        "value_model" in item["candidate_source"].split("+")
        for item in selected
    )
    assert old_value_count == 0
    assert selected[0]["selection_rank"] == 1
    assert selected[0]["code"] == "sz.000012"
