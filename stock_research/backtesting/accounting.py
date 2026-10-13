"""Cash affordability, project order quantities and dated transaction fees."""
import pandas as pd

STAMP_DUTY_REDUCTION_DATE = pd.Timestamp("2023-08-28")


def sell_stamp_duty_rate_for_date(
    trade_date,
    current_rate,
    before_2023_08_28_rate=None,
):
    """Return the configured A-share sell stamp duty for a trade date."""
    if (
        before_2023_08_28_rate is not None
        and pd.Timestamp(trade_date).normalize() < STAMP_DUTY_REDUCTION_DATE
    ):
        return float(before_2023_08_28_rate)
    return float(current_rate)



def _capped_entry_size(current_exposure, planned_size, max_symbol_exposure=0.70):
    """Cap one symbol's combined left/right exposure without changing ratios."""
    available = max(0.0, float(max_symbol_exposure) - float(current_exposure))
    return max(0.0, min(float(planned_size), available))



def _affordable_buy_notional(
    cash, requested_notional, *, commission_rate, minimum_commission, slippage_rate,
):
    """Maximum notional whose principal plus buy costs fits available cash."""
    cash = max(0.0, float(cash))
    requested = max(0.0, float(requested_notional))
    minimum = max(0.0, float(minimum_commission))
    if requested <= 0 or cash <= minimum:
        return 0.0

    def required(notional):
        commission = max(float(notional) * float(commission_rate), minimum)
        return float(notional) + commission + float(notional) * float(slippage_rate)

    low, high = 0.0, requested
    for _ in range(64):
        middle = (low + high) / 2.0
        if required(middle) <= cash:
            low = middle
        else:
            high = middle
    return low



def board_lot_size(code) -> int:
    """Project execution rule: STAR Market 200 shares, all others 100."""
    normalized = str(code).lower()
    return 200 if normalized.startswith("sh.688") or normalized.startswith("688") else 100



def _board_lot_quantity(code, quantity) -> int:
    lot = board_lot_size(code)
    return max(0, int(float(quantity) // lot) * lot)



def _effective_profit_tranches(code, quantity, configured_parts) -> int:
    """Never promise more exit slices than the acquired board lots support."""
    available_board_lots = int(float(quantity) // board_lot_size(code))
    return min(max(1, int(configured_parts)), max(1, available_board_lots))

