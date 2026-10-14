"""Read-only projection of queued exits; this module never executes an order."""


def pending_exit_intents(states, observation_date, *, candidate_exits=(), weak_exits=(),
                         left_exits=None, left_quota_exits=None):
    result = []
    sources = [
        (dict.fromkeys(candidate_exits, "candidate_removed"), "candidate_exit"),
        (dict.fromkeys(weak_exits, "weak_position_replacement"), "weak_exit"),
        (left_exits or {}, "left_value_exit"),
        (left_quota_exits or {}, "left_quota_exit"),
    ]
    for reasons, kind in sources:
        for code, reason in sorted(reasons.items()):
            result.append({"code": code, "signal": kind, "reason": reason, "batch": None})
    for code, state in sorted(states.items()):
        for batch, reason in sorted(state.pending_lot_exits.items()):
            result.append({"code": code, "signal": "lot_exit", "batch": batch, "reason": reason})
        for signal in sorted(state.pending_profit_ids):
            result.append({"code": code, "signal": "profit_tranche_exit", "batch": None, "reason": signal})
    for item in result:
        item.update(observation_date=observation_date, side="sell", quantity=None,
                    status="conditional_next_session_recheck", signal_detected=True,
                    triggered=False, filled=False, research_only=True)
    return result
