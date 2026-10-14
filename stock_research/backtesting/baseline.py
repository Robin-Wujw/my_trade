"""Versioned audited strategy settings, shared by CLI and forward paper runs.

The low-level engine retains its historical research API. Production adapters
must pass explicit settings from this profile; input units/dates are separate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys

from stock_research.core.paths import PATHS


BASELINE_PATH = PATHS.project_root / "config" / "strategy_baseline.json"
INPUT_PARAMETERS = {"requested_start", "end_date", "price_frames_prepared", "price_amount_multiplier"}


def load_baseline() -> dict:
    document = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    if document.get("schema_version") != 1 or document.get("research_only") is not True:
        raise ValueError("unsupported or unqualified baseline configuration")
    if document.get("paper_only") is not True:
        raise ValueError("baseline must remain paper-only")
    return document


def baseline_engine_parameters() -> dict:
    return load_baseline()["engine_parameters"]


def configure_parser_defaults(parser):
    parameters = baseline_engine_parameters()
    destinations = {action.dest for action in parser._actions}
    parser.set_defaults(**{
        key: value for key, value in parameters.items()
        if key in destinations and key not in INPUT_PARAMETERS
    })
    return parser


class BaselineArgumentParser(argparse.ArgumentParser):
    def parse_args(self, args=None, namespace=None):
        tokens = list(sys.argv[1:] if args is None else args)
        parsed = super().parse_args(tokens, namespace)
        explicit_multiplier = any(
            token.split("=", 1)[0] == "--right-entry-size-multiplier"
            for token in tokens
        )
        # Risk sizing is an explicit research alternative to the baseline's
        # multiplier. Keep older risk-budget invocations working; an explicit
        # incompatible multiplier still reaches the engine's validation.
        if getattr(parsed, "right_entry_risk_budget_pct", None) is not None and not explicit_multiplier:
            parsed.right_entry_size_multiplier = 1.0
        return parsed


def configuration_record(args) -> dict:
    document = load_baseline()
    expected = document["engine_parameters"]
    values = vars(args)
    actual = {key: values[key] for key in expected if key in values}
    actual.update({
        "requested_start": args.start_date,
        "max_positions": None if args.max_positions == 0 else args.max_positions,
        "signals_effective_next_day": True,
        "auto_price_structure": not args.no_auto_price_structure,
        "allow_structure_pullback": not args.no_structure_pullback,
        "allow_pullback_pilot": not args.disable_pullback_pilot,
        "enable_breadth_reversal_half_pullback": (
            args.enable_breadth_reversal_channel or args.factor_only_breadth_reversal_only
        ),
    })
    deviations = {
        key: {"baseline": value, "effective": actual.get(key)}
        for key, value in expected.items()
        if key not in INPUT_PARAMETERS and actual.get(key) != value
    }
    return {
        "version": document["version"],
        "path": str(BASELINE_PATH),
        "sha256": hashlib.sha256(BASELINE_PATH.read_bytes()).hexdigest(),
        "baseline_model_id": document["model_id"],
        "effective_engine_parameters": actual,
        "parameter_deviations": deviations,
        "matches_baseline_parameters": not deviations,
        "candidate_model_verification": "requires_candidate_manifest",
        "research_only": True,
    }
