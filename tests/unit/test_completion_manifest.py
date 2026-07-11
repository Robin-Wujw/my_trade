import json

import pytest

from stock_research.core.completion_manifest import CompletionManifest


def finish_manifest(tmp_path):
    output = tmp_path / "formula.csv"
    output.write_text("date,count\n", encoding="utf-8")
    manifest = CompletionManifest(tmp_path / "formula33.json")
    manifest.finish(
        observation_date="2026-07-10",
        arguments={"lookback": 21, "market_cap_source": "auto"},
        universe_codes=["sh.600000", "sz.000001"],
        outputs=[output],
        summary={"universe": 2},
        code_version="formula33-v2",
    )
    return manifest, output


def test_matching_completed_manifest_is_reusable_with_stable_ordering(tmp_path):
    manifest, _ = finish_manifest(tmp_path)

    assert manifest.matches(
        observation_date="2026-07-10",
        arguments={"market_cap_source": "auto", "lookback": 21},
        universe_codes=["sz.000001", "sh.600000", "sz.000001"],
        code_version="formula33-v2",
    )

    payload = json.loads(manifest.path.read_text(encoding="utf-8"))
    assert payload["status"] == "completed"
    assert payload["arguments"] == {
        "lookback": 21,
        "market_cap_source": "auto",
    }
    assert payload["universe_size"] == 2
    assert len(payload["universe_sha256"]) == 64
    assert not list(tmp_path.glob(".formula33.json.*.tmp"))


@pytest.mark.parametrize(
    ("overrides"),
    [
        {"observation_date": "2026-07-13"},
        {"arguments": {"lookback": 22, "market_cap_source": "auto"}},
        {"universe_codes": ["sh.600000", "sz.000002"]},
        {"code_version": "formula33-v3"},
    ],
)
def test_changed_identity_invalidates_manifest(tmp_path, overrides):
    manifest, _ = finish_manifest(tmp_path)
    identity = {
        "observation_date": "2026-07-10",
        "arguments": {"lookback": 21, "market_cap_source": "auto"},
        "universe_codes": ["sh.600000", "sz.000001"],
        "code_version": "formula33-v2",
    }
    identity.update(overrides)

    assert not manifest.matches(**identity)


def test_missing_output_or_unreadable_manifest_is_not_reusable(tmp_path):
    manifest, output = finish_manifest(tmp_path)
    output.unlink()

    assert not manifest.matches(
        observation_date="2026-07-10",
        arguments={"lookback": 21, "market_cap_source": "auto"},
        universe_codes=["sh.600000", "sz.000001"],
        code_version="formula33-v2",
    )

    manifest.path.write_text("{not-json", encoding="utf-8")
    assert not manifest.matches(
        observation_date="2026-07-10",
        arguments={"lookback": 21, "market_cap_source": "auto"},
        universe_codes=["sh.600000", "sz.000001"],
        code_version="formula33-v2",
    )
