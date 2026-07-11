"""Atomic completion markers for safely reusable pipeline outputs."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from datetime import date, datetime, timezone
from enum import Enum
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Union


PathLike = Union[str, os.PathLike]


def _normalize(value: Any) -> Any:
    """Convert supported values into a deterministic JSON representation."""
    if is_dataclass(value) and not isinstance(value, type):
        return _normalize(asdict(value))
    if isinstance(value, Enum):
        return _normalize(value.value)
    if isinstance(value, Mapping):
        return {
            str(key): _normalize(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (set, frozenset)):
        normalized = [_normalize(item) for item in value]
        return sorted(normalized, key=_canonical_json)
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("manifest values must not contain NaN or infinity")
        return value

    item_method = getattr(value, "item", None)
    if callable(item_method):
        converted = item_method()
        if converted is not value:
            return _normalize(converted)
    raise TypeError(f"unsupported manifest value: {type(value).__name__}")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _normalize(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def stable_hash(value: Any) -> str:
    """Return a SHA-256 fingerprint independent of mapping key order."""
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _observation_text(value: Any) -> str:
    if isinstance(value, (datetime, date)):
        return value.isoformat()[:10]
    return str(value).strip()


def _universe_identity(codes: Iterable[Any]) -> list[str]:
    normalized = set()
    for code in codes:
        if code is None:
            continue
        text = str(code).strip()
        if text:
            normalized.add(text)
    return sorted(normalized)


def _resolved_outputs(outputs: Iterable[PathLike]) -> list[str]:
    return [str(Path(path).resolve()) for path in outputs]


class CompletionManifest:
    """Persist and validate the identity of one fully completed pipeline run."""

    def __init__(self, path: PathLike):
        self.path = Path(path)

    def read(self) -> dict[str, Any]:
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def finish(
        self,
        *,
        observation_date: Any,
        arguments: Mapping[str, Any],
        universe_codes: Iterable[Any],
        outputs: Iterable[PathLike],
        summary: Mapping[str, Any],
        code_version: str,
    ) -> None:
        normalized_arguments = _normalize(arguments)
        universe = _universe_identity(universe_codes)
        payload = {
            "status": "completed",
            "observation_date": _observation_text(observation_date),
            "arguments": normalized_arguments,
            "arguments_sha256": stable_hash(normalized_arguments),
            "universe_sha256": stable_hash(universe),
            "universe_size": len(universe),
            "code_version": str(code_version),
            "outputs": _resolved_outputs(outputs),
            "summary": _normalize(summary),
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        self._atomic_write(payload)

    def matches(
        self,
        *,
        observation_date: Any,
        arguments: Mapping[str, Any],
        universe_codes: Iterable[Any],
        code_version: str,
    ) -> bool:
        payload = self.read()
        if payload.get("status") != "completed":
            return False
        if payload.get("observation_date") != _observation_text(observation_date):
            return False
        if payload.get("code_version") != str(code_version):
            return False

        normalized_arguments = _normalize(arguments)
        if payload.get("arguments") != normalized_arguments:
            return False
        if payload.get("arguments_sha256") != stable_hash(normalized_arguments):
            return False
        universe = _universe_identity(universe_codes)
        if payload.get("universe_sha256") != stable_hash(universe):
            return False
        if payload.get("universe_size") != len(universe):
            return False

        outputs = payload.get("outputs")
        if not isinstance(outputs, list) or not outputs:
            return False
        return all(isinstance(path, str) and Path(path).is_file() for path in outputs)

    def _atomic_write(self, payload: Mapping[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        file_descriptor, temporary_name = tempfile.mkstemp(
            dir=str(self.path.parent),
            prefix=f".{self.path.name}.",
            suffix=".tmp",
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(
                    payload,
                    handle,
                    ensure_ascii=False,
                    sort_keys=True,
                    indent=2,
                    allow_nan=False,
                )
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, self.path)
        finally:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass
