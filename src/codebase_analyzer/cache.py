from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Generic, Literal, TypeVar

from pydantic import BaseModel, ValidationError

CACHE_FORMAT_VERSION = "1"
CacheMode = Literal["use", "refresh", "off"]
SchemaT = TypeVar("SchemaT", bound=BaseModel)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AnalysisCacheContext:
    repository_identity: str
    requested_ref: str | None
    resolved_commit: str | None
    source_fingerprint: str
    analyzer_version: str

    def as_key_data(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CacheLookup(Generic[SchemaT]):
    value: SchemaT | None
    status: Literal["hit", "miss", "corrupt"]
    path: Path


class JsonCache:
    """Validated, content-addressed JSON storage for map and reduce results."""

    def __init__(self, root: Path, mode: CacheMode = "use") -> None:
        self.root = root
        self.mode = mode

    def key(self, namespace: str, payload: dict[str, object]) -> str:
        serialized = canonical_json(
            {
                "cache_format_version": CACHE_FORMAT_VERSION,
                "namespace": namespace,
                "payload": payload,
            }
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def load(self, namespace: str, key: str, schema: type[SchemaT]) -> CacheLookup[SchemaT]:
        path = self.path(namespace, key)
        if self.mode != "use" or not path.exists():
            return CacheLookup(value=None, status="miss", path=path)

        try:
            envelope = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(envelope, dict):
                raise ValueError("cache envelope is not an object")
            if envelope.get("cache_format_version") != CACHE_FORMAT_VERSION:
                raise ValueError("cache format version does not match")
            if envelope.get("namespace") != namespace or envelope.get("key") != key:
                raise ValueError("cache identity does not match its path")
            value = schema.model_validate(envelope.get("result"))
        except (OSError, UnicodeError, json.JSONDecodeError, ValidationError, ValueError) as exc:
            logger.warning("Ignoring corrupt %s cache entry %s: %s", namespace, path, exc)
            return CacheLookup(value=None, status="corrupt", path=path)
        return CacheLookup(value=value, status="hit", path=path)

    def save(self, namespace: str, key: str, value: BaseModel) -> Path | None:
        if self.mode == "off":
            return None
        target = self.path(namespace, key)
        content = (
            json.dumps(
                {
                    "cache_format_version": CACHE_FORMAT_VERSION,
                    "namespace": namespace,
                    "key": key,
                    "result": value.model_dump(mode="json"),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        atomic_write_text(target, content)
        return target

    def path(self, namespace: str, key: str) -> Path:
        return self.root / namespace / f"{key}.json"


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def atomic_write_text(path: Path, content: str) -> None:
    """Replace a text file atomically without exposing a partially written target."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
