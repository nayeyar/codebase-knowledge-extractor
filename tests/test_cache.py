import json
from pathlib import Path

import pytest
from pydantic import BaseModel

from codebase_analyzer.cache import (
    CACHE_FORMAT_VERSION,
    JsonCache,
    atomic_write_text,
)


class CachedValue(BaseModel):
    value: str


def test_cache_use_round_trip_and_separate_namespaces(tmp_path: Path) -> None:
    cache = JsonCache(tmp_path, "use")
    payload = {"b": 2, "a": 1}
    map_key = cache.key("map", payload)
    reduce_key = cache.key("reduce", payload)

    assert map_key != reduce_key
    assert cache.load("map", map_key, CachedValue).status == "miss"

    map_path = cache.save("map", map_key, CachedValue(value="map"))
    reduce_path = cache.save("reduce", reduce_key, CachedValue(value="reduce"))

    assert map_path == tmp_path / "map" / f"{map_key}.json"
    assert reduce_path == tmp_path / "reduce" / f"{reduce_key}.json"
    assert cache.load("map", map_key, CachedValue).value == CachedValue(value="map")
    assert cache.load("reduce", reduce_key, CachedValue).value == CachedValue(value="reduce")


def test_cache_key_is_canonical() -> None:
    cache = JsonCache(Path("unused"))
    assert cache.key("map", {"a": 1, "b": 2}) == cache.key(
        "map",
        {"b": 2, "a": 1},
    )


def test_refresh_skips_reads_and_replaces_after_success(tmp_path: Path) -> None:
    use_cache = JsonCache(tmp_path, "use")
    key = use_cache.key("map", {"input": "same"})
    use_cache.save("map", key, CachedValue(value="old"))

    refresh_cache = JsonCache(tmp_path, "refresh")
    assert refresh_cache.load("map", key, CachedValue).status == "miss"
    refresh_cache.save("map", key, CachedValue(value="new"))

    assert use_cache.load("map", key, CachedValue).value == CachedValue(value="new")


def test_off_cache_neither_reads_nor_writes(tmp_path: Path) -> None:
    use_cache = JsonCache(tmp_path, "use")
    key = use_cache.key("map", {"input": "same"})
    use_cache.save("map", key, CachedValue(value="existing"))

    off_cache = JsonCache(tmp_path, "off")
    assert off_cache.load("map", key, CachedValue).status == "miss"
    assert off_cache.save("map", key, CachedValue(value="replacement")) is None
    assert use_cache.load("map", key, CachedValue).value == CachedValue(value="existing")


@pytest.mark.parametrize(
    "envelope",
    [
        "{",
        "[]",
        json.dumps({"cache_format_version": "wrong"}),
        json.dumps(
            {
                "cache_format_version": CACHE_FORMAT_VERSION,
                "namespace": "reduce",
                "key": "wrong",
                "result": {"value": "x"},
            }
        ),
        json.dumps(
            {
                "cache_format_version": CACHE_FORMAT_VERSION,
                "namespace": "map",
                "key": "KEY",
                "result": {"not_value": "x"},
            }
        ),
    ],
)
def test_corrupt_truncated_and_wrong_envelopes_are_misses(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    envelope: str,
) -> None:
    cache = JsonCache(tmp_path, "use")
    key = "KEY"
    path = cache.path("map", key)
    path.parent.mkdir(parents=True)
    path.write_text(envelope, encoding="utf-8")

    lookup = cache.load("map", key, CachedValue)

    assert lookup.status == "corrupt"
    assert lookup.value is None
    assert "Ignoring corrupt map cache entry" in caplog.text


def test_atomic_write_replaces_content_and_cleans_temporary_files(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "result.json"
    atomic_write_text(target, "first\n")
    atomic_write_text(target, "second\n")

    assert target.read_text(encoding="utf-8") == "second\n"
    assert list(target.parent.glob(f".{target.name}.*.tmp")) == []


def test_atomic_write_preserves_old_target_when_replace_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "result.json"
    target.write_text("valid old report\n", encoding="utf-8")
    original_replace = Path.replace

    def fail_for_target(path: Path, destination: Path) -> Path:
        if destination == target:
            raise OSError("simulated replace failure")
        return original_replace(path, destination)

    monkeypatch.setattr(Path, "replace", fail_for_target)

    with pytest.raises(OSError, match="simulated replace failure"):
        atomic_write_text(target, "incomplete new report\n")

    assert target.read_text(encoding="utf-8") == "valid old report\n"
    assert list(tmp_path.glob(f".{target.name}.*.tmp")) == []
