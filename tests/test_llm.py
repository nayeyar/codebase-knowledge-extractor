import json
import logging
import threading
from dataclasses import replace
from pathlib import Path

import httpx
import pytest
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI as RealChatOpenAI
from openai import APIConnectionError, APIStatusError
from pydantic import ValidationError

import codebase_analyzer.llm as llm_module
from codebase_analyzer.cache import AnalysisCacheContext
from codebase_analyzer.llm import (
    OllamaKnowledgeModel,
    OpenAIKnowledgeModel,
    normalize_ollama_base_url,
)
from codebase_analyzer.models import (
    BatchChunkAnalysis,
    ChunkAnalysis,
    ProjectSynthesis,
)
from codebase_analyzer.token_budget import CodeChunk


class ScriptedRunnable:
    def __init__(self, responses: list[object]) -> None:
        self.responses = responses
        self.calls = 0

    def invoke(self, messages: list[object]) -> object:
        self.calls += 1
        if not self.responses:
            raise AssertionError("unexpected model call")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        if callable(response):
            return response(messages)
        return response


class FakeChatModel:
    def __init__(
        self,
        runnables: dict[type[object], ScriptedRunnable],
    ) -> None:
        self.runnables = runnables

    def with_structured_output(self, schema: type[object], *, method: str) -> ScriptedRunnable:
        assert method == "json_schema"
        return self.runnables[schema]


def _install_fake_chat(
    monkeypatch: pytest.MonkeyPatch,
    *,
    map_responses: list[object] | None = None,
    reduce_responses: list[object] | None = None,
) -> tuple[
    dict[str, object],
    ScriptedRunnable,
    ScriptedRunnable,
]:
    captured: dict[str, object] = {}
    map_runnable = ScriptedRunnable(map_responses or [])
    reduce_runnable = ScriptedRunnable(reduce_responses or [])
    fake = FakeChatModel(
        {
            BatchChunkAnalysis: map_runnable,
            ProjectSynthesis: reduce_runnable,
        }
    )

    def build_fake(**kwargs: object) -> FakeChatModel:
        captured.update(kwargs)
        return fake

    monkeypatch.setattr(llm_module, "ChatOpenAI", build_fake)
    return captured, map_runnable, reduce_runnable


def _chunk(name: str = "A.java") -> CodeChunk:
    return CodeChunk(
        chunk_id=f"{name}::part-1",
        file_path=name,
        part=1,
        content=f"class {Path(name).stem} {{}}",
        token_count=5,
    )


def _batch_result(*chunks: CodeChunk, purpose: str = "purpose") -> BatchChunkAnalysis:
    return BatchChunkAnalysis(
        chunks=[
            ChunkAnalysis(
                chunk_id=chunk.chunk_id,
                purpose=purpose,
                responsibilities=["responsibility"],
                noteworthy_aspects=["noteworthy"],
                method_descriptions=[],
            )
            for chunk in chunks
        ]
    )


def _synthesis(purpose: str = "project") -> ProjectSynthesis:
    return ProjectSynthesis(
        purpose=purpose,
        overview="overview",
        primary_capabilities=["capability"],
        architecture_style="layered",
        architecture_layers=[],
        typical_request_flow=["request"],
        noteworthy_aspects=["noteworthy"],
        assumptions_and_limitations=["limitation"],
    )


def _context(**overrides: object) -> AnalysisCacheContext:
    values: dict[str, object] = {
        "repository_identity": "https://example.test/repository.git",
        "requested_ref": "main",
        "resolved_commit": "abc123",
        "source_fingerprint": "fingerprint",
        "analyzer_version": "0.1.0",
    }
    values.update(overrides)
    return AnalysisCacheContext(**values)


def _ollama(
    tmp_path: Path,
    *,
    cache_mode: str = "use",
    max_retries: int = 0,
    model: str = "qwen3.5-32k",
    base_url: str = "http://localhost:11434",
    max_output_tokens: int = 8_000,
) -> OllamaKnowledgeModel:
    return OllamaKnowledgeModel(
        model=model,
        base_url=base_url,
        api_key="ollama",
        cache_dir=tmp_path,
        cache_mode=cache_mode,
        max_output_tokens=max_output_tokens,
        read_timeout_seconds=7_200,
        connect_timeout_seconds=10,
        max_retries=max_retries,
    )


def test_normalize_ollama_base_url() -> None:
    assert normalize_ollama_base_url("http://localhost:11434") == "http://localhost:11434/v1"
    assert normalize_ollama_base_url("https://ollama.example/v1/") == "https://ollama.example/v1"


def test_ollama_adapter_uses_explicit_timeouts_and_disables_sdk_retries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured, _, _ = _install_fake_chat(monkeypatch)

    model = _ollama(tmp_path)

    timeout = captured["timeout"]
    assert isinstance(timeout, httpx.Timeout)
    assert timeout.connect == 10
    assert timeout.read == 7_200
    assert timeout.write == 60
    assert timeout.pool == 10
    assert captured["max_retries"] == 0
    assert captured["extra_body"] == {"max_tokens": 8_000}
    assert "max_tokens" not in {key for key in captured if key != "extra_body"}
    assert captured["temperature"] == 0
    assert captured["reasoning_effort"] == "none"
    assert model.max_retries == 0


def test_openai_adapter_uses_explicit_timeouts_and_disables_sdk_retries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured, _, _ = _install_fake_chat(monkeypatch)

    model = OpenAIKnowledgeModel(
        model="gpt-test",
        cache_dir=tmp_path,
        cache_mode="use",
        max_output_tokens=8_000,
        read_timeout_seconds=180,
        connect_timeout_seconds=10,
        max_retries=3,
    )

    timeout = captured["timeout"]
    assert isinstance(timeout, httpx.Timeout)
    assert (timeout.connect, timeout.read, timeout.write, timeout.pool) == (10, 180, 60, 10)
    assert captured["max_retries"] == 0
    assert captured["max_tokens"] == 8_000
    assert "extra_body" not in captured
    assert model.max_retries == 3


def test_langchain_serializes_ollama_limit_as_top_level_max_tokens() -> None:
    payloads: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        return httpx.Response(
            200,
            request=request,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 1,
                "model": "qwen3.5-32k",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        model = RealChatOpenAI(
            model="qwen3.5-32k",
            base_url="http://ollama.test/v1",
            api_key="ollama",
            http_client=client,
            max_retries=0,
            extra_body={"max_tokens": 8_000},
        )
        model.invoke([HumanMessage(content="test")])

    assert payloads[0]["max_tokens"] == 8_000
    assert "max_completion_tokens" not in payloads[0]


@pytest.mark.parametrize("status_code", [408, 409, 429, 500, 503])
def test_retryable_http_statuses_are_retried(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    status_code: int,
) -> None:
    chunk = _chunk()
    request = httpx.Request("POST", "http://model.test")
    response = httpx.Response(status_code, request=request)
    error = APIStatusError("transient", response=response, body={})
    _, runnable, _ = _install_fake_chat(
        monkeypatch,
        map_responses=[error, _batch_result(chunk)],
    )
    sleeps: list[float] = []
    monkeypatch.setattr(llm_module, "_sleep", sleeps.append)

    result = _ollama(tmp_path, max_retries=1).analyze_chunks(
        [chunk],
        {},
        cache_context=_context(),
        progress_label="map batch 1/1",
    )

    assert result == _batch_result(chunk)
    assert runnable.calls == 2
    assert sleeps == [1]


def test_connection_retry_logs_attempt_reason_and_backoff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    chunk = _chunk()
    request = httpx.Request("POST", "http://model.test")
    _, runnable, _ = _install_fake_chat(
        monkeypatch,
        map_responses=[
            APIConnectionError(request=request),
            _batch_result(chunk),
        ],
    )
    sleeps: list[float] = []
    monkeypatch.setattr(llm_module, "_sleep", sleeps.append)
    caplog.set_level(logging.INFO)

    _ollama(tmp_path, max_retries=1).analyze_chunks(
        [chunk],
        {},
        cache_context=_context(),
        progress_label="map batch 1/1",
    )

    assert runnable.calls == 2
    assert sleeps == [1]
    assert "request attempt 1/2 failed" in caplog.text
    assert "APIConnectionError" in caplog.text
    assert "retrying in 1.0s" in caplog.text


@pytest.mark.parametrize(
    "failure",
    [
        ValueError("deterministic validation failed"),
        APIStatusError(
            "bad request",
            response=httpx.Response(
                400,
                request=httpx.Request("POST", "http://model.test"),
            ),
            body={},
        ),
    ],
)
def test_deterministic_and_configuration_failures_are_not_retried(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: Exception,
) -> None:
    _, runnable, _ = _install_fake_chat(
        monkeypatch,
        map_responses=[failure, _batch_result(_chunk())],
    )
    monkeypatch.setattr(
        llm_module,
        "_sleep",
        lambda _: pytest.fail("non-retryable failure slept"),
    )

    with pytest.raises(type(failure)):
        _ollama(tmp_path, max_retries=3).analyze_chunks(
            [_chunk()],
            {},
            cache_context=_context(),
        )

    assert runnable.calls == 1


def test_pydantic_and_chunk_identifier_validation_are_not_retried_or_cached(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunk = _chunk()
    invalid_shape = {"chunks": [{"chunk_id": chunk.chunk_id}]}
    wrong_id = _batch_result(replace(chunk, chunk_id="wrong"))
    _, shape_runnable, _ = _install_fake_chat(
        monkeypatch,
        map_responses=[invalid_shape],
    )
    model = _ollama(tmp_path, max_retries=3)

    with pytest.raises(ValidationError):
        model.analyze_chunks([chunk], {}, cache_context=_context())
    assert shape_runnable.calls == 1
    assert list(tmp_path.rglob("*.json")) == []

    _, id_runnable, _ = _install_fake_chat(
        monkeypatch,
        map_responses=[wrong_id],
    )
    model = _ollama(tmp_path, max_retries=3)
    with pytest.raises(ValueError, match="invalid chunk set"):
        model.analyze_chunks([chunk], {}, cache_context=_context())
    assert id_runnable.calls == 1
    assert list(tmp_path.rglob("*.json")) == []


def test_heartbeat_logs_without_waiting_sixty_seconds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    chunk = _chunk()
    release = threading.Event()

    def delayed_response(messages: list[object]) -> BatchChunkAnalysis:
        assert messages
        release.wait(0.04)
        return _batch_result(chunk)

    _install_fake_chat(monkeypatch, map_responses=[delayed_response])
    monkeypatch.setattr(llm_module, "HEARTBEAT_INTERVAL_SECONDS", 0.01)
    caplog.set_level(logging.INFO)

    _ollama(tmp_path).analyze_chunks(
        [chunk],
        {},
        cache_context=_context(),
        progress_label="map batch 1/1",
    )

    assert "map batch 1/1 request heartbeat" in caplog.text
    assert "read timeout 7200s" in caplog.text
    assert chunk.content not in caplog.text


def test_map_and_reduce_cache_are_reused_without_model_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    chunk = _chunk()
    map_result = _batch_result(chunk)
    reduce_result = _synthesis()
    _, first_map, first_reduce = _install_fake_chat(
        monkeypatch,
        map_responses=[map_result],
        reduce_responses=[reduce_result],
    )
    first = _ollama(tmp_path)
    context = _context()
    caplog.set_level(logging.INFO)

    observed_map = first.analyze_chunks([chunk], {}, cache_context=context)
    observed_reduce = first.synthesize_project(
        {"evidence": "same"},
        cache_context=context,
        map_results=observed_map.chunks,
    )

    assert observed_reduce == reduce_result
    assert first_map.calls == first_reduce.calls == 1
    assert len(list((tmp_path / "map").glob("*.json"))) == 1
    assert len(list((tmp_path / "reduce").glob("*.json"))) == 1

    _, second_map, second_reduce = _install_fake_chat(monkeypatch)
    second = _ollama(tmp_path)
    cached_map = second.analyze_chunks([chunk], {}, cache_context=context)
    cached_reduce = second.synthesize_project(
        {"evidence": "same"},
        cache_context=context,
        map_results=cached_map.chunks,
    )

    assert cached_map == map_result
    assert cached_reduce == reduce_result
    assert second_map.calls == second_reduce.calls == 0
    assert "map batch cache hit" in caplog.text
    assert "reduce cache hit" in caplog.text


def test_later_map_failure_preserves_and_reuses_earlier_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunk_a = _chunk("A.java")
    chunk_b = _chunk("B.java")
    request = httpx.Request("POST", "http://model.test")
    _, first_map, _ = _install_fake_chat(
        monkeypatch,
        map_responses=[
            _batch_result(chunk_a),
            APIConnectionError(request=request),
        ],
    )
    context = _context()
    first = _ollama(tmp_path)

    first.analyze_chunks([chunk_a], {}, cache_context=context, progress_label="map batch 1/2")
    with pytest.raises(APIConnectionError):
        first.analyze_chunks(
            [chunk_b],
            {},
            cache_context=context,
            progress_label="map batch 2/2",
        )

    assert first_map.calls == 2
    assert len(list((tmp_path / "map").glob("*.json"))) == 1

    _, resumed_map, resumed_reduce = _install_fake_chat(
        monkeypatch,
        map_responses=[_batch_result(chunk_b)],
        reduce_responses=[_synthesis()],
    )
    resumed = _ollama(tmp_path)
    map_a = resumed.analyze_chunks([chunk_a], {}, cache_context=context)
    map_b = resumed.analyze_chunks([chunk_b], {}, cache_context=context)
    resumed.synthesize_project(
        {"evidence": "complete"},
        cache_context=context,
        map_results=[*map_a.chunks, *map_b.chunks],
    )

    assert resumed_map.calls == 1
    assert resumed_reduce.calls == 1
    assert len(list((tmp_path / "map").glob("*.json"))) == 2


def test_reduce_failure_reuses_all_map_batches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunk = _chunk()
    request = httpx.Request("POST", "http://model.test")
    _, first_map, first_reduce = _install_fake_chat(
        monkeypatch,
        map_responses=[_batch_result(chunk)],
        reduce_responses=[APIConnectionError(request=request)],
    )
    context = _context()
    first = _ollama(tmp_path)
    map_result = first.analyze_chunks([chunk], {}, cache_context=context)
    with pytest.raises(APIConnectionError):
        first.synthesize_project(
            {"evidence": "same"},
            cache_context=context,
            map_results=map_result.chunks,
        )

    assert first_map.calls == first_reduce.calls == 1
    assert len(list((tmp_path / "map").glob("*.json"))) == 1
    assert len(list((tmp_path / "reduce").glob("*.json"))) == 0

    _, resumed_map, resumed_reduce = _install_fake_chat(
        monkeypatch,
        reduce_responses=[_synthesis()],
    )
    resumed = _ollama(tmp_path)
    cached_map = resumed.analyze_chunks([chunk], {}, cache_context=context)
    resumed.synthesize_project(
        {"evidence": "same"},
        cache_context=context,
        map_results=cached_map.chunks,
    )
    assert resumed_map.calls == 0
    assert resumed_reduce.calls == 1


@pytest.mark.parametrize(
    "changed_context",
    [
        _context(source_fingerprint="changed"),
        _context(resolved_commit="changed"),
        _context(repository_identity="https://other.test/repository.git"),
        _context(requested_ref="release"),
    ],
)
def test_repository_context_changes_invalidate_map_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    changed_context: AnalysisCacheContext,
) -> None:
    chunk = _chunk()
    _install_fake_chat(monkeypatch, map_responses=[_batch_result(chunk)])
    _ollama(tmp_path).analyze_chunks([chunk], {}, cache_context=_context())

    _, changed_map, _ = _install_fake_chat(
        monkeypatch,
        map_responses=[_batch_result(chunk)],
    )
    _ollama(tmp_path).analyze_chunks([chunk], {}, cache_context=changed_context)

    assert changed_map.calls == 1


@pytest.mark.parametrize(
    ("model", "base_url", "max_output_tokens"),
    [
        ("other-model", "http://localhost:11434", 8_000),
        ("qwen3.5-32k", "http://other-ollama:11434", 8_000),
        ("qwen3.5-32k", "http://localhost:11434", 4_000),
    ],
)
def test_model_endpoint_and_generation_settings_invalidate_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    model: str,
    base_url: str,
    max_output_tokens: int,
) -> None:
    chunk = _chunk()
    _install_fake_chat(monkeypatch, map_responses=[_batch_result(chunk)])
    _ollama(tmp_path).analyze_chunks([chunk], {}, cache_context=_context())

    _, changed_map, _ = _install_fake_chat(
        monkeypatch,
        map_responses=[_batch_result(chunk)],
    )
    _ollama(
        tmp_path,
        model=model,
        base_url=base_url,
        max_output_tokens=max_output_tokens,
    ).analyze_chunks([chunk], {}, cache_context=_context())

    assert changed_map.calls == 1


def test_exact_batch_contents_and_ordered_reduce_results_invalidate_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunk_a = _chunk("A.java")
    chunk_b = _chunk("B.java")
    context = _context()
    map_a = _batch_result(chunk_a)
    map_b = _batch_result(chunk_b)
    _install_fake_chat(
        monkeypatch,
        map_responses=[map_a],
        reduce_responses=[_synthesis("initial")],
    )
    initial = _ollama(tmp_path)
    initial.analyze_chunks([chunk_a], {}, cache_context=context)
    initial.synthesize_project(
        {"evidence": "same"},
        cache_context=context,
        map_results=[*map_a.chunks, *map_b.chunks],
    )

    changed_chunk = replace(chunk_a, content="class A { int changed; }", token_count=8)
    _, changed_map, changed_reduce = _install_fake_chat(
        monkeypatch,
        map_responses=[_batch_result(changed_chunk)],
        reduce_responses=[_synthesis("changed")],
    )
    changed = _ollama(tmp_path)
    changed.analyze_chunks([changed_chunk], {}, cache_context=context)
    changed.synthesize_project(
        {"evidence": "same"},
        cache_context=context,
        map_results=[*map_b.chunks, *map_a.chunks],
    )

    assert changed_map.calls == 1
    assert changed_reduce.calls == 1


def test_provider_change_invalidates_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunk = _chunk()
    _install_fake_chat(monkeypatch, map_responses=[_batch_result(chunk)])
    _ollama(tmp_path, model="shared-name").analyze_chunks(
        [chunk],
        {},
        cache_context=_context(),
    )

    _, openai_map, _ = _install_fake_chat(
        monkeypatch,
        map_responses=[_batch_result(chunk)],
    )
    OpenAIKnowledgeModel(
        model="shared-name",
        cache_dir=tmp_path,
        cache_mode="use",
        max_output_tokens=8_000,
        read_timeout_seconds=180,
        connect_timeout_seconds=10,
        max_retries=3,
    ).analyze_chunks([chunk], {}, cache_context=_context())

    assert openai_map.calls == 1


def test_prompt_and_schema_changes_invalidate_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunk = _chunk()
    _install_fake_chat(monkeypatch, map_responses=[_batch_result(chunk)])
    _ollama(tmp_path).analyze_chunks([chunk], {}, cache_context=_context())

    _, prompt_map, _ = _install_fake_chat(
        monkeypatch,
        map_responses=[_batch_result(chunk)],
    )
    monkeypatch.setattr(
        llm_module,
        "MAP_SYSTEM_PROMPT",
        f"{llm_module.MAP_SYSTEM_PROMPT}\nChanged prompt.",
    )
    _ollama(tmp_path).analyze_chunks([chunk], {}, cache_context=_context())
    assert prompt_map.calls == 1

    _, schema_map, _ = _install_fake_chat(
        monkeypatch,
        map_responses=[_batch_result(chunk)],
    )
    original_schema = BatchChunkAnalysis.model_json_schema
    monkeypatch.setattr(
        BatchChunkAnalysis,
        "model_json_schema",
        classmethod(
            lambda cls: {
                **original_schema(),
                "$comment": "changed schema",
            }
        ),
    )
    _ollama(tmp_path).analyze_chunks([chunk], {}, cache_context=_context())
    assert schema_map.calls == 1


def test_cached_map_identifier_validation_treats_entry_as_miss(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    chunk = _chunk()
    _install_fake_chat(monkeypatch, map_responses=[_batch_result(chunk)])
    _ollama(tmp_path).analyze_chunks([chunk], {}, cache_context=_context())

    cache_file = next((tmp_path / "map").glob("*.json"))
    envelope = json.loads(cache_file.read_text(encoding="utf-8"))
    envelope["result"]["chunks"][0]["chunk_id"] = "wrong::part-1"
    cache_file.write_text(json.dumps(envelope), encoding="utf-8")

    _, regenerated, _ = _install_fake_chat(
        monkeypatch,
        map_responses=[_batch_result(chunk)],
    )
    caplog.set_level(logging.INFO)
    result = _ollama(tmp_path).analyze_chunks([chunk], {}, cache_context=_context())

    assert result == _batch_result(chunk)
    assert regenerated.calls == 1
    assert "failed chunk-ID validation" in caplog.text


def test_refresh_and_off_modes_control_model_cache_behavior(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunk = _chunk()
    _install_fake_chat(
        monkeypatch,
        map_responses=[_batch_result(chunk, purpose="initial")],
    )
    _ollama(tmp_path).analyze_chunks([chunk], {}, cache_context=_context())

    _, refreshed, _ = _install_fake_chat(
        monkeypatch,
        map_responses=[_batch_result(chunk, purpose="refreshed")],
    )
    refreshed_result = _ollama(tmp_path, cache_mode="refresh").analyze_chunks(
        [chunk],
        {},
        cache_context=_context(),
    )
    assert refreshed.calls == 1
    assert refreshed_result.chunks[0].purpose == "refreshed"

    _, off_calls, _ = _install_fake_chat(
        monkeypatch,
        map_responses=[
            _batch_result(chunk, purpose="off-one"),
            _batch_result(chunk, purpose="off-two"),
        ],
    )
    off_model = _ollama(tmp_path / "off", cache_mode="off")
    off_model.analyze_chunks([chunk], {}, cache_context=_context())
    off_model.analyze_chunks([chunk], {}, cache_context=_context())
    assert off_calls.calls == 2
    assert not (tmp_path / "off").exists()
