from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Protocol, TypeVar

import httpx
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from openai import APIConnectionError, APIStatusError

from codebase_analyzer import __version__
from codebase_analyzer.cache import AnalysisCacheContext, CacheMode, JsonCache
from codebase_analyzer.models import (
    ArchitectureLayer,
    BatchChunkAnalysis,
    ChunkAnalysis,
    MethodDescription,
    MethodFact,
    ProjectSynthesis,
)
from codebase_analyzer.prompts import (
    MAP_SYSTEM_PROMPT,
    MAP_USER_TEMPLATE,
    SYNTHESIS_SYSTEM_PROMPT,
    SYNTHESIS_USER_TEMPLATE,
)
from codebase_analyzer.token_budget import CodeChunk

PROMPT_VERSION = "2026-07-22.3"
STRUCTURED_OUTPUT_METHOD = "json_schema"
HEARTBEAT_INTERVAL_SECONDS = 60.0
MAX_RETRY_BACKOFF_SECONDS = 60.0

SchemaT = TypeVar("SchemaT", BatchChunkAnalysis, ProjectSynthesis)
logger = logging.getLogger(__name__)
_sleep = time.sleep


class KnowledgeModel(Protocol):
    model_name: str
    mode: str
    provider_name: str

    def analyze_chunks(
        self,
        chunks: list[CodeChunk],
        methods_by_path: dict[str, list[MethodFact]],
        *,
        cache_context: AnalysisCacheContext | None = None,
        progress_label: str = "map batch",
    ) -> BatchChunkAnalysis: ...

    def synthesize_project(
        self,
        evidence: dict[str, object],
        *,
        cache_context: AnalysisCacheContext | None = None,
        progress_label: str = "reduce",
        map_results: list[ChunkAnalysis] | None = None,
    ) -> ProjectSynthesis: ...


class StructuredKnowledgeModel:
    """Shared map/reduce behavior for schema-constrained chat model adapters."""

    mode = "llm"

    def _configure(
        self,
        *,
        model_name: str,
        endpoint_identity: str,
        cache_dir: Path,
        cache_mode: CacheMode,
        base_model: object,
        include_schema_in_prompt: bool,
        material_generation_settings: dict[str, object],
        max_retries: int,
        read_timeout_seconds: float,
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must be nonnegative")
        if read_timeout_seconds <= 0:
            raise ValueError("read_timeout_seconds must be positive")
        self.model_name = model_name
        self.endpoint_identity = endpoint_identity
        self.cache = JsonCache(cache_dir, cache_mode)
        self.include_schema_in_prompt = include_schema_in_prompt
        self.material_generation_settings = material_generation_settings
        self.max_retries = max_retries
        self.read_timeout_seconds = read_timeout_seconds
        self.chunk_model = base_model.with_structured_output(
            BatchChunkAnalysis,
            method=STRUCTURED_OUTPUT_METHOD,
        )
        self.synthesis_model = base_model.with_structured_output(
            ProjectSynthesis,
            method=STRUCTURED_OUTPUT_METHOD,
        )

    def analyze_chunks(
        self,
        chunks: list[CodeChunk],
        methods_by_path: dict[str, list[MethodFact]],
        *,
        cache_context: AnalysisCacheContext | None = None,
        progress_label: str = "map batch",
    ) -> BatchChunkAnalysis:
        repository_data = _render_chunks(chunks, methods_by_path)
        user_prompt = MAP_USER_TEMPLATE.format(repository_data=repository_data)
        messages = self._messages(MAP_SYSTEM_PROMPT, user_prompt, BatchChunkAnalysis)
        cache_key = self.cache.key(
            "map",
            self._cache_key_data(
                stage="map",
                cache_context=cache_context,
                schema=BatchChunkAnalysis,
                system_prompt=MAP_SYSTEM_PROMPT,
                user_template=MAP_USER_TEMPLATE,
                messages=messages,
                stage_input={
                    "ordered_chunks_and_method_facts": repository_data,
                },
            ),
        )
        cached = self._load_cache("map", cache_key, BatchChunkAnalysis, progress_label)
        if cached is not None:
            try:
                _validate_chunk_identifiers(chunks, cached)
            except ValueError as exc:
                logger.warning(
                    "%s cache entry failed chunk-ID validation and will be regenerated: %s",
                    progress_label,
                    exc,
                )
            else:
                return cached

        result = self._invoke_with_retry(self.chunk_model, messages, progress_label)
        validated = BatchChunkAnalysis.model_validate(result)
        _validate_chunk_identifiers(chunks, validated)
        self._save_cache("map", cache_key, validated, progress_label)
        return validated

    def synthesize_project(
        self,
        evidence: dict[str, object],
        *,
        cache_context: AnalysisCacheContext | None = None,
        progress_label: str = "reduce",
        map_results: list[ChunkAnalysis] | None = None,
    ) -> ProjectSynthesis:
        serialized = json.dumps(evidence, indent=2, sort_keys=True)
        user_prompt = SYNTHESIS_USER_TEMPLATE.format(analysis_evidence=serialized)
        messages = self._messages(
            SYNTHESIS_SYSTEM_PROMPT,
            user_prompt,
            ProjectSynthesis,
        )
        cache_key = self.cache.key(
            "reduce",
            self._cache_key_data(
                stage="reduce",
                cache_context=cache_context,
                schema=ProjectSynthesis,
                system_prompt=SYNTHESIS_SYSTEM_PROMPT,
                user_template=SYNTHESIS_USER_TEMPLATE,
                messages=messages,
                stage_input={
                    "evidence": evidence,
                    "complete_ordered_map_results": [
                        item.model_dump(mode="json") for item in (map_results or [])
                    ],
                },
            ),
        )
        cached = self._load_cache("reduce", cache_key, ProjectSynthesis, progress_label)
        if cached is not None:
            return cached

        result = self._invoke_with_retry(self.synthesis_model, messages, progress_label)
        validated = ProjectSynthesis.model_validate(result)
        self._save_cache("reduce", cache_key, validated, progress_label)
        return validated

    def _cache_key_data(
        self,
        *,
        stage: str,
        cache_context: AnalysisCacheContext | None,
        schema: type[SchemaT],
        system_prompt: str,
        user_template: str,
        messages: list[SystemMessage | HumanMessage],
        stage_input: dict[str, object],
    ) -> dict[str, object]:
        return {
            "analyzer_version": __version__,
            "analysis_context": cache_context.as_key_data() if cache_context else None,
            "stage": stage,
            "provider": self.provider_name,
            "model": self.model_name,
            "endpoint": self.endpoint_identity,
            "material_generation_settings": self.material_generation_settings,
            "prompt_version": PROMPT_VERSION,
            "system_prompt": system_prompt,
            "user_template": user_template,
            "complete_rendered_messages": [
                {"type": message.type, "content": message.content} for message in messages
            ],
            "structured_output_method": STRUCTURED_OUTPUT_METHOD,
            "structured_output_schema": schema.model_json_schema(),
            "stage_input": stage_input,
        }

    def _load_cache(
        self,
        namespace: str,
        key: str,
        schema: type[SchemaT],
        progress_label: str,
    ) -> SchemaT | None:
        if self.cache.mode == "off":
            logger.info("%s cache off", progress_label)
            return None
        if self.cache.mode == "refresh":
            logger.info("%s cache refresh; skipping existing entry", progress_label)
            return None

        lookup = self.cache.load(namespace, key, schema)
        if lookup.status == "hit":
            logger.info("%s cache hit: %s", progress_label, lookup.path)
            return lookup.value
        if lookup.status == "corrupt":
            logger.info("%s cache corrupt; treating as miss: %s", progress_label, lookup.path)
        else:
            logger.info("%s cache miss: %s", progress_label, lookup.path)
        return None

    def _save_cache(
        self,
        namespace: str,
        key: str,
        value: SchemaT,
        progress_label: str,
    ) -> None:
        path = self.cache.save(namespace, key, value)
        if path is not None:
            logger.info("%s cache saved: %s", progress_label, path)

    def _invoke_with_retry(
        self,
        runnable: object,
        messages: list[SystemMessage | HumanMessage],
        progress_label: str,
    ) -> object:
        total_attempts = self.max_retries + 1
        for attempt in range(1, total_attempts + 1):
            retry_backoff: float | None = None
            logger.info(
                "%s request attempt %d/%d started (read timeout %.0fs)",
                progress_label,
                attempt,
                total_attempts,
                self.read_timeout_seconds,
            )
            attempt_started = time.monotonic()
            stop_heartbeat = threading.Event()
            heartbeat = threading.Thread(
                target=self._heartbeat,
                args=(stop_heartbeat, progress_label, attempt, attempt_started),
                name="codebase-analyzer-heartbeat",
                daemon=True,
            )
            heartbeat.start()
            try:
                return runnable.invoke(messages)
            except Exception as exc:
                elapsed = time.monotonic() - attempt_started
                if attempt >= total_attempts or not _is_retryable(exc):
                    logger.error(
                        "%s request attempt %d/%d failed after %.1fs: %s",
                        progress_label,
                        attempt,
                        total_attempts,
                        elapsed,
                        _failure_reason(exc),
                    )
                    raise
                backoff = min(2 ** (attempt - 1), MAX_RETRY_BACKOFF_SECONDS)
                logger.warning(
                    "%s request attempt %d/%d failed after %.1fs: %s; retrying in %.1fs",
                    progress_label,
                    attempt,
                    total_attempts,
                    elapsed,
                    _failure_reason(exc),
                    backoff,
                )
                retry_backoff = backoff
            finally:
                stop_heartbeat.set()
                heartbeat.join(timeout=1.0)
            if retry_backoff is not None:
                _sleep(retry_backoff)
        raise AssertionError("retry loop exited unexpectedly")

    def _heartbeat(
        self,
        stop: threading.Event,
        progress_label: str,
        attempt: int,
        attempt_started: float,
    ) -> None:
        while not stop.wait(HEARTBEAT_INTERVAL_SECONDS):
            logger.info(
                "%s request heartbeat: attempt %d, elapsed %.1fs, read timeout %.0fs",
                progress_label,
                attempt,
                time.monotonic() - attempt_started,
                self.read_timeout_seconds,
            )

    def _messages(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: type[SchemaT],
    ) -> list[SystemMessage | HumanMessage]:
        if self.include_schema_in_prompt:
            user_prompt = "\n\n".join(
                [
                    user_prompt,
                    "Return JSON matching this schema exactly:",
                    json.dumps(schema.model_json_schema(), sort_keys=True),
                ]
            )
        return [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]


class OpenAIKnowledgeModel(StructuredKnowledgeModel):
    provider_name = "openai"

    def __init__(
        self,
        *,
        model: str,
        cache_dir: Path,
        cache_mode: CacheMode = "use",
        max_output_tokens: int = 8_000,
        read_timeout_seconds: float = 180,
        connect_timeout_seconds: float = 10,
        max_retries: int = 3,
    ) -> None:
        timeout = _httpx_timeout(connect_timeout_seconds, read_timeout_seconds)
        base_model = ChatOpenAI(
            model=model,
            max_retries=0,
            timeout=timeout,
            max_tokens=max_output_tokens,
        )
        endpoint = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self._configure(
            model_name=model,
            endpoint_identity=endpoint,
            cache_dir=cache_dir,
            cache_mode=cache_mode,
            base_model=base_model,
            include_schema_in_prompt=False,
            material_generation_settings={
                "max_completion_tokens": max_output_tokens,
            },
            max_retries=max_retries,
            read_timeout_seconds=read_timeout_seconds,
        )


class OllamaKnowledgeModel(StructuredKnowledgeModel):
    """Ollama adapter using its OpenAI-compatible chat-completions API."""

    provider_name = "ollama"

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: str,
        cache_dir: Path,
        cache_mode: CacheMode = "use",
        max_output_tokens: int = 8_000,
        read_timeout_seconds: float = 7_200,
        connect_timeout_seconds: float = 10,
        max_retries: int = 0,
    ) -> None:
        normalized_base_url = normalize_ollama_base_url(base_url)
        timeout = _httpx_timeout(connect_timeout_seconds, read_timeout_seconds)
        base_model = ChatOpenAI(
            model=model,
            base_url=normalized_base_url,
            api_key=api_key,
            max_retries=0,
            timeout=timeout,
            extra_body={"max_tokens": max_output_tokens},
            temperature=0,
            reasoning_effort="none",
        )
        self._configure(
            model_name=model,
            endpoint_identity=normalized_base_url,
            cache_dir=cache_dir,
            cache_mode=cache_mode,
            base_model=base_model,
            include_schema_in_prompt=True,
            material_generation_settings={
                "max_tokens": max_output_tokens,
                "temperature": 0,
                "reasoning_effort": "none",
                "schema_in_prompt": True,
            },
            max_retries=max_retries,
            read_timeout_seconds=read_timeout_seconds,
        )


def _httpx_timeout(connect_seconds: float, read_seconds: float) -> httpx.Timeout:
    if connect_seconds <= 0:
        raise ValueError("connect_timeout_seconds must be positive")
    if read_seconds <= 0:
        raise ValueError("read_timeout_seconds must be positive")
    return httpx.Timeout(
        connect=connect_seconds,
        read=read_seconds,
        write=60.0,
        pool=connect_seconds,
    )


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, APIStatusError):
        return exc.status_code in {408, 409, 429} or exc.status_code >= 500
    return isinstance(
        exc,
        (
            APIConnectionError,
            httpx.TimeoutException,
            httpx.NetworkError,
        ),
    )


def _failure_reason(exc: Exception) -> str:
    detail = re.sub(r"\s+", " ", str(exc)).strip()
    return f"{type(exc).__name__}: {detail or 'no detail'}"


def normalize_ollama_base_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if not normalized:
        raise ValueError("Ollama base URL must not be empty")
    return normalized if normalized.endswith("/v1") else f"{normalized}/v1"


class OfflineKnowledgeModel:
    """Credential-free deterministic adapter for tests and pipeline inspection only."""

    mode = "offline"
    provider_name = "offline"
    model_name = "deterministic-offline"

    def analyze_chunks(
        self,
        chunks: list[CodeChunk],
        methods_by_path: dict[str, list[MethodFact]],
        *,
        cache_context: AnalysisCacheContext | None = None,
        progress_label: str = "map batch",
    ) -> BatchChunkAnalysis:
        del cache_context, progress_label
        analyses: list[ChunkAnalysis] = []
        for chunk in chunks:
            methods = methods_by_path.get(chunk.file_path, [])
            analyses.append(
                ChunkAnalysis(
                    chunk_id=chunk.chunk_id,
                    purpose=(
                        f"Contains {_humanize_path(chunk.file_path)} "
                        "implementation or configuration."
                    ),
                    responsibilities=[f"Defines behavior represented by {chunk.file_path}."],
                    noteworthy_aspects=[
                        "Offline description generated without semantic LLM interpretation."
                    ],
                    method_descriptions=[
                        MethodDescription(
                            method_id=method.method_id,
                            description=f"Implements {_split_identifier(method.name)} behavior.",
                        )
                        for method in methods
                    ],
                )
            )
        return BatchChunkAnalysis(chunks=analyses)

    def synthesize_project(
        self,
        evidence: dict[str, object],
        *,
        cache_context: AnalysisCacheContext | None = None,
        progress_label: str = "reduce",
        map_results: list[ChunkAnalysis] | None = None,
    ) -> ProjectSynthesis:
        del cache_context, progress_label, map_results
        technologies = [str(item) for item in evidence.get("technologies", [])]
        modules = [str(item) for item in evidence.get("modules", [])]
        return ProjectSynthesis(
            purpose="Provides a REST API over the MySQL Sakila sample database.",
            overview=(
                "A Spring Boot application exposing DVD-rental domain data and operations through "
                "layered HTTP, service, and persistence components."
            ),
            primary_capabilities=[
                "CRUD and query operations for Sakila domain resources",
                "JWT-based authentication and authorization",
                "Hypermedia-oriented REST responses and generated API documentation",
                "Health and Prometheus observability endpoints",
            ],
            architecture_style="Layered Spring Boot monolith with domain-oriented service modules",
            architecture_layers=[
                ArchitectureLayer(
                    name="Web/API",
                    responsibility="Accepts REST requests and assembles HTTP representations.",
                    representative_paths=["src/main/java/com/example/app/services"],
                ),
                ArchitectureLayer(
                    name="Service",
                    responsibility="Coordinates domain operations and transactions.",
                    representative_paths=["src/main/java/com/example/app/services"],
                ),
                ArchitectureLayer(
                    name="Persistence",
                    responsibility="Accesses MySQL through JPA and Querydsl repositories.",
                    representative_paths=["src/main/java/com/example/app/services"],
                ),
            ],
            typical_request_flow=[
                "A controller receives and validates an HTTP request.",
                "A service applies application logic and calls a repository.",
                "JPA or Querydsl reads or updates the Sakila database.",
                "A mapper and assembler create a DTO or HATEOAS response.",
            ],
            noteworthy_aspects=[
                f"Detected technologies: {', '.join(technologies)}.",
                f"Detected modules: {', '.join(modules)}.",
            ],
            assumptions_and_limitations=[
                "Offline mode validates the pipeline but is not a substitute for the required "
                "LLM run.",
                "Runtime behavior and database-dependent paths were not executed by static "
                "analysis.",
            ],
        )


def _render_chunks(
    chunks: list[CodeChunk],
    methods_by_path: dict[str, list[MethodFact]],
) -> str:
    sections: list[str] = []
    for chunk in chunks:
        method_facts = [
            {
                "method_id": method.method_id,
                "signature": method.signature,
                "lines": [method.start_line, method.end_line],
                "cyclomatic_complexity": method.cyclomatic_complexity,
            }
            for method in methods_by_path.get(chunk.file_path, [])
        ]
        sections.append(
            "\n".join(
                [
                    f'<CHUNK id="{chunk.chunk_id}" path="{chunk.file_path}">',
                    "<DETERMINISTIC_METHOD_FACTS>",
                    json.dumps(method_facts, indent=2),
                    "</DETERMINISTIC_METHOD_FACTS>",
                    "<SOURCE>",
                    chunk.content,
                    "</SOURCE>",
                    "</CHUNK>",
                ]
            )
        )
    return "\n\n".join(sections)


def _validate_chunk_identifiers(
    expected_chunks: list[CodeChunk],
    result: BatchChunkAnalysis,
) -> None:
    expected = {chunk.chunk_id for chunk in expected_chunks}
    actual = {chunk.chunk_id for chunk in result.chunks}
    if expected != actual or len(actual) != len(result.chunks):
        raise ValueError(
            "LLM returned an invalid chunk set; "
            f"missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}"
        )


def _humanize_path(path: str) -> str:
    stem = Path(path).stem
    return _split_identifier(stem)


def _split_identifier(value: str) -> str:
    spaced = re.sub(r"(?<!^)(?=[A-Z])", " ", value).replace("_", " ").replace("-", " ")
    return re.sub(r"\s+", " ", spaced).strip().lower()
