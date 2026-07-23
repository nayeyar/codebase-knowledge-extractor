from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

from codebase_analyzer import __version__
from codebase_analyzer.cache import AnalysisCacheContext, canonical_json
from codebase_analyzer.config import AnalyzerConfig
from codebase_analyzer.discovery import DiscoveryResult, SourceFile, discover_source_files
from codebase_analyzer.java_parser import parse_java_file
from codebase_analyzer.llm import PROMPT_VERSION, KnowledgeModel
from codebase_analyzer.models import (
    AnalysisReport,
    ComplexityHotspot,
    ComplexitySummary,
    GenerationMetadata,
    JavaFileAnalysis,
    KeyMethod,
    MethodFact,
    RepositorySource,
    ScopeStatistics,
    TokenBudgetSummary,
)
from codebase_analyzer.token_budget import TokenBudgeter

logger = logging.getLogger(__name__)


class CodebaseAnalysisPipeline:
    def __init__(self, config: AnalyzerConfig, model: KnowledgeModel) -> None:
        self.config = config
        self.model = model
        self.budgeter = TokenBudgeter(config.model)

    def run(
        self,
        root: Path,
        *,
        repository_url: str | None,
        requested_ref: str | None,
        resolved_commit: str | None,
    ) -> AnalysisReport:
        pipeline_started = time.monotonic()
        logger.info("Discovery started for %s", root)
        discovery_started = time.monotonic()
        discovery = discover_source_files(
            root,
            max_file_bytes=self.config.max_file_bytes,
            include_tests=self.config.include_tests,
        )
        source_fingerprint = _source_fingerprint(discovery.files)
        logger.info(
            "Discovery completed in %.1fs: %d discovered, %d processed, %d excluded, %d oversized",
            time.monotonic() - discovery_started,
            discovery.discovered_count,
            len(discovery.files),
            discovery.excluded_count,
            discovery.skipped_large_count,
        )

        cache_context = AnalysisCacheContext(
            repository_identity=repository_url or str(root.resolve()),
            requested_ref=requested_ref,
            resolved_commit=resolved_commit,
            source_fingerprint=source_fingerprint,
            analyzer_version=__version__,
        )

        parsing_started = time.monotonic()
        logger.info("Static Java parsing started")
        java_analyses = self._parse_java(discovery.files)
        all_methods = [method for analysis in java_analyses for method in analysis.methods]
        logger.info(
            "Static Java parsing completed in %.1fs: %d Java files, %d methods",
            time.monotonic() - parsing_started,
            len(java_analyses),
            len(all_methods),
        )
        key_methods = self._select_key_methods(all_methods)
        methods_by_path = self._methods_by_path(key_methods)

        selected_files = self._select_llm_files(discovery.files, java_analyses)
        chunks = []
        for file in selected_files:
            chunks.extend(
                self.budgeter.chunk_file(
                    file.relative_path,
                    file.content,
                    self.config.max_chunk_tokens,
                )
            )
        usable_batch_tokens = self.config.max_batch_tokens - self.config.prompt_reserve_tokens
        extra_tokens = {
            chunk.chunk_id: self._chunk_metadata_tokens(
                chunk.file_path, methods_by_path.get(chunk.file_path, [])
            )
            for chunk in chunks
        }
        batches = self.budgeter.pack_batches(
            chunks,
            usable_batch_tokens,
            extra_tokens_by_chunk=extra_tokens,
            max_chunks_per_batch=self.config.max_chunks_per_batch,
        )
        batch_estimates = [
            self.config.prompt_reserve_tokens
            + sum(chunk.token_count + extra_tokens[chunk.chunk_id] for chunk in batch)
            for batch in batches
        ]
        logger.info(
            "Semantic selection prepared: %d files, %d chunks, %d batches, "
            "%d selected source tokens",
            len(selected_files),
            len(chunks),
            len(batches),
            sum(chunk.token_count for chunk in chunks),
        )

        chunk_analyses = []
        for batch_index, batch in enumerate(batches, start=1):
            progress_label = f"map batch {batch_index}/{len(batches)}"
            batch_started = time.monotonic()
            logger.info(
                "%s started: %d unique files, %d chunks, %d source tokens, "
                "%d estimated prompt tokens",
                progress_label,
                len({chunk.file_path for chunk in batch}),
                len(batch),
                sum(chunk.token_count for chunk in batch),
                batch_estimates[batch_index - 1],
            )
            try:
                result = self.model.analyze_chunks(
                    batch,
                    methods_by_path,
                    cache_context=cache_context,
                    progress_label=progress_label,
                )
            except Exception as exc:
                logger.error(
                    "%s failed after %.1fs: %s",
                    progress_label,
                    time.monotonic() - batch_started,
                    exc,
                )
                raise
            chunk_analyses.extend(result.chunks)
            logger.info(
                "%s completed in %.1fs",
                progress_label,
                time.monotonic() - batch_started,
            )

        technologies = detect_technologies(discovery.files)
        modules = detect_modules(discovery.files)
        complexity = build_complexity_summary(all_methods)
        descriptions = {
            item.method_id: item.description
            for chunk in chunk_analyses
            for item in chunk.method_descriptions
            if item.method_id in {method.method_id for method in key_methods}
        }
        base_project_evidence = {
            "repository": repository_url or root.name,
            "technologies": technologies,
            "modules": modules,
            "statistics": {
                "processed_files": len(discovery.files),
                "java_files": len(java_analyses),
                "java_methods": len(all_methods),
                "source_lines": sum(file.line_count for file in discovery.files),
            },
            "complexity": complexity.model_dump(),
        }
        project_evidence = self._bounded_project_evidence(
            base_project_evidence,
            chunk_analyses,
            usable_batch_tokens,
        )
        reduce_evidence_tokens = self.budgeter.count(json.dumps(project_evidence))
        reduce_started = time.monotonic()
        logger.info(
            "reduce started: %d evidence tokens, %d ordered map results",
            reduce_evidence_tokens,
            len(chunk_analyses),
        )
        try:
            project = self.model.synthesize_project(
                project_evidence,
                cache_context=cache_context,
                progress_label="reduce",
                map_results=chunk_analyses,
            )
        except Exception as exc:
            logger.error(
                "reduce failed after %.1fs: %s",
                time.monotonic() - reduce_started,
                exc,
            )
            raise
        logger.info("reduce completed in %.1fs", time.monotonic() - reduce_started)

        report = AnalysisReport(
            schema_version="1.0.0",
            source=RepositorySource(
                repository_url=repository_url,
                requested_ref=requested_ref,
                resolved_commit=resolved_commit,
                analyzed_path=root.name,
            ),
            generation=GenerationMetadata(
                generated_at=datetime.now(UTC),
                generator_version=__version__,
                mode=self.model.mode,
                provider=self.model.provider_name,
                model=self.model.model_name,
                token_strategy=(
                    "Tree-sitter semantic extraction, token-counted file chunks, bounded map "
                    "batches, "
                    "and project-level reduce synthesis"
                ),
                prompt_version=PROMPT_VERSION,
            ),
            scope=_scope_statistics(discovery, java_analyses, all_methods, selected_files),
            token_budget=TokenBudgetSummary(
                encoding=self.budgeter.encoding.name,
                max_chunk_tokens=self.config.max_chunk_tokens,
                max_batch_input_tokens=self.config.max_batch_tokens,
                prompt_reserve_tokens=self.config.prompt_reserve_tokens,
                max_output_tokens=self.config.max_output_tokens,
                max_chunks_per_batch=self.config.max_chunks_per_batch,
                selected_source_tokens=sum(chunk.token_count for chunk in chunks),
                map_chunk_count=len(chunks),
                map_batch_count=len(batches),
                largest_estimated_map_input_tokens=max(batch_estimates, default=0),
                reduce_evidence_tokens=reduce_evidence_tokens,
            ),
            project=project,
            technologies=technologies,
            modules=modules,
            complexity=complexity,
            key_methods=[
                KeyMethod(
                    method_id=method.method_id,
                    file_path=method.file_path,
                    class_name=method.class_name,
                    signature=method.signature,
                    description=descriptions.get(
                        method.method_id,
                        f"Implements {humanize_identifier(method.name)} behavior.",
                    ),
                    start_line=method.start_line,
                    end_line=method.end_line,
                    cyclomatic_complexity=method.cyclomatic_complexity,
                    complexity_rating=method.complexity_rating,
                    annotations=method.annotations,
                )
                for method in key_methods
            ],
            analyzed_file_summaries=chunk_analyses,
        )
        logger.info(
            "Analysis pipeline completed in %.1fs",
            time.monotonic() - pipeline_started,
        )
        return report

    def _parse_java(self, files: list[SourceFile]) -> list[JavaFileAnalysis]:
        return [
            parse_java_file(file.relative_path, file.content)
            for file in files
            if file.path.suffix.lower() == ".java"
        ]

    def _select_key_methods(self, methods: list[MethodFact]) -> list[MethodFact]:
        ranked = sorted(
            methods,
            key=lambda method: (-method_importance(method), method.file_path, method.start_line),
        )
        return ranked[: self.config.max_key_methods]

    def _select_llm_files(
        self,
        files: list[SourceFile],
        java_analyses: list[JavaFileAnalysis],
    ) -> list[SourceFile]:
        methods_by_path = {analysis.file_path: analysis.methods for analysis in java_analyses}
        ranked = sorted(
            files,
            key=lambda file: (
                -file_importance(file, methods_by_path.get(file.relative_path, [])),
                file.relative_path,
            ),
        )
        return ranked[: self.config.max_llm_files]

    @staticmethod
    def _methods_by_path(methods: list[MethodFact]) -> dict[str, list[MethodFact]]:
        grouped: dict[str, list[MethodFact]] = defaultdict(list)
        for method in methods:
            grouped[method.file_path].append(method)
        return dict(grouped)

    def _chunk_metadata_tokens(self, file_path: str, methods: list[MethodFact]) -> int:
        metadata = {
            "chunk_wrapper_path": file_path,
            "methods": [
                {
                    "method_id": method.method_id,
                    "signature": method.signature,
                    "lines": [method.start_line, method.end_line],
                    "cyclomatic_complexity": method.cyclomatic_complexity,
                }
                for method in methods
            ],
        }
        return self.budgeter.count(json.dumps(metadata)) + 50

    def _bounded_project_evidence(
        self,
        base_evidence: dict[str, object],
        chunk_analyses: list,
        max_tokens: int,
    ) -> dict[str, object]:
        evidence = dict(base_evidence)
        evidence["file_analyses"] = []
        for analysis in chunk_analyses:
            candidate = list(evidence["file_analyses"])
            candidate.append(analysis.model_dump())
            evidence["file_analyses"] = candidate
            if self.budgeter.count(json.dumps(evidence)) > max_tokens:
                evidence["file_analyses"] = candidate[:-1]
                evidence["file_analysis_truncated"] = True
                evidence["file_analysis_count_included"] = len(candidate) - 1
                evidence["file_analysis_count_available"] = len(chunk_analyses)
                while (
                    self.budgeter.count(json.dumps(evidence)) > max_tokens
                    and evidence["file_analyses"]
                ):
                    evidence["file_analyses"].pop()
                    evidence["file_analysis_count_included"] = len(evidence["file_analyses"])
                break
        else:
            evidence["file_analysis_truncated"] = False
        return evidence


def method_importance(method: MethodFact) -> float:
    path = method.file_path.lower()
    annotations = {annotation.lower() for annotation in method.annotations}
    score = min(method.cyclomatic_complexity, 20) / 4
    if method.visibility == "public":
        score += 2
    if any("mapping" in annotation for annotation in annotations):
        score += 15
    if "controller" in path:
        score += 8
    if "/service/" in path:
        score += 7
    if "/security/" in path or "/auth/" in path:
        score += 7
    if "/repository/custom/" in path:
        score += 5
    if method.name == "main":
        score += 20
    if re.match(r"^(get|set|is|equals|hashCode|toString)$", method.name):
        score -= 5
    return score


def file_importance(file: SourceFile, methods: list[MethodFact]) -> float:
    path = file.relative_path.lower()
    score = max((method_importance(method) for method in methods), default=0)
    if path in {"readme.md", "build.gradle.kts", "settings.gradle.kts"}:
        score += 100
    if path.endswith(("application.yaml", "application.yml", "libs.versions.toml")):
        score += 90
    if any(term in path for term in ("controller", "serviceimpl", "security", "authconfig")):
        score += 20
    if any(term in path for term in ("entity", "dto", "mapper", "serializer")):
        score -= 5
    return score


def build_complexity_summary(methods: list[MethodFact]) -> ComplexitySummary:
    values = [method.cyclomatic_complexity for method in methods]
    counts = Counter(method.complexity_rating for method in methods)
    hotspots = sorted(
        methods,
        key=lambda method: (-method.cyclomatic_complexity, method.file_path, method.start_line),
    )[:10]
    return ComplexitySummary(
        metric="Cyclomatic complexity (1 + decision points)",
        average_per_method=round(sum(values) / len(values), 2) if values else 0.0,
        maximum=max(values, default=0),
        low_count=counts["low"],
        moderate_count=counts["moderate"],
        high_count=counts["high"],
        very_high_count=counts["very_high"],
        hotspots=[
            ComplexityHotspot(
                method_id=method.method_id,
                file_path=method.file_path,
                signature=method.signature,
                cyclomatic_complexity=method.cyclomatic_complexity,
                complexity_rating=method.complexity_rating,
            )
            for method in hotspots
        ],
    )


def detect_technologies(files: list[SourceFile]) -> list[str]:
    evidence = "\n".join(
        file.content
        for file in files
        if file.relative_path.lower().endswith(("readme.md", ".gradle.kts", ".toml", ".yaml"))
    ).lower()
    patterns = {
        "Java 17": ("java 17", "javalanguageversion.of(17)"),
        "Spring Boot": ("spring boot", "spring.boot"),
        "Spring Web": ("spring.web", "spring-boot-starter-web"),
        "Spring Data JPA": ("spring.data.jpa", "spring-boot-starter-data-jpa"),
        "Spring Security": ("spring.security", "spring-boot-starter-security"),
        "Spring HATEOAS": ("spring.hateoas", "spring-boot-starter-hateoas"),
        "MySQL": ("mysql",),
        "Redis": ("redis",),
        "Querydsl": ("querydsl",),
        "MapStruct": ("mapstruct",),
        "Blaze-Persistence": ("blaze-persistence", "blaze.persistence"),
        "Lombok": ("lombok",),
        "JWT": ("jjwt", "jwt"),
        "Spring REST Docs": ("restdocs", "rest docs"),
        "Spring Boot Actuator": ("actuator",),
        "Prometheus": ("prometheus",),
        "Gradle": ("gradle",),
    }
    return [name for name, needles in patterns.items() if any(item in evidence for item in needles)]


def detect_modules(files: list[SourceFile]) -> list[str]:
    modules: set[str] = set()
    prefix = "src/main/java/com/example/app/"
    for file in files:
        path = file.relative_path
        if not path.startswith(prefix):
            continue
        remainder = path[len(prefix) :]
        parts = remainder.split("/")
        if parts[0] == "services" and len(parts) > 1:
            modules.add(f"services/{parts[1]}")
        elif len(parts) > 1:
            modules.add(parts[0])
        else:
            modules.add("application")
    return sorted(modules)


def humanize_identifier(value: str) -> str:
    spaced = re.sub(r"(?<!^)(?=[A-Z])", " ", value).replace("_", " ")
    return re.sub(r"\s+", " ", spaced).strip().lower()


def _source_fingerprint(files: list[SourceFile]) -> str:
    ordered_sources = [
        {"path": file.relative_path, "content": file.content}
        for file in sorted(files, key=lambda item: item.relative_path)
    ]
    return hashlib.sha256(canonical_json(ordered_sources).encode("utf-8")).hexdigest()


def _scope_statistics(
    discovery: DiscoveryResult,
    java_analyses: list[JavaFileAnalysis],
    methods: list[MethodFact],
    selected_files: list[SourceFile],
) -> ScopeStatistics:
    return ScopeStatistics(
        files_discovered=discovery.discovered_count,
        files_processed=len(discovery.files),
        files_sent_to_llm=len(selected_files),
        java_files=len(java_analyses),
        java_methods=len(methods),
        source_lines=sum(file.line_count for file in discovery.files),
        skipped_large_files=discovery.skipped_large_count,
        excluded_files=discovery.excluded_count,
    )
