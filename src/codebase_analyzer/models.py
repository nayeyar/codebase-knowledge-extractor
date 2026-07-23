from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


ComplexityRating = Literal["low", "moderate", "high", "very_high"]


class MethodFact(StrictModel):
    method_id: str
    file_path: str
    class_name: str
    name: str
    signature: str
    start_line: int
    end_line: int
    cyclomatic_complexity: int
    complexity_rating: ComplexityRating
    annotations: list[str]
    visibility: str


class JavaFileAnalysis(StrictModel):
    file_path: str
    package: str
    classes: list[str]
    line_count: int
    methods: list[MethodFact]


class MethodDescription(StrictModel):
    method_id: str = Field(description="Exact method identifier supplied in the input")
    description: str = Field(description="Concise behavior and business-purpose description")


class ChunkAnalysis(StrictModel):
    chunk_id: str = Field(description="Exact chunk identifier supplied in the input")
    purpose: str
    responsibilities: list[str]
    noteworthy_aspects: list[str]
    method_descriptions: list[MethodDescription]


class BatchChunkAnalysis(StrictModel):
    chunks: list[ChunkAnalysis]


class ArchitectureLayer(StrictModel):
    name: str
    responsibility: str
    representative_paths: list[str]


class ProjectSynthesis(StrictModel):
    purpose: str
    overview: str
    primary_capabilities: list[str]
    architecture_style: str
    architecture_layers: list[ArchitectureLayer]
    typical_request_flow: list[str]
    noteworthy_aspects: list[str]
    assumptions_and_limitations: list[str]


class RepositorySource(StrictModel):
    repository_url: str | None
    requested_ref: str | None
    resolved_commit: str | None
    analyzed_path: str


class GenerationMetadata(StrictModel):
    generated_at: datetime
    generator_version: str
    mode: Literal["llm", "offline"]
    provider: Literal["openai", "ollama", "offline"]
    model: str
    token_strategy: str
    prompt_version: str


class ScopeStatistics(StrictModel):
    files_discovered: int
    files_processed: int
    files_sent_to_llm: int
    java_files: int
    java_methods: int
    source_lines: int
    skipped_large_files: int
    excluded_files: int


class TokenBudgetSummary(StrictModel):
    encoding: str
    max_chunk_tokens: int
    max_batch_input_tokens: int
    prompt_reserve_tokens: int
    max_output_tokens: int
    max_chunks_per_batch: int
    selected_source_tokens: int
    map_chunk_count: int
    map_batch_count: int
    largest_estimated_map_input_tokens: int
    reduce_evidence_tokens: int


class ComplexityHotspot(StrictModel):
    method_id: str
    file_path: str
    signature: str
    cyclomatic_complexity: int
    complexity_rating: ComplexityRating


class ComplexitySummary(StrictModel):
    metric: str
    average_per_method: float
    maximum: int
    low_count: int
    moderate_count: int
    high_count: int
    very_high_count: int
    hotspots: list[ComplexityHotspot]


class KeyMethod(StrictModel):
    method_id: str
    file_path: str
    class_name: str
    signature: str
    description: str
    start_line: int
    end_line: int
    cyclomatic_complexity: int
    complexity_rating: ComplexityRating
    annotations: list[str]


class AnalysisReport(StrictModel):
    schema_version: str
    source: RepositorySource
    generation: GenerationMetadata
    scope: ScopeStatistics
    token_budget: TokenBudgetSummary
    project: ProjectSynthesis
    technologies: list[str]
    modules: list[str]
    complexity: ComplexitySummary
    key_methods: list[KeyMethod]
    analyzed_file_summaries: list[ChunkAnalysis]
