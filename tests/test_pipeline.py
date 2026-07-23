import logging
from pathlib import Path

import pytest

from codebase_analyzer.config import AnalyzerConfig
from codebase_analyzer.discovery import SourceFile
from codebase_analyzer.llm import OfflineKnowledgeModel
from codebase_analyzer.pipeline import CodebaseAnalysisPipeline, _source_fingerprint

FIXTURE = Path(__file__).parent / "fixtures" / "SampleController.java"


def test_offline_pipeline_produces_valid_report(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    source_dir = tmp_path / "sample-repo" / "src" / "main" / "java" / "com" / "example"
    source_dir.mkdir(parents=True)
    (source_dir / "SampleController.java").write_text(FIXTURE.read_text(), encoding="utf-8")
    (tmp_path / "sample-repo" / "README.md").write_text(
        "Spring Boot REST API backed by MySQL", encoding="utf-8"
    )

    config = AnalyzerConfig(
        model="gpt-5.4-mini",
        max_chunk_tokens=300,
        max_batch_tokens=800,
        prompt_reserve_tokens=200,
        max_llm_files=5,
        max_key_methods=5,
        cache_dir=tmp_path / "cache",
    )
    caplog.set_level(logging.INFO)
    report = CodebaseAnalysisPipeline(config, OfflineKnowledgeModel()).run(
        tmp_path / "sample-repo",
        repository_url="https://example.test/sample.git",
        requested_ref="main",
        resolved_commit="abc123",
    )

    assert report.generation.mode == "offline"
    assert report.generation.provider == "offline"
    assert report.source.resolved_commit == "abc123"
    assert report.scope.java_files == 1
    assert report.scope.java_methods == 1
    assert report.key_methods[0].signature == "public List<String> findActive(List<String> values)"
    assert report.complexity.maximum == 4
    assert "Spring Boot" in report.technologies
    assert report.token_budget.map_batch_count >= 1
    assert (
        report.token_budget.largest_estimated_map_input_tokens
        <= report.token_budget.max_batch_input_tokens
    )
    assert "Discovery completed" in caplog.text
    assert "Static Java parsing completed" in caplog.text
    assert "map batch 1/" in caplog.text
    assert "estimated prompt tokens" in caplog.text
    assert "reduce started" in caplog.text
    assert "reduce completed" in caplog.text


def test_source_fingerprint_is_ordered_and_changes_with_path_or_content(tmp_path: Path) -> None:
    first_path = tmp_path / "A.java"
    second_path = tmp_path / "B.java"
    first = SourceFile(
        path=first_path,
        relative_path="A.java",
        language="Java",
        size_bytes=10,
        line_count=1,
        content="class A {}",
    )
    second = SourceFile(
        path=second_path,
        relative_path="B.java",
        language="Java",
        size_bytes=10,
        line_count=1,
        content="class B {}",
    )

    baseline = _source_fingerprint([first, second])

    assert baseline == _source_fingerprint([second, first])
    assert baseline != _source_fingerprint(
        [
            first,
            SourceFile(
                path=second.path,
                relative_path=second.relative_path,
                language=second.language,
                size_bytes=second.size_bytes,
                line_count=second.line_count,
                content="class B { int changed; }",
            ),
        ]
    )
    assert baseline != _source_fingerprint(
        [
            SourceFile(
                path=first.path,
                relative_path="renamed/A.java",
                language=first.language,
                size_bytes=first.size_bytes,
                line_count=first.line_count,
                content=first.content,
            ),
            second,
        ]
    )
