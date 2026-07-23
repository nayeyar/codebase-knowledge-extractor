from pathlib import Path

from codebase_analyzer.discovery import discover_source_files, redact_likely_secrets


def test_redacts_common_key_value_secrets() -> None:
    text = "password: secret123\napi-key=abc123\nhost: localhost\n"
    redacted = redact_likely_secrets(text)
    assert "secret123" not in redacted
    assert "abc123" not in redacted
    assert redacted.count("[REDACTED]") == 2
    assert "host: localhost" in redacted


def test_discovery_filters_build_test_and_large_files(tmp_path: Path) -> None:
    (tmp_path / "src" / "main").mkdir(parents=True)
    (tmp_path / "src" / "test").mkdir(parents=True)
    (tmp_path / "build").mkdir()
    (tmp_path / "src" / "main" / "App.java").write_text("class App {}", encoding="utf-8")
    (tmp_path / "src" / "test" / "AppTest.java").write_text("class AppTest {}", encoding="utf-8")
    (tmp_path / "build" / "Generated.java").write_text("class Generated {}", encoding="utf-8")
    (tmp_path / "README.md").write_text("x" * 50, encoding="utf-8")

    result = discover_source_files(tmp_path, max_file_bytes=20, include_tests=False)

    assert [file.relative_path for file in result.files] == ["src/main/App.java"]
    assert result.skipped_large_count == 1
    assert result.excluded_count == 2
