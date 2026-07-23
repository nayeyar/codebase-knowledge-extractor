import logging
import os
from pathlib import Path

import pytest

import codebase_analyzer.cli as cli_module
from codebase_analyzer.cli import build_parser, main


class FakeReport:
    def model_dump_json(self, *, indent: int) -> str:
        assert indent == 2
        return '{"valid": true}'


def test_cli_timeout_cache_and_retry_defaults() -> None:
    args = build_parser().parse_args(["--source-path", "."])

    assert args.ollama_timeout_seconds == 7_200
    assert args.openai_timeout_seconds == 180
    assert args.connect_timeout_seconds == 10
    assert args.max_retries is None
    assert args.cache_dir == Path(".codebase-analyzer-cache")
    assert args.cache_mode == "use"


def test_config_profile_loads_dotenv_and_cli_options_override_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = tmp_path / "analysis.toml"
    profile.write_text(
        """
[analysis]
provider = "openai"
model = "profile-model"
source_path = "."
output = "profile-output.json"
max_retries = 1
""".strip(),
        encoding="utf-8",
    )
    (tmp_path / ".env").write_text("OPENAI_API_KEY=dotenv-test-key\n", encoding="utf-8")
    captured: dict[str, object] = {}

    class FakeModel:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)
            captured["api_key_from_environment"] = os.getenv("OPENAI_API_KEY")

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(cli_module, "OpenAIKnowledgeModel", FakeModel)
    monkeypatch.setattr(cli_module, "_analyze", lambda *args, **kwargs: FakeReport())

    result = main(
        [
            "--config",
            str(profile),
            "--model",
            "cli-model",
            "--max-retries",
            "4",
        ]
    )

    assert result == 0
    assert captured["model"] == "cli-model"
    assert captured["max_retries"] == 4
    assert captured["api_key_from_environment"] == "dotenv-test-key"
    assert (tmp_path / "profile-output.json").is_file()


@pytest.mark.parametrize(
    ("provider", "expected_retries"),
    [("ollama", 0), ("openai", 3)],
)
def test_cli_applies_provider_specific_retry_defaults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    expected_retries: int,
) -> None:
    captured: dict[str, object] = {}

    class FakeModel:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setattr(
        cli_module,
        "OllamaKnowledgeModel" if provider == "ollama" else "OpenAIKnowledgeModel",
        FakeModel,
    )
    monkeypatch.setattr(cli_module, "_analyze", lambda *args, **kwargs: FakeReport())

    result = main(
        [
            "--provider",
            provider,
            "--source-path",
            str(tmp_path),
            "--output",
            str(tmp_path / "report.json"),
        ]
    )

    assert result == 0
    assert captured["max_retries"] == expected_retries


def test_cli_explicit_retry_setting_overrides_provider_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeModel:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(cli_module, "OllamaKnowledgeModel", FakeModel)
    monkeypatch.setattr(cli_module, "_analyze", lambda *args, **kwargs: FakeReport())

    result = main(
        [
            "--provider",
            "ollama",
            "--max-retries",
            "2",
            "--source-path",
            str(tmp_path),
            "--output",
            str(tmp_path / "report.json"),
        ]
    )

    assert result == 0
    assert captured["max_retries"] == 2


def test_cli_returns_one_without_traceback_and_preserves_old_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    output = tmp_path / "report.json"
    output.write_text('{"old": "valid"}\n', encoding="utf-8")

    def fail(*args: object, **kwargs: object) -> FakeReport:
        raise RuntimeError("simulated analysis failure")

    monkeypatch.setattr(cli_module, "_analyze", fail)
    caplog.set_level(logging.INFO)

    result = main(
        [
            "--offline",
            "--source-path",
            str(tmp_path),
            "--output",
            str(output),
        ]
    )

    assert result == 1
    assert output.read_text(encoding="utf-8") == '{"old": "valid"}\n'
    failure_record = next(
        record for record in caplog.records if "Analysis failed during analysis" in record.message
    )
    assert failure_record.exc_info is None
    assert "simulated analysis failure" in failure_record.message


def test_cli_debug_failure_includes_traceback_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def fail(*args: object, **kwargs: object) -> FakeReport:
        raise RuntimeError("debug failure")

    monkeypatch.setattr(cli_module, "_analyze", fail)

    result = main(
        [
            "--offline",
            "--debug",
            "--source-path",
            str(tmp_path),
            "--output",
            str(tmp_path / "report.json"),
        ]
    )

    assert result == 1
    failure_record = next(
        record for record in caplog.records if "Analysis failed during analysis" in record.message
    )
    assert failure_record.exc_info is not None


def test_cli_keyboard_interrupt_returns_130_and_preserves_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "report.json"
    output.write_text('{"old": "valid"}\n', encoding="utf-8")

    def interrupt(*args: object, **kwargs: object) -> FakeReport:
        raise KeyboardInterrupt

    monkeypatch.setattr(cli_module, "_analyze", interrupt)

    result = main(
        [
            "--offline",
            "--source-path",
            str(tmp_path),
            "--output",
            str(output),
        ]
    )

    assert result == 130
    assert output.read_text(encoding="utf-8") == '{"old": "valid"}\n'


def test_cli_success_writes_final_output_atomically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "nested" / "report.json"
    monkeypatch.setattr(cli_module, "_analyze", lambda *args, **kwargs: FakeReport())

    result = main(
        [
            "--offline",
            "--source-path",
            str(tmp_path),
            "--output",
            str(output),
        ]
    )

    assert result == 0
    assert output.read_text(encoding="utf-8") == '{"valid": true}\n'
    assert list(output.parent.glob(f".{output.name}.*.tmp")) == []
