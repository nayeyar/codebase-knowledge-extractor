from pathlib import Path

import pytest

from codebase_analyzer.config import RunConfigError, load_run_config_arguments


def test_run_config_translates_analysis_values_to_cli_arguments(tmp_path: Path) -> None:
    profile = tmp_path / "analysis.toml"
    profile.write_text(
        """
[analysis]
provider = "openai"
model = "gpt-test"
repo_url = "https://example.test/repository.git"
ref = "main"
max_retries = 2
include_tests = true
debug = false
""".strip(),
        encoding="utf-8",
    )

    arguments = load_run_config_arguments(profile, [])

    assert arguments == [
        "--provider",
        "openai",
        "--model",
        "gpt-test",
        "--repo-url",
        "https://example.test/repository.git",
        "--ref",
        "main",
        "--max-retries",
        "2",
        "--include-tests",
    ]


def test_explicit_cli_source_replaces_configured_source(tmp_path: Path) -> None:
    profile = tmp_path / "analysis.toml"
    profile.write_text(
        """
[analysis]
repo_url = "https://example.test/repository.git"
ref = "main"
""".strip(),
        encoding="utf-8",
    )

    arguments = load_run_config_arguments(
        profile,
        ["--source-path", "/tmp/local-repository"],
    )

    assert "--repo-url" not in arguments
    assert arguments == ["--ref", "main"]


@pytest.mark.parametrize(
    "content",
    [
        "provider = 'openai'",
        "[unexpected]\nprovider = 'openai'",
        "[analysis]\nunknown = 'value'",
        "[analysis]\ninclude_tests = 'yes'",
        "[analysis]\nmax_retries = [1, 2]",
    ],
)
def test_invalid_run_configs_are_rejected(tmp_path: Path, content: str) -> None:
    profile = tmp_path / "invalid.toml"
    profile.write_text(content, encoding="utf-8")

    with pytest.raises(RunConfigError):
        load_run_config_arguments(profile, [])
