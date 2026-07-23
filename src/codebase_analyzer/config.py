from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from codebase_analyzer.cache import CacheMode

RUN_CONFIG_OPTIONS = {
    "repo_url": "--repo-url",
    "source_path": "--source-path",
    "ref": "--ref",
    "output": "--output",
    "provider": "--provider",
    "model": "--model",
    "ollama_base_url": "--ollama-base-url",
    "ollama_timeout_seconds": "--ollama-timeout-seconds",
    "openai_timeout_seconds": "--openai-timeout-seconds",
    "connect_timeout_seconds": "--connect-timeout-seconds",
    "max_retries": "--max-retries",
    "cache_dir": "--cache-dir",
    "cache_mode": "--cache-mode",
    "include_tests": "--include-tests",
    "max_llm_files": "--max-llm-files",
    "max_key_methods": "--max-key-methods",
    "offline": "--offline",
    "debug": "--debug",
}
BOOLEAN_RUN_CONFIG_KEYS = {"include_tests", "offline", "debug"}
SOURCE_OPTIONS = {"--repo-url", "--source-path"}


class RunConfigError(ValueError):
    pass


@dataclass(frozen=True)
class AnalyzerConfig:
    """Runtime controls for repository processing and LLM usage."""

    provider: Literal["openai", "ollama", "offline"] = "openai"
    model: str = "gpt-5.4-mini"
    max_file_bytes: int = 512_000
    max_chunk_tokens: int = 8_000
    max_batch_tokens: int = 24_000
    prompt_reserve_tokens: int = 2_500
    max_output_tokens: int = 8_000
    max_chunks_per_batch: int = 12
    max_llm_files: int = 60
    max_key_methods: int = 75
    include_tests: bool = False
    cache_dir: Path = Path(".codebase-analyzer-cache")
    cache_mode: CacheMode = "use"

    def __post_init__(self) -> None:
        if self.provider not in {"openai", "ollama", "offline"}:
            raise ValueError(f"unsupported provider: {self.provider}")
        if self.max_chunk_tokens <= 0:
            raise ValueError("max_chunk_tokens must be positive")
        if self.max_batch_tokens <= self.prompt_reserve_tokens:
            raise ValueError("max_batch_tokens must exceed prompt_reserve_tokens")
        if self.max_chunk_tokens > self.max_batch_tokens - self.prompt_reserve_tokens:
            raise ValueError("max_chunk_tokens must fit within the usable batch budget")
        if self.max_chunks_per_batch <= 0:
            raise ValueError("max_chunks_per_batch must be positive")
        if self.max_llm_files <= 0 or self.max_key_methods <= 0:
            raise ValueError("selection limits must be positive")
        if self.cache_mode not in {"use", "refresh", "off"}:
            raise ValueError(f"unsupported cache mode: {self.cache_mode}")


def load_run_config_arguments(path: Path, explicit_argv: list[str]) -> list[str]:
    """Translate an [analysis] TOML profile into lower-precedence CLI arguments."""

    try:
        document = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as exc:
        raise RunConfigError(f"Unable to read config {path}: {exc}") from exc

    unexpected_sections = set(document) - {"analysis"}
    if unexpected_sections:
        raise RunConfigError(
            f"Unsupported top-level config sections: {sorted(unexpected_sections)}"
        )
    analysis = document.get("analysis")
    if not isinstance(analysis, dict):
        raise RunConfigError(f"Config {path} must contain an [analysis] table")

    unknown_keys = set(analysis) - set(RUN_CONFIG_OPTIONS)
    if unknown_keys:
        raise RunConfigError(f"Unsupported analysis config keys: {sorted(unknown_keys)}")

    explicit_options = {
        argument.split("=", 1)[0] for argument in explicit_argv if argument.startswith("--")
    }
    explicit_source = bool(explicit_options & SOURCE_OPTIONS)
    arguments: list[str] = []
    for key, value in analysis.items():
        option = RUN_CONFIG_OPTIONS[key]
        if explicit_source and option in SOURCE_OPTIONS:
            continue
        if key in BOOLEAN_RUN_CONFIG_KEYS:
            if not isinstance(value, bool):
                raise RunConfigError(f"Config value analysis.{key} must be true or false")
            if value:
                arguments.append(option)
            continue
        if isinstance(value, bool) or not isinstance(value, str | int | float):
            raise RunConfigError(f"Config value analysis.{key} must be a scalar")
        arguments.extend([option, str(value)])
    return arguments
