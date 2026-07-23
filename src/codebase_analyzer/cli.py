from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

from dotenv import load_dotenv

from codebase_analyzer.cache import atomic_write_text
from codebase_analyzer.config import (
    AnalyzerConfig,
    RunConfigError,
    load_run_config_arguments,
)
from codebase_analyzer.llm import (
    OfflineKnowledgeModel,
    OllamaKnowledgeModel,
    OpenAIKnowledgeModel,
)
from codebase_analyzer.pipeline import CodebaseAnalysisPipeline
from codebase_analyzer.repository import clone_repository, resolve_commit

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="codebase-analyzer",
        description="Extract structured codebase knowledge with static analysis and an LLM.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="TOML run profile; explicit CLI options override profile values",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--repo-url", help="Git repository URL to clone and analyze")
    source.add_argument("--source-path", type=Path, help="Existing local repository path")
    parser.add_argument(
        "--ref",
        help="Requested branch, tag, or revision recorded with the analysis",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/codebase-analysis.json"),
        help="JSON report path (default: %(default)s)",
    )
    parser.add_argument(
        "--provider",
        choices=("openai", "ollama"),
        default=os.getenv("CODE_ANALYZER_PROVIDER", "openai"),
        help="LLM provider (default: %(default)s)",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("CODE_ANALYZER_MODEL"),
        help="Provider model name (defaults to gpt-5.4-mini or qwen3.5:latest)",
    )
    parser.add_argument(
        "--ollama-base-url",
        default=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        help="Ollama server URL; /v1 is added when omitted (default: %(default)s)",
    )
    parser.add_argument(
        "--ollama-timeout-seconds",
        type=float,
        default=float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "7200")),
        help=(
            "Ollama response/read timeout in seconds; this is not a whole-stage deadline "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--openai-timeout-seconds",
        type=float,
        default=float(os.getenv("OPENAI_TIMEOUT_SECONDS", "180")),
        help=(
            "OpenAI response/read timeout in seconds; this is not a whole-stage deadline "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--connect-timeout-seconds",
        type=float,
        default=float(os.getenv("CONNECT_TIMEOUT_SECONDS", "10")),
        help="Connection and connection-pool timeout in seconds (default: %(default)s)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        help="Application-level transient retries (default: Ollama 0, OpenAI 3)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".codebase-analyzer-cache"),
        help="Validated map/reduce cache directory (default: %(default)s)",
    )
    parser.add_argument(
        "--cache-mode",
        choices=("use", "refresh", "off"),
        default="use",
        help="Cache behavior: reuse, replace after fresh calls, or disable (default: %(default)s)",
    )
    parser.add_argument("--include-tests", action="store_true")
    parser.add_argument("--max-llm-files", type=int, default=60)
    parser.add_argument("--max-key-methods", type=int, default=75)
    parser.add_argument(
        "--offline",
        action="store_true",
        help=(
            "Run deterministic pipeline validation without API calls; "
            "not for final submission output"
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging and tracebacks for failures",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    load_dotenv(dotenv_path=Path(".env"), override=False)
    parser = build_parser()
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path)
    config_args, _ = config_parser.parse_known_args(raw_argv)
    profile_arguments: list[str] = []
    if config_args.config is not None:
        try:
            profile_arguments = load_run_config_arguments(config_args.config, raw_argv)
        except RunConfigError as exc:
            parser.error(str(exc))
    args = parser.parse_args([*profile_arguments, *raw_argv])
    _configure_logging(args.debug)
    if args.config is not None:
        logger.info("Loaded run configuration from %s", args.config)
    started = time.monotonic()
    stage = "configuration"

    try:
        provider = "offline" if args.offline else args.provider
        if provider == "openai" and not os.getenv("OPENAI_API_KEY"):
            parser.error(
                "OPENAI_API_KEY is required for the LLM run. Export it in your shell, or use "
                "--offline only to validate the local pipeline."
            )
        _validate_positive(parser, "--ollama-timeout-seconds", args.ollama_timeout_seconds)
        _validate_positive(parser, "--openai-timeout-seconds", args.openai_timeout_seconds)
        _validate_positive(parser, "--connect-timeout-seconds", args.connect_timeout_seconds)
        if args.max_retries is not None and args.max_retries < 0:
            parser.error("--max-retries must be nonnegative")

        model_name = args.model or ("qwen3.5:latest" if provider == "ollama" else "gpt-5.4-mini")
        max_retries = args.max_retries
        if max_retries is None:
            max_retries = 0 if provider == "ollama" else 3

        config = AnalyzerConfig(
            provider=provider,
            model=model_name,
            max_llm_files=args.max_llm_files,
            max_key_methods=args.max_key_methods,
            include_tests=args.include_tests,
            cache_dir=args.cache_dir,
            cache_mode=args.cache_mode,
        )
        if provider == "offline":
            model = OfflineKnowledgeModel()
        elif provider == "ollama":
            model = OllamaKnowledgeModel(
                model=config.model,
                base_url=args.ollama_base_url,
                api_key=os.getenv("OLLAMA_API_KEY", "ollama"),
                cache_dir=config.cache_dir,
                cache_mode=config.cache_mode,
                max_output_tokens=config.max_output_tokens,
                read_timeout_seconds=args.ollama_timeout_seconds,
                connect_timeout_seconds=args.connect_timeout_seconds,
                max_retries=max_retries,
            )
        else:
            model = OpenAIKnowledgeModel(
                model=config.model,
                cache_dir=config.cache_dir,
                cache_mode=config.cache_mode,
                max_output_tokens=config.max_output_tokens,
                read_timeout_seconds=args.openai_timeout_seconds,
                connect_timeout_seconds=args.connect_timeout_seconds,
                max_retries=max_retries,
            )

        output_path = args.output.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if args.source_path:
            stage = "repository acquisition"
            root = args.source_path.resolve()
            logger.info("Using local repository %s (requested ref: %s)", root, args.ref or "none")
            if not root.is_dir():
                parser.error(f"Source path is not a directory: {root}")
            stage = "analysis"
            report = _analyze(
                root,
                config=config,
                model=model,
                repository_url=None,
                requested_ref=args.ref,
            )
        else:
            stage = "repository acquisition"
            logger.info(
                "Cloning repository %s (requested ref: %s)",
                args.repo_url,
                args.ref or "default",
            )
            with tempfile.TemporaryDirectory(prefix="codebase-analyzer-") as temporary:
                root = Path(temporary) / "repository"
                clone_repository(args.repo_url, args.ref, root)
                logger.info("Repository acquisition completed: %s", root)
                stage = "analysis"
                report = _analyze(
                    root,
                    config=config,
                    model=model,
                    repository_url=args.repo_url,
                    requested_ref=args.ref,
                )

        stage = "final output writing"
        logger.info("Writing final validated report atomically to %s", output_path)
        atomic_write_text(output_path, report.model_dump_json(indent=2) + "\n")
        logger.info(
            "Analysis completed in %.1fs; output written to %s",
            time.monotonic() - started,
            output_path,
        )
        return 0
    except KeyboardInterrupt:
        logger.warning(
            "Analysis interrupted during %s after %.1fs; completed cache entries were preserved",
            stage,
            time.monotonic() - started,
        )
        return 130
    except Exception as exc:
        if args.debug:
            logger.exception(
                "Analysis failed during %s after %.1fs",
                stage,
                time.monotonic() - started,
            )
        else:
            logger.error(
                "Analysis failed during %s after %.1fs: %s",
                stage,
                time.monotonic() - started,
                exc,
            )
        return 1


def _configure_logging(debug: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )


def _validate_positive(
    parser: argparse.ArgumentParser,
    option: str,
    value: float,
) -> None:
    if value <= 0:
        parser.error(f"{option} must be positive")


def _analyze(root: Path, *, config, model, repository_url, requested_ref):
    resolved_commit = resolve_commit(root)
    logger.info(
        "Repository identity established: requested ref=%s, resolved SHA=%s",
        requested_ref or "none",
        resolved_commit or "unavailable",
    )
    pipeline = CodebaseAnalysisPipeline(config, model)
    return pipeline.run(
        root,
        repository_url=repository_url,
        requested_ref=requested_ref,
        resolved_commit=resolved_commit,
    )


if __name__ == "__main__":
    sys.exit(main())
