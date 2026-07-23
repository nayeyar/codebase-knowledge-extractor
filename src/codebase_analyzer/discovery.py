from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

SUPPORTED_SUFFIXES = {
    ".java",
    ".kt",
    ".kts",
    ".gradle",
    ".xml",
    ".yaml",
    ".yml",
    ".properties",
    ".toml",
    ".sql",
    ".md",
    ".adoc",
}

EXCLUDED_DIRECTORY_NAMES = {
    ".git",
    ".gradle",
    ".idea",
    ".vscode",
    "build",
    "dist",
    "generated",
    "node_modules",
    "target",
    "vendor",
}

SENSITIVE_FILE_NAMES = {
    ".env",
    ".env.local",
    "id_rsa",
    "id_ed25519",
}

SECRET_VALUE_PATTERN = re.compile(
    r"(?im)^(\s*(?:password|passwd|secret|token|api[-_.]?key|signing[-_.]?key)\s*[:=]\s*)"
    r"([^\s#]+)"
)


@dataclass(frozen=True)
class SourceFile:
    path: Path
    relative_path: str
    language: str
    size_bytes: int
    line_count: int
    content: str


@dataclass(frozen=True)
class DiscoveryResult:
    files: list[SourceFile]
    discovered_count: int
    skipped_large_count: int
    excluded_count: int


def detect_language(path: Path) -> str:
    suffix = path.suffix.lower()
    return {
        ".java": "Java",
        ".kt": "Kotlin",
        ".kts": "Kotlin DSL",
        ".gradle": "Gradle",
        ".xml": "XML",
        ".yaml": "YAML",
        ".yml": "YAML",
        ".properties": "Properties",
        ".toml": "TOML",
        ".sql": "SQL",
        ".md": "Markdown",
        ".adoc": "AsciiDoc",
    }.get(suffix, "Text")


def redact_likely_secrets(content: str) -> str:
    """Redact common key/value secrets before repository text is sent to an LLM."""

    return SECRET_VALUE_PATTERN.sub(r"\1[REDACTED]", content)


def discover_source_files(
    root: Path,
    *,
    max_file_bytes: int,
    include_tests: bool,
) -> DiscoveryResult:
    root = root.resolve()
    files: list[SourceFile] = []
    discovered = 0
    skipped_large = 0
    excluded = 0

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        discovered += 1
        relative = path.relative_to(root)
        parts = set(relative.parts)

        if parts & EXCLUDED_DIRECTORY_NAMES:
            excluded += 1
            continue
        if path.name.lower() in SENSITIVE_FILE_NAMES:
            excluded += 1
            continue
        if not include_tests and "test" in relative.parts:
            excluded += 1
            continue
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            excluded += 1
            continue

        size = path.stat().st_size
        if size > max_file_bytes:
            skipped_large += 1
            continue

        try:
            raw = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            excluded += 1
            continue

        files.append(
            SourceFile(
                path=path,
                relative_path=relative.as_posix(),
                language=detect_language(path),
                size_bytes=size,
                line_count=raw.count("\n") + (1 if raw else 0),
                content=redact_likely_secrets(raw),
            )
        )

    return DiscoveryResult(
        files=files,
        discovered_count=discovered,
        skipped_large_count=skipped_large,
        excluded_count=excluded,
    )
