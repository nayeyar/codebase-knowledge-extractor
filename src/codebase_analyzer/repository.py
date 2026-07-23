from __future__ import annotations

import subprocess
from pathlib import Path


class RepositoryError(RuntimeError):
    pass


def clone_repository(repository_url: str, ref: str | None, destination: Path) -> None:
    """Clone a repository without invoking a shell, preventing argument injection."""

    command = ["git", "clone", "--depth", "1"]
    if ref:
        command.extend(["--branch", ref])
    command.extend(["--", repository_url, str(destination)])
    _run(command, cwd=destination.parent)


def resolve_commit(root: Path) -> str | None:
    try:
        return _run(["git", "rev-parse", "HEAD"], cwd=root).strip()
    except RepositoryError:
        return None


def _run(command: list[str], *, cwd: Path) -> str:
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        stderr = getattr(exc, "stderr", "") or ""
        raise RepositoryError(f"Command failed: {' '.join(command)}\n{stderr.strip()}") from exc
    return result.stdout
