#!/usr/bin/env python3
"""Task-neutral runtime helpers for the three prediction wrappers."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def emit(payload: Mapping[str, Any]) -> None:
    print(json.dumps(dict(payload), ensure_ascii=False, indent=2))


def fail(message: str, *, code: str, details: Mapping[str, Any] | None = None) -> None:
    emit(
        {
            "status": "error",
            "error": {
                "code": code,
                "message": message,
                "details": dict(details or {}),
            },
        }
    )
    raise SystemExit(2)


def project_root(value: str | None) -> Path:
    configured = value or os.environ.get("MOLNEXUS_PROJECT_ROOT")
    root = Path(configured).expanduser() if configured else DEFAULT_PROJECT_ROOT
    root = root.resolve()
    if not root.is_dir():
        fail(
            "MolNexus project root is not an existing directory.",
            code="invalid_project_root",
            details={"project_root": str(root)},
        )
    return root


def local_path(value: str, *, root: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def require_file(value: str, *, label: str, root: Path) -> Path:
    path = local_path(value, root=root)
    if not path.is_file():
        fail(
            f"{label} is not an existing file.",
            code="missing_input_file",
            details={label: str(path)},
        )
    return path


def optional_file(value: str | None, *, label: str, root: Path) -> Path | None:
    if not value:
        return None
    return require_file(value, label=label, root=root)


def require_dir(value: str, *, label: str, root: Path) -> Path:
    path = local_path(value, root=root)
    if not path.is_dir():
        fail(
            f"{label} is not an existing directory.",
            code="missing_directory",
            details={label: str(path)},
        )
    return path


def run_or_dry_run(
    *,
    command: Sequence[str],
    root: Path,
    execute: bool,
    response: dict[str, Any],
    env_updates: Mapping[str, str] | None = None,
) -> int:
    response["status"] = "running" if execute else "dry_run"
    response["project_root"] = str(root)
    response["command"] = list(command)

    if not execute:
        emit(response)
        return 0

    env = os.environ.copy()
    env.update(dict(env_updates or {}))
    proc = subprocess.run(
        list(command),
        cwd=str(root),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    response["status"] = "completed" if proc.returncode == 0 else "error"
    response["returncode"] = proc.returncode
    response["stdout_tail"] = proc.stdout[-8000:]
    response["stderr_tail"] = proc.stderr[-8000:]
    if proc.returncode != 0:
        response["error"] = {
            "code": "prediction_failed",
            "message": "The task prediction command returned a non-zero exit code.",
        }
    emit(response)
    return 0 if proc.returncode == 0 else 1
