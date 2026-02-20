from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


_LOADED = False


def _candidate_paths() -> Iterable[Path]:
    explicit = os.getenv("APP_ENV_FILE")
    if explicit:
        yield Path(explicit)

    cwd_env = Path.cwd() / ".env"
    yield cwd_env

    project_root_env = Path(__file__).resolve().parents[1] / ".env"
    yield project_root_env


def _parse_line(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None

    if stripped.startswith("export "):
        stripped = stripped[len("export ") :].strip()

    if "=" not in stripped:
        return None

    key, value = stripped.split("=", 1)
    key = key.strip()
    value = value.strip()
    if not key:
        return None

    if (
        len(value) >= 2
        and ((value[0] == '"' and value[-1] == '"') or (value[0] == "'" and value[-1] == "'"))
    ):
        value = value[1:-1]

    return key, value


def load_env_file(force: bool = False) -> None:
    global _LOADED
    if _LOADED and not force:
        return

    for path in _candidate_paths():
        try:
            if not path.is_file():
                continue
            for raw_line in path.read_text(encoding="utf-8").splitlines():
                parsed = _parse_line(raw_line)
                if not parsed:
                    continue
                key, value = parsed
                os.environ.setdefault(key, value)
            _LOADED = True
            return
        except OSError:
            continue
