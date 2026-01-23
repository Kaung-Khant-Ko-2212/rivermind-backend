from __future__ import annotations

from dataclasses import dataclass
import os


def _env_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass
class AppConfig:
    replay_enabled: bool = False
    replay_capacity: int = 10000
    ai_mode: str = "random"
    ai_seed: int | None = None

    @classmethod
    def from_env(cls) -> "AppConfig":
        enabled = _env_bool(os.getenv("REPLAY_ENABLED", "false"))
        capacity = int(os.getenv("REPLAY_CAPACITY", "10000"))
        ai_mode = os.getenv("AI_MODE", "random").strip().lower()
        ai_seed_raw = os.getenv("AI_SEED")
        ai_seed = int(ai_seed_raw) if ai_seed_raw else None
        return cls(
            replay_enabled=enabled,
            replay_capacity=capacity,
            ai_mode=ai_mode,
            ai_seed=ai_seed,
        )
