from __future__ import annotations

from dataclasses import dataclass
import os

from .env_loader import load_env_file


load_env_file()


def _env_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass
class AppConfig:
    replay_enabled: bool = False
    replay_capacity: int = 10000
    ai_mode: str = "random"
    ai_seed: int | None = None
    ai_turn_delay_ms: int = 800
    hand_end_pause_ms: int = 5000
    game_trace: bool = True

    @classmethod
    def from_env(cls) -> "AppConfig":
        enabled = _env_bool(os.getenv("REPLAY_ENABLED", "false"))
        capacity = int(os.getenv("REPLAY_CAPACITY", "10000"))
        ai_mode = os.getenv("AI_MODE", "random").strip().lower()
        ai_seed_raw = os.getenv("AI_SEED")
        ai_seed = int(ai_seed_raw) if ai_seed_raw else None
        ai_turn_delay_raw = os.getenv("AI_TURN_DELAY_MS", "800")
        try:
            ai_turn_delay_ms = int(ai_turn_delay_raw)
        except ValueError:
            ai_turn_delay_ms = 800
        ai_turn_delay_ms = max(0, ai_turn_delay_ms)
        hand_end_pause_raw = os.getenv("HAND_END_PAUSE_MS", "5000")
        try:
            hand_end_pause_ms = int(hand_end_pause_raw)
        except ValueError:
            hand_end_pause_ms = 5000
        hand_end_pause_ms = max(0, hand_end_pause_ms)
        game_trace = _env_bool(os.getenv("GAME_TRACE", "true"))
        return cls(
            replay_enabled=enabled,
            replay_capacity=capacity,
            ai_mode=ai_mode,
            ai_seed=ai_seed,
            ai_turn_delay_ms=ai_turn_delay_ms,
            hand_end_pause_ms=hand_end_pause_ms,
            game_trace=game_trace,
        )
