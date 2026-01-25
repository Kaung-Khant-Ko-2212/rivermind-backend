from __future__ import annotations

import csv
import json
from dataclasses import asdict
from typing import Iterable

from .cards import parse_cards
from .ehs import EHSEngine


def _load_states(path: str) -> list[dict]:
    states: list[dict] = []
    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                states.append(json.loads(line))
    elif path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            states = json.load(f)
    else:
        raise ValueError("Input must be .json or .jsonl")
    return states


def export_dataset(
    input_path: str,
    output_path: str,
    rollouts: int = 1000,
    n_opponents: int = 1,
    seed: int | None = None,
    cache_path: str | None = None,
    n_jobs: int = 1,
) -> None:
    states = _load_states(input_path)
    engine = EHSEngine(
        rollouts=rollouts,
        n_opponents=n_opponents,
        seed=seed,
        cache_path=cache_path,
        n_jobs=n_jobs,
    )
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "hero",
                "board",
                "ehs",
                "wins",
                "ties",
                "rollouts",
            ],
        )
        writer.writeheader()
        for state in states:
            hero = parse_cards(state.get("hero"))
            board = parse_cards(state.get("board"))
            result = engine.compute(hero, board)
            writer.writerow(
                {
                    "hero": " ".join(state.get("hero", [])) if not isinstance(state.get("hero"), str) else state.get("hero"),
                    "board": " ".join(state.get("board", [])) if not isinstance(state.get("board"), str) else state.get("board"),
                    **asdict(result),
                }
            )
