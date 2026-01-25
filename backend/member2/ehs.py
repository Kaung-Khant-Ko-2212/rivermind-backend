from __future__ import annotations

import math
import multiprocessing as mp
import random
import shelve
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

from .cards import deck_excluding
from .evaluator import compare_hands


def _simulate_chunk(
    hero: Tuple[int, int],
    board: Tuple[int, ...],
    n_opponents: int,
    rollouts: int,
    seed: Optional[int],
) -> Tuple[int, int, int]:
    rng = random.Random(seed)
    wins = 0
    ties = 0
    board_len = len(board)
    for _ in range(rollouts):
        used = list(hero) + list(board)
        deck = deck_excluding(used)
        needed_board = 5 - board_len
        draw_count = needed_board + (2 * n_opponents)
        draw = rng.sample(deck, draw_count)

        board_draw = draw[:needed_board]
        opp_cards = draw[needed_board:]

        full_board = list(board) + board_draw
        hero_cards7 = list(hero) + full_board

        best_result = 1
        for i in range(n_opponents):
            opp = opp_cards[i * 2:(i + 1) * 2]
            opp_cards7 = opp + full_board
            result = compare_hands(hero_cards7, opp_cards7)
            if result < best_result:
                best_result = result
                if best_result == -1:
                    break

        if best_result == 1:
            wins += 1
        elif best_result == 0:
            ties += 1
    return wins, ties, rollouts


@dataclass
class EHSResult:
    ehs: float
    wins: int
    ties: int
    rollouts: int


class EHSEngine:
    def __init__(
        self,
        rollouts: int = 1000,
        n_opponents: int = 1,
        seed: Optional[int] = None,
        cache_path: Optional[str] = None,
        n_jobs: int = 1,
    ) -> None:
        if rollouts <= 0:
            raise ValueError("rollouts must be positive")
        if n_opponents <= 0:
            raise ValueError("n_opponents must be positive")
        self.rollouts = rollouts
        self.n_opponents = n_opponents
        self.seed = seed
        self.cache_path = cache_path
        self.n_jobs = max(1, n_jobs)
        self._cache = {}

    def _cache_key(self, hero: Tuple[int, int], board: Tuple[int, ...]) -> str:
        return f"{hero}|{board}|{self.rollouts}|{self.n_opponents}|{self.seed}"

    def _load_cache(self, key: str) -> Optional[EHSResult]:
        if key in self._cache:
            return self._cache[key]
        if self.cache_path:
            with shelve.open(self.cache_path) as db:
                if key in db:
                    self._cache[key] = db[key]
                    return db[key]
        return None

    def _store_cache(self, key: str, result: EHSResult) -> None:
        self._cache[key] = result
        if self.cache_path:
            with shelve.open(self.cache_path) as db:
                db[key] = result

    def compute(self, hero: Iterable[int], board: Iterable[int]) -> EHSResult:
        hero_t = tuple(hero)
        board_t = tuple(board)
        if len(hero_t) != 2:
            raise ValueError("hero must contain exactly 2 cards")
        if len(set(hero_t)) != 2:
            raise ValueError("hero has duplicate cards")
        if len(board_t) > 5:
            raise ValueError("board cannot exceed 5 cards")
        if len(set(board_t)) != len(board_t):
            raise ValueError("board has duplicate cards")
        if set(hero_t).intersection(board_t):
            raise ValueError("hero and board overlap")

        key = self._cache_key(hero_t, board_t)
        cached = self._load_cache(key)
        if cached:
            return cached

        if self.n_jobs == 1:
            wins, ties, rollouts = _simulate_chunk(
                hero_t, board_t, self.n_opponents, self.rollouts, self.seed
            )
        else:
            chunks = _split_rollouts(self.rollouts, self.n_jobs)
            ctx = mp.get_context("spawn")
            with ctx.Pool(processes=self.n_jobs) as pool:
                tasks = []
                for i, chunk in enumerate(chunks):
                    seed = None if self.seed is None else self.seed + i
                    tasks.append(
                        pool.apply_async(
                            _simulate_chunk,
                            (hero_t, board_t, self.n_opponents, chunk, seed),
                        )
                    )
                wins = ties = rollouts = 0
                for t in tasks:
                    w, ti, r = t.get()
                    wins += w
                    ties += ti
                    rollouts += r

        ehs = (wins + ties * 0.5) / rollouts
        result = EHSResult(ehs=ehs, wins=wins, ties=ties, rollouts=rollouts)
        self._store_cache(key, result)
        return result


def _split_rollouts(rollouts: int, n_jobs: int) -> list[int]:
    base = rollouts // n_jobs
    remainder = rollouts % n_jobs
    return [base + (1 if i < remainder else 0) for i in range(n_jobs)]


def estimate_ehs(
    hero: Iterable[int],
    board: Iterable[int],
    rollouts: int = 1000,
    n_opponents: int = 1,
    seed: Optional[int] = None,
    cache_path: Optional[str] = None,
    n_jobs: int = 1,
) -> EHSResult:
    engine = EHSEngine(
        rollouts=rollouts,
        n_opponents=n_opponents,
        seed=seed,
        cache_path=cache_path,
        n_jobs=n_jobs,
    )
    return engine.compute(hero, board)
