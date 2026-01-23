from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Sequence


RANKS: Sequence[str] = ("A", "K", "Q", "J", "T", "9", "8", "7", "6", "5", "4", "3", "2")
SUITS: Sequence[str] = ("s", "h", "d", "c")


def build_deck() -> List[str]:
    return [f"{rank}{suit}" for rank in RANKS for suit in SUITS]


@dataclass
class Deck:
    cards: List[str] = field(default_factory=build_deck)

    def shuffle(self, rng: random.Random | None = None) -> None:
        if rng is None:
            rng = random.Random()
        rng.shuffle(self.cards)

    def deal(self, count: int) -> List[str]:
        if count < 0:
            raise ValueError("count must be non-negative")
        if count > len(self.cards):
            raise ValueError("not enough cards to deal")
        dealt = self.cards[:count]
        del self.cards[:count]
        return dealt
