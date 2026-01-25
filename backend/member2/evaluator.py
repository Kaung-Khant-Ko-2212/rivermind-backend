from __future__ import annotations

import itertools
from typing import Iterable, Tuple


def _rank_5cards(cards: Iterable[int]) -> Tuple[int, Tuple[int, ...]]:
    ranks = [(c % 13) + 2 for c in cards]  # 2..14
    suits = [c // 13 for c in cards]
    ranks_sorted = sorted(ranks, reverse=True)

    is_flush = len(set(suits)) == 1

    unique_ranks = sorted(set(ranks), reverse=True)
    is_straight = False
    straight_high = 0
    if len(unique_ranks) == 5:
        high = unique_ranks[0]
        low = unique_ranks[-1]
        if high - low == 4:
            is_straight = True
            straight_high = high
        elif unique_ranks == [14, 5, 4, 3, 2]:
            is_straight = True
            straight_high = 5

    if is_straight and is_flush:
        return 8, (straight_high,)

    counts = {}
    for r in ranks:
        counts[r] = counts.get(r, 0) + 1
    count_groups = sorted(((cnt, r) for r, cnt in counts.items()), reverse=True)

    if count_groups[0][0] == 4:
        quad_rank = count_groups[0][1]
        kicker = max(r for r in ranks if r != quad_rank)
        return 7, (quad_rank, kicker)

    if count_groups[0][0] == 3 and count_groups[1][0] == 2:
        trip_rank = count_groups[0][1]
        pair_rank = count_groups[1][1]
        return 6, (trip_rank, pair_rank)

    if is_flush:
        return 5, tuple(ranks_sorted)

    if is_straight:
        return 4, (straight_high,)

    if count_groups[0][0] == 3:
        trip_rank = count_groups[0][1]
        kickers = sorted((r for r in ranks if r != trip_rank), reverse=True)
        return 3, (trip_rank, *kickers)

    if count_groups[0][0] == 2 and count_groups[1][0] == 2:
        pair_high = max(count_groups[0][1], count_groups[1][1])
        pair_low = min(count_groups[0][1], count_groups[1][1])
        kicker = max(r for r in ranks if r != pair_high and r != pair_low)
        return 2, (pair_high, pair_low, kicker)

    if count_groups[0][0] == 2:
        pair_rank = count_groups[0][1]
        kickers = sorted((r for r in ranks if r != pair_rank), reverse=True)
        return 1, (pair_rank, *kickers)

    return 0, tuple(ranks_sorted)


def best_hand_rank(cards7: Iterable[int]) -> Tuple[int, Tuple[int, ...]]:
    """Return best 5-card hand rank tuple from 7 cards."""
    best = (-1, ())
    for combo in itertools.combinations(cards7, 5):
        rank = _rank_5cards(combo)
        if rank > best:
            best = rank
    return best


def compare_hands(cards7_a: Iterable[int], cards7_b: Iterable[int]) -> int:
    """Return 1 if A wins, -1 if B wins, 0 for tie."""
    rank_a = best_hand_rank(cards7_a)
    rank_b = best_hand_rank(cards7_b)
    if rank_a > rank_b:
        return 1
    if rank_b > rank_a:
        return -1
    return 0
