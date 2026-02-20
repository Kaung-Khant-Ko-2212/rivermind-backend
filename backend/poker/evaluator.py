from __future__ import annotations

from typing import List, Tuple

try:
    from treys import Card, Evaluator
except ImportError as exc:  # pragma: no cover - exercised only when treys missing
    raise ImportError(
        "treys is required for hand evaluation. Install with: pip install treys"
    ) from exc

_EVALUATOR = Evaluator()


def to_treys(cards: List[str]) -> List[int]:
    return [Card.new(card) for card in cards]


def evaluate_hand(hole_cards: List[str], board: List[str]) -> int:
    return _EVALUATOR.evaluate(to_treys(board), to_treys(hole_cards))


def compare_hands(
    hole_one: List[str], hole_two: List[str], board: List[str]
) -> Tuple[str, int, int]:
    board_cards = to_treys(board)
    score_one = _EVALUATOR.evaluate(board_cards, to_treys(hole_one))
    score_two = _EVALUATOR.evaluate(board_cards, to_treys(hole_two))

    if score_one < score_two:
        winner = "p1"
    elif score_two < score_one:
        winner = "p2"
    else:
        winner = "tie"

    return winner, score_one, score_two


def hand_category(score: int) -> str:
    rank_class = _EVALUATOR.get_rank_class(score)
    return _EVALUATOR.class_to_string(rank_class)
