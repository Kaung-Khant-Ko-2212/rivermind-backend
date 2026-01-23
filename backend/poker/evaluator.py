from __future__ import annotations

from typing import List, Tuple

try:
    from treys import Card, Evaluator
except ImportError as exc:  # pragma: no cover - exercised only when treys missing
    raise ImportError(
        "treys is required for hand evaluation. Install with: pip install treys"
    ) from exc


def to_treys(cards: List[str]) -> List[int]:
    return [Card.new(card) for card in cards]


def evaluate_hand(hole_cards: List[str], board: List[str]) -> int:
    evaluator = Evaluator()
    return evaluator.evaluate(to_treys(board), to_treys(hole_cards))


def compare_hands(
    hole_one: List[str], hole_two: List[str], board: List[str]
) -> Tuple[str, int, int]:
    evaluator = Evaluator()
    board_cards = to_treys(board)
    score_one = evaluator.evaluate(board_cards, to_treys(hole_one))
    score_two = evaluator.evaluate(board_cards, to_treys(hole_two))

    if score_one < score_two:
        winner = "p1"
    elif score_two < score_one:
        winner = "p2"
    else:
        winner = "tie"

    return winner, score_one, score_two


def hand_category(score: int) -> str:
    evaluator = Evaluator()
    rank_class = evaluator.get_rank_class(score)
    return evaluator.class_to_string(rank_class)
