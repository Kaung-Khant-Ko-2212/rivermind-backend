import random

from backend.ai.policy import get_ai_action
from backend.schemas import ActionType


def test_ai_action_is_legal() -> None:
    state = {
        "legal_actions": ["check", "call", "fold", "raise"],
        "min_raise_to": 20,
        "max_raise_to": 100,
    }
    rng = random.Random(1)
    action = get_ai_action(state, rng=rng)
    assert action.action in {ActionType.CHECK, ActionType.CALL, ActionType.FOLD, ActionType.RAISE}


def test_ai_raise_amount_valid() -> None:
    state = {
        "legal_actions": ["raise"],
        "min_raise_to": 30,
        "max_raise_to": 60,
    }
    rng = random.Random(2)
    action = get_ai_action(state, rng=rng)
    assert action.action == ActionType.RAISE
    assert action.amount is not None
    assert 30 <= action.amount <= 60
