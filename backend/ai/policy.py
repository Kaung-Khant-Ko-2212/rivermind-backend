from __future__ import annotations

import os
import random
from typing import Any, Iterable, Mapping, Optional

from ..schemas import Action, ActionType


_AI_MODE = os.getenv("AI_MODE", "random").strip().lower()
_AI_SEED_RAW = os.getenv("AI_SEED")
_AI_RNG = random.Random(int(_AI_SEED_RAW)) if _AI_SEED_RAW else random.Random()


def _normalize_actions(actions: Iterable[Any]) -> list[ActionType]:
    normalized: list[ActionType] = []
    for action in actions:
        if isinstance(action, ActionType):
            normalized.append(action)
        else:
            normalized.append(ActionType(str(action)))
    return normalized


def get_ai_action(
    state: Mapping[str, Any], rng: Optional[random.Random] = None
) -> Action:
    rng = rng or _AI_RNG
    legal_actions = _normalize_actions(state.get("legal_actions", []))
    if not legal_actions:
        raise ValueError("No legal actions available for AI")

    if _AI_MODE == "passive":
        for option in (ActionType.CHECK, ActionType.CALL, ActionType.FOLD, ActionType.RAISE):
            if option in legal_actions:
                if option == ActionType.RAISE:
                    min_raise_to = int(state.get("min_raise_to", 0))
                    return Action(action=ActionType.RAISE, amount=min_raise_to)
                return Action(action=option)

    chosen = rng.choice(legal_actions)

    if chosen != ActionType.RAISE:
        return Action(action=chosen)

    min_raise_to = int(state.get("min_raise_to", 0))
    max_raise_to = int(state.get("max_raise_to", min_raise_to))
    if max_raise_to < min_raise_to:
        raise_amount = max_raise_to
    else:
        raise_amount = rng.randint(min_raise_to, max_raise_to)

    return Action(action=ActionType.RAISE, amount=raise_amount)
