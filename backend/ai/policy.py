from __future__ import annotations

import json
import os
from pathlib import Path
import random
from typing import Any, Iterable, Mapping, Optional

from ..env_loader import load_env_file
from ..schemas import Action, ActionType
from ..member2.bucketing import compute_infoset_id


load_env_file()

_AI_MODE = os.getenv("AI_MODE", "random").strip().lower()
_AI_SEED_RAW = os.getenv("AI_SEED")
_AI_RNG = random.Random(int(_AI_SEED_RAW)) if _AI_SEED_RAW else random.Random()
_STRATEGY_MODES = {"strategy", "mccfr", "member3"}
_DEFAULT_STRATEGY_PATH = Path(__file__).with_name("strategy.json")
_STRATEGY_PATH = Path(os.getenv("AI_STRATEGY_PATH", str(_DEFAULT_STRATEGY_PATH))).expanduser()


def _load_strategy(path: Path) -> dict[str, dict[str, float]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    if not isinstance(payload, dict):
        return {}
    strategy: dict[str, dict[str, float]] = {}
    for infoset, action_probs in payload.items():
        if not isinstance(infoset, str) or not isinstance(action_probs, dict):
            continue
        normalized_row: dict[str, float] = {}
        for action, prob in action_probs.items():
            try:
                normalized_row[str(action)] = float(prob)
            except (TypeError, ValueError):
                continue
        if normalized_row:
            strategy[infoset] = normalized_row
    return strategy


_STRATEGY = _load_strategy(_STRATEGY_PATH)


def _normalize_actions(actions: Iterable[Any]) -> list[ActionType]:
    normalized: list[ActionType] = []
    for action in actions:
        if isinstance(action, ActionType):
            normalized.append(action)
        else:
            normalized.append(ActionType(str(action)))
    return normalized


def _build_infoset_candidates(state: Mapping[str, Any]) -> list[str]:
    current_player = str(state.get("current_player") or "")
    if not current_player:
        return []

    hole_cards = list(state.get("hand") or [])
    board = list(state.get("community_cards") or state.get("board") or [])
    street = str(state.get("street", "preflop"))
    action_history = list(state.get("action_history") or state.get("history") or [])
    pot = int(state.get("pot", 0) or 0)
    stacks = state.get("stacks", {})
    player_stack = int((stacks or {}).get(current_player, 0) or 0)

    big_blind = int(state.get("big_blind", 0) or 0)
    if big_blind <= 0:
        bets = state.get("bets", {})
        if isinstance(bets, Mapping) and bets:
            try:
                big_blind = max(int(v) for v in bets.values())
            except (TypeError, ValueError):
                big_blind = 10
        else:
            big_blind = 10
    if big_blind <= 0:
        big_blind = 10

    candidates: list[str] = []

    detailed_infoset = compute_infoset_id(
        player_id=current_player,
        hole_cards=hole_cards,
        board=board,
        street=street,
        action_history=action_history,
        pot=pot,
        player_stack=player_stack,
        big_blind=big_blind,
    )
    candidates.append(detailed_infoset)

    abstract_infoset = compute_infoset_id(
        player_id=current_player,
        hole_cards=[],
        board=board,
        street=street,
        action_history=action_history,
        pot=pot,
        player_stack=player_stack,
        big_blind=big_blind,
    )
    if abstract_infoset != detailed_infoset:
        candidates.append(abstract_infoset)

    return candidates


def _strategy_pick(
    state: Mapping[str, Any],
    legal_actions: list[ActionType],
    rng: random.Random,
) -> Optional[ActionType]:
    if not _STRATEGY:
        return None

    legal_set = set(legal_actions)
    infosets = _build_infoset_candidates(state)

    for infoset in infosets:
        row = _STRATEGY.get(infoset)
        if not row:
            continue

        choices: list[ActionType] = []
        probs: list[float] = []
        for action_name, prob in row.items():
            try:
                action = ActionType(action_name)
            except ValueError:
                continue
            if action not in legal_set:
                continue
            if prob <= 0:
                continue
            choices.append(action)
            probs.append(prob)

        if choices:
            return rng.choices(choices, weights=probs, k=1)[0]

    return None


def _sample_raise_amount(state: Mapping[str, Any], rng: random.Random) -> int:
    min_raise_to = int(state.get("min_raise_to", 0))
    max_raise_to = int(state.get("max_raise_to", min_raise_to))
    if max_raise_to < min_raise_to:
        return max_raise_to
    return rng.randint(min_raise_to, max_raise_to)


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
                    return Action(action=ActionType.RAISE, amount=int(state.get("min_raise_to", 0)))
                return Action(action=option)

    chosen: Optional[ActionType] = None
    if _AI_MODE in _STRATEGY_MODES:
        chosen = _strategy_pick(state, legal_actions, rng)

    if chosen is None:
        chosen = rng.choice(legal_actions)

    if chosen != ActionType.RAISE:
        return Action(action=chosen)

    return Action(action=ActionType.RAISE, amount=_sample_raise_amount(state, rng))
