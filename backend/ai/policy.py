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
    bets = state.get("bets", {})
    to_call = int(state.get("to_call", 0) or 0)
    button_player_raw = state.get("button_player")
    button_player = str(button_player_raw) if isinstance(button_player_raw, str) else None
    active_players_raw = state.get("active_players") or state.get("players") or []
    active_players = (
        [str(player) for player in active_players_raw if isinstance(player, str)]
        if isinstance(active_players_raw, (list, tuple))
        else None
    )
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
        bets=bets if isinstance(bets, Mapping) else None,
        to_call=to_call,
        button_player=button_player,
        active_players=active_players,
        schema_version="v2",
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
        bets=bets if isinstance(bets, Mapping) else None,
        to_call=to_call,
        button_player=button_player,
        active_players=active_players,
        schema_version="v2",
    )
    if abstract_infoset != detailed_infoset:
        candidates.append(abstract_infoset)

    legacy_detailed = compute_infoset_id(
        player_id=current_player,
        hole_cards=hole_cards,
        board=board,
        street=street,
        action_history=action_history,
        pot=pot,
        player_stack=player_stack,
        big_blind=big_blind,
        schema_version="v1",
    )
    if legacy_detailed not in candidates:
        candidates.append(legacy_detailed)

    legacy_abstract = compute_infoset_id(
        player_id=current_player,
        hole_cards=[],
        board=board,
        street=street,
        action_history=action_history,
        pot=pot,
        player_stack=player_stack,
        big_blind=big_blind,
        schema_version="v1",
    )
    if legacy_abstract not in candidates:
        candidates.append(legacy_abstract)

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


def _opponent_profile(state: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    profile = state.get("opponent_profile")
    return profile if isinstance(profile, Mapping) else None


def _profile_int(profile: Optional[Mapping[str, Any]], key: str) -> int:
    if not profile:
        return 0
    try:
        return int(profile.get(key, 0) or 0)
    except (TypeError, ValueError):
        return 0


def _profile_float(profile: Optional[Mapping[str, Any]], key: str) -> float:
    if not profile:
        return 0.0
    try:
        return float(profile.get(key, 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _apply_exploitative_adjustment(
    chosen: ActionType,
    state: Mapping[str, Any],
    legal_actions: list[ActionType],
    rng: random.Random,
) -> ActionType:
    profile = _opponent_profile(state)
    if not profile:
        return chosen

    samples = _profile_int(profile, "samples")
    facing_bet_samples = _profile_int(profile, "facing_bet_samples")
    if samples < 10:
        return chosen

    legal_set = set(legal_actions)
    pot = max(0, int(state.get("pot", 0) or 0))
    to_call = max(0, int(state.get("to_call", 0) or 0))
    cheap_continue = to_call > 0 and to_call <= max(1, pot // 4)

    fold_vs_bet_rate = _profile_float(profile, "fold_vs_bet_rate")
    continue_vs_bet_rate = _profile_float(profile, "continue_vs_bet_rate")
    raise_rate = _profile_float(profile, "raise_rate")
    vpip_rate = _profile_float(profile, "vpip_rate")

    # Exploit likely over-folders by taking more aggressive lines in low-pressure spots.
    if (
        ActionType.RAISE in legal_set
        and chosen in {ActionType.CHECK, ActionType.CALL}
        and facing_bet_samples >= 8
        and fold_vs_bet_rate >= 0.58
        and (to_call == 0 or cheap_continue)
    ):
        boost = min(0.35, 0.10 + (fold_vs_bet_rate - 0.58) * 0.8)
        if rng.random() < boost:
            return ActionType.RAISE

    # Versus sticky players, trim some raise frequency in favor of lower-variance continues.
    if (
        chosen == ActionType.RAISE
        and facing_bet_samples >= 10
        and continue_vs_bet_rate >= 0.72
        and fold_vs_bet_rate <= 0.28
    ):
        trim = min(0.40, 0.15 + (continue_vs_bet_rate - 0.72) * 0.6)
        if rng.random() < trim:
            if to_call == 0 and ActionType.CHECK in legal_set:
                return ActionType.CHECK
            if ActionType.CALL in legal_set:
                return ActionType.CALL
            if ActionType.CHECK in legal_set:
                return ActionType.CHECK

    # Slightly widen defense vs aggressive opponents when the price is cheap.
    if (
        chosen == ActionType.FOLD
        and ActionType.CALL in legal_set
        and cheap_continue
        and raise_rate >= 0.28
        and samples >= 15
    ):
        defend_prob = min(0.20, 0.05 + (raise_rate - 0.28) * 0.4)
        if vpip_rate >= 0.30 and rng.random() < defend_prob:
            return ActionType.CALL

    return chosen


def _sample_raise_amount(state: Mapping[str, Any], rng: random.Random) -> int:
    min_raise_to = int(state.get("min_raise_to", 0))
    max_raise_to = int(state.get("max_raise_to", min_raise_to))
    if max_raise_to < min_raise_to:
        return max_raise_to
    if max_raise_to == min_raise_to:
        return min_raise_to

    current_player = str(state.get("current_player") or "")
    bets = state.get("bets", {})
    pot = max(0, int(state.get("pot", 0) or 0))
    to_call = max(0, int(state.get("to_call", 0) or 0))

    current_contribution = 0
    if isinstance(bets, Mapping) and current_player:
        try:
            current_contribution = max(0, int((bets or {}).get(current_player, 0) or 0))
        except (TypeError, ValueError):
            current_contribution = 0

    def clamp_target(value: int) -> int:
        return max(min_raise_to, min(max_raise_to, int(value)))

    # Work in "raise-to" totals. Pot-fraction buckets reduce erratic all-range sampling.
    candidates: list[tuple[str, int]] = []
    seen_targets: set[int] = set()

    def add_candidate(label: str, target: int) -> None:
        clamped = clamp_target(target)
        if clamped in seen_targets:
            return
        seen_targets.add(clamped)
        candidates.append((label, clamped))

    add_candidate("min", min_raise_to)

    if pot > 0:
        add_candidate("half_pot", current_contribution + to_call + max(1, pot // 2))
        add_candidate("three_quarter_pot", current_contribution + to_call + max(1, (pot * 3) // 4))
        add_candidate("pot", current_contribution + to_call + pot)
        add_candidate("one_half_pot", current_contribution + to_call + max(1, (pot * 3) // 2))
    else:
        midpoint = min_raise_to + ((max_raise_to - min_raise_to) // 2)
        add_candidate("mid", midpoint)

    add_candidate("all_in", max_raise_to)

    if not candidates:
        return min_raise_to

    weights_by_label = {
        "min": 1.5,
        "half_pot": 2.25,
        "three_quarter_pot": 2.0,
        "pot": 2.25,
        "one_half_pot": 1.25,
        "mid": 2.0,
        "all_in": 0.75,
    }
    profile = _opponent_profile(state)
    if profile and _profile_int(profile, "facing_bet_samples") >= 8:
        fold_vs_bet_rate = _profile_float(profile, "fold_vs_bet_rate")
        continue_vs_bet_rate = _profile_float(profile, "continue_vs_bet_rate")
        if fold_vs_bet_rate >= 0.58:
            # Charge more against players who over-fold to aggression.
            for label, mult in {
                "min": 0.85,
                "half_pot": 0.95,
                "pot": 1.15,
                "one_half_pot": 1.25,
                "all_in": 1.05,
            }.items():
                weights_by_label[label] = weights_by_label.get(label, 1.0) * mult
        elif continue_vs_bet_rate >= 0.72 and fold_vs_bet_rate <= 0.28:
            # Against sticky profiles, bias down to smaller sizings.
            for label, mult in {
                "min": 1.25,
                "half_pot": 1.10,
                "pot": 0.85,
                "one_half_pot": 0.70,
                "all_in": 0.60,
            }.items():
                weights_by_label[label] = weights_by_label.get(label, 1.0) * mult
    weights = [weights_by_label.get(label, 1.0) for label, _ in candidates]
    chosen_index = rng.choices(range(len(candidates)), weights=weights, k=1)[0]
    return candidates[chosen_index][1]


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

    if _AI_MODE != "passive":
        chosen = _apply_exploitative_adjustment(chosen, state, legal_actions, rng)

    if chosen != ActionType.RAISE:
        return Action(action=chosen)

    return Action(action=ActionType.RAISE, amount=_sample_raise_amount(state, rng))
