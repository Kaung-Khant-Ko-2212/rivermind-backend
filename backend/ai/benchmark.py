from __future__ import annotations

import argparse
import json
from contextlib import contextmanager
from dataclasses import dataclass
import random
from typing import Dict, Iterable, Iterator, Optional

from ..poker.engine import Engine
from ..schemas import Street
from . import policy as policy_module


SUPPORTED_POLICIES = {"current", "strategy", "mccfr", "member3", "random", "passive"}


@dataclass
class HandResultSummary:
    chip_delta: Dict[str, int]
    winner_ids: list[str]
    tie: bool
    showdown: bool
    pot: int
    action_count: int


@contextmanager
def _temporary_ai_mode(mode: Optional[str]) -> Iterator[None]:
    if mode is None:
        yield
        return
    previous = getattr(policy_module, "_AI_MODE", "random")
    policy_module._AI_MODE = mode  # type: ignore[attr-defined]
    try:
        yield
    finally:
        policy_module._AI_MODE = previous  # type: ignore[attr-defined]


def _pick_action_for_policy(mode: str, state: dict[str, object], rng: random.Random):
    if mode == "current":
        return policy_module.get_ai_action(state, rng=rng)
    with _temporary_ai_mode(mode):
        return policy_module.get_ai_action(state, rng=rng)


def _find_next_eligible_actor(engine: Engine) -> Optional[str]:
    betting = engine.betting
    current = betting.current_player

    if (
        current
        and current in betting.pending_players
        and current not in betting.folded_players
        and current not in betting.all_in_players
    ):
        return current

    if current:
        try:
            candidate = betting._next_player(current)  # type: ignore[attr-defined]
        except Exception:
            candidate = None
        if candidate:
            return candidate

    for player in engine.players:
        if (
            player in betting.pending_players
            and player not in betting.folded_players
            and player not in betting.all_in_players
        ):
            return player
    return None


def _advance_without_actor(engine: Engine) -> bool:
    next_player = _find_next_eligible_actor(engine)
    if next_player:
        if engine.betting.current_player != next_player:
            engine.betting.current_player = next_player
            return True
        return False

    street = engine.street
    if street == Street.PREFLOP:
        engine.deal_flop()
        engine.betting.start_new_round(first_to_act=engine._first_to_act_postflop())  # type: ignore[attr-defined]
        return True
    if street == Street.FLOP:
        engine.deal_turn()
        engine.betting.start_new_round(first_to_act=engine._first_to_act_postflop())  # type: ignore[attr-defined]
        return True
    if street == Street.TURN:
        engine.deal_river()
        engine.betting.start_new_round(first_to_act=engine._first_to_act_postflop())  # type: ignore[attr-defined]
        return True
    if street == Street.RIVER:
        engine.resolve_showdown()
        return True
    return False


def _extract_hand_end(events: Iterable[object]) -> tuple[list[str], bool, bool, int]:
    winner_ids: list[str] = []
    tie = False
    showdown = False
    pot = 0
    found = False

    for event in events:
        event_name = getattr(event, "event", None)
        normalized_name = getattr(event_name, "value", str(event_name))
        if normalized_name != "HAND_END":
            continue
        found = True
        data = getattr(event, "data", None) or {}
        raw_winner = data.get("winner")
        if isinstance(raw_winner, list):
            winner_ids = [w for w in raw_winner if isinstance(w, str)]
            tie = len(winner_ids) > 1
        elif isinstance(raw_winner, str):
            winner_ids = [raw_winner]
            tie = False
        showdown = data.get("hand_category") is not None
        try:
            pot = int(data.get("pot", 0) or 0)
        except (TypeError, ValueError):
            pot = 0
        break

    if not found:
        raise RuntimeError("HAND_END event not found after simulated hand")

    return winner_ids, tie, showdown, pot


def play_hand(
    *,
    hand_seed: int,
    policy_by_player: Dict[str, str],
    stack: int,
    small_blind: int,
    big_blind: int,
    button_index: int,
    max_actions: int,
) -> HandResultSummary:
    engine = Engine(players=("p1", "p2"))
    engine.betting.starting_stack = stack
    engine.betting.small_blind = small_blind
    engine.betting.big_blind = big_blind
    engine.button_index = button_index
    engine.new_hand(seed=hand_seed, rotate_button=False)

    rng_by_player = {
        "p1": random.Random(hand_seed * 1009 + 11),
        "p2": random.Random(hand_seed * 1009 + 29),
    }

    action_count = 0
    while not engine.betting.hand_over:
        if _advance_without_actor(engine):
            if engine.betting.hand_over:
                break
            continue

        actor = engine.betting.current_player
        if not actor:
            break

        if actor not in policy_by_player:
            raise ValueError(f"No policy configured for actor {actor}")

        state = engine.to_ai_state()
        action = _pick_action_for_policy(policy_by_player[actor], state, rng_by_player[actor])
        engine.step(action, player_id=actor)
        action_count += 1

        if action_count > max_actions:
            raise RuntimeError(
                f"Exceeded max_actions={max_actions} in hand seed={hand_seed}; likely loop bug"
            )

    events = engine.drain_events()
    winner_ids, tie, showdown, pot = _extract_hand_end(events)
    chip_delta = {player: engine.get_chip_change(player) for player in ("p1", "p2")}
    return HandResultSummary(
        chip_delta=chip_delta,
        winner_ids=winner_ids,
        tie=tie,
        showdown=showdown,
        pot=pot,
        action_count=action_count,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark heads-up poker policies over many seeded hands."
    )
    parser.add_argument("--hands", type=int, default=1000, help="Number of hands to simulate")
    parser.add_argument("--seed", type=int, default=12345, help="Base RNG seed")
    parser.add_argument(
        "--policy-a",
        type=str,
        default="current",
        choices=sorted(SUPPORTED_POLICIES),
        help="Policy label for competitor A",
    )
    parser.add_argument(
        "--policy-b",
        type=str,
        default="random",
        choices=sorted(SUPPORTED_POLICIES),
        help="Policy label for competitor B",
    )
    parser.add_argument("--stack", type=int, default=1000, help="Starting stack per hand")
    parser.add_argument("--small-blind", type=int, default=5)
    parser.add_argument("--big-blind", type=int, default=10)
    parser.add_argument(
        "--max-actions",
        type=int,
        default=200,
        help="Safety cap for actions per hand",
    )
    parser.add_argument(
        "--no-seat-swap",
        action="store_true",
        help="Do not alternate which policy sits in p1/p2 across hands",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=0,
        help="Print progress every N hands (0 disables)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON summary",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.hands <= 0:
        raise SystemExit("--hands must be > 0")
    if args.stack <= 0:
        raise SystemExit("--stack must be > 0")
    if args.small_blind <= 0 or args.big_blind <= 0:
        raise SystemExit("Blinds must be > 0")
    if args.big_blind < args.small_blind:
        raise SystemExit("--big-blind must be >= --small-blind")

    seat_swap = not args.no_seat_swap

    policy_stats = {
        "A": {"label": args.policy_a, "chip_delta": 0, "wins": 0, "ties": 0},
        "B": {"label": args.policy_b, "chip_delta": 0, "wins": 0, "ties": 0},
    }
    seat_stats = {
        "p1": {"chip_delta": 0, "wins": 0},
        "p2": {"chip_delta": 0, "wins": 0},
    }
    meta_stats = {
        "showdown_hands": 0,
        "fold_end_hands": 0,
        "ties": 0,
        "total_pot": 0,
        "total_actions": 0,
    }

    for hand_index in range(args.hands):
        hand_seed = args.seed + hand_index

        if seat_swap and hand_index % 2 == 1:
            seat_to_competitor = {"p1": "B", "p2": "A"}
        else:
            seat_to_competitor = {"p1": "A", "p2": "B"}

        policy_by_player = {
            seat: policy_stats[competitor]["label"]
            for seat, competitor in seat_to_competitor.items()
        }

        result = play_hand(
            hand_seed=hand_seed,
            policy_by_player=policy_by_player,
            stack=args.stack,
            small_blind=args.small_blind,
            big_blind=args.big_blind,
            button_index=hand_index % 2,
            max_actions=args.max_actions,
        )

        for seat in ("p1", "p2"):
            delta = result.chip_delta.get(seat, 0)
            seat_stats[seat]["chip_delta"] += delta
            competitor = seat_to_competitor[seat]
            policy_stats[competitor]["chip_delta"] += delta

        if result.tie:
            meta_stats["ties"] += 1
            policy_stats["A"]["ties"] += 1
            policy_stats["B"]["ties"] += 1
        else:
            for winner in result.winner_ids:
                if winner in seat_stats:
                    seat_stats[winner]["wins"] += 1
                    competitor = seat_to_competitor[winner]
                    policy_stats[competitor]["wins"] += 1

        if result.showdown:
            meta_stats["showdown_hands"] += 1
        else:
            meta_stats["fold_end_hands"] += 1
        meta_stats["total_pot"] += result.pot
        meta_stats["total_actions"] += result.action_count

        if args.progress_every and (hand_index + 1) % args.progress_every == 0:
            print(
                f"[{hand_index + 1}/{args.hands}] "
                f"A({args.policy_a})={policy_stats['A']['chip_delta']} "
                f"B({args.policy_b})={policy_stats['B']['chip_delta']}"
            )

    hands = args.hands
    summary = {
        "config": {
            "hands": hands,
            "seed": args.seed,
            "policy_a": args.policy_a,
            "policy_b": args.policy_b,
            "seat_swap": seat_swap,
            "stack": args.stack,
            "small_blind": args.small_blind,
            "big_blind": args.big_blind,
        },
        "policy_results": {
            "A": {
                **policy_stats["A"],
                "avg_chip_delta_per_hand": round(policy_stats["A"]["chip_delta"] / hands, 4),
                "win_rate": round(policy_stats["A"]["wins"] / hands, 4),
            },
            "B": {
                **policy_stats["B"],
                "avg_chip_delta_per_hand": round(policy_stats["B"]["chip_delta"] / hands, 4),
                "win_rate": round(policy_stats["B"]["wins"] / hands, 4),
            },
        },
        "seat_results": {
            "p1": {
                **seat_stats["p1"],
                "avg_chip_delta_per_hand": round(seat_stats["p1"]["chip_delta"] / hands, 4),
            },
            "p2": {
                **seat_stats["p2"],
                "avg_chip_delta_per_hand": round(seat_stats["p2"]["chip_delta"] / hands, 4),
            },
        },
        "meta": {
            **meta_stats,
            "showdown_rate": round(meta_stats["showdown_hands"] / hands, 4),
            "avg_pot": round(meta_stats["total_pot"] / hands, 4),
            "avg_actions_per_hand": round(meta_stats["total_actions"] / hands, 4),
        },
    }

    if args.json:
        print(json.dumps(summary, indent=2))
        return

    print("Benchmark Summary")
    print(
        f"- Hands: {hands} | Seed: {args.seed} | Seat swap: {'on' if seat_swap else 'off'} | "
        f"Blinds: {args.small_blind}/{args.big_blind}"
    )
    print(
        f"- A ({summary['policy_results']['A']['label']}): "
        f"chip_delta={summary['policy_results']['A']['chip_delta']} "
        f"avg/hand={summary['policy_results']['A']['avg_chip_delta_per_hand']} "
        f"wins={summary['policy_results']['A']['wins']} "
        f"ties={summary['policy_results']['A']['ties']}"
    )
    print(
        f"- B ({summary['policy_results']['B']['label']}): "
        f"chip_delta={summary['policy_results']['B']['chip_delta']} "
        f"avg/hand={summary['policy_results']['B']['avg_chip_delta_per_hand']} "
        f"wins={summary['policy_results']['B']['wins']} "
        f"ties={summary['policy_results']['B']['ties']}"
    )
    print(
        f"- Meta: showdown_rate={summary['meta']['showdown_rate']} "
        f"avg_pot={summary['meta']['avg_pot']} "
        f"avg_actions={summary['meta']['avg_actions_per_hand']}"
    )


if __name__ == "__main__":
    main()
