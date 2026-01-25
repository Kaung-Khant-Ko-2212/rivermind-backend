from __future__ import annotations

import argparse
import json

from .cards import parse_cards
from .dataset import export_dataset
from .ehs import EHSEngine
from .game import run_cli_game


def _cmd_ehs(args: argparse.Namespace) -> None:
    hero = parse_cards(args.hero)
    board = parse_cards(args.board)
    engine = EHSEngine(
        rollouts=args.rollouts,
        n_opponents=args.opponents,
        seed=args.seed,
        cache_path=args.cache,
        n_jobs=args.jobs,
    )
    result = engine.compute(hero, board)
    print(json.dumps(result.__dict__, indent=2))


def _cmd_dataset(args: argparse.Namespace) -> None:
    export_dataset(
        input_path=args.input,
        output_path=args.output,
        rollouts=args.rollouts,
        n_opponents=args.opponents,
        seed=args.seed,
        cache_path=args.cache,
        n_jobs=args.jobs,
    )


def _cmd_play(args: argparse.Namespace) -> None:
    run_cli_game(
        num_players=args.players,
        starting_stack=args.stack,
        small_blind=args.small_blind,
        big_blind=args.big_blind,
        ai_style=args.ai_style,
        rng_seed=args.seed,
        log_path=args.log,
        show_ehs=args.show_ehs,
        ehs_rollouts=args.ehs_rollouts,
        ai_rollouts=args.ai_rollouts,
        max_hands=args.max_hands,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Texas Hold'em EHS tools")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ehs = sub.add_parser("ehs", help="Estimate EHS for a hand state")
    p_ehs.add_argument("--hero", required=True, help="Hero cards, e.g. 'As Kd'")
    p_ehs.add_argument("--board", default="", help="Board cards, e.g. '2c 7d Jh'")
    p_ehs.add_argument("--rollouts", type=int, default=1000)
    p_ehs.add_argument("--opponents", type=int, default=1)
    p_ehs.add_argument("--seed", type=int, default=None)
    p_ehs.add_argument("--cache", type=str, default=None)
    p_ehs.add_argument("--jobs", type=int, default=1)
    p_ehs.set_defaults(func=_cmd_ehs)

    p_data = sub.add_parser("dataset", help="Export EHS dataset from JSON/JSONL input")
    p_data.add_argument("--input", required=True, help="Input .json or .jsonl with hero/board")
    p_data.add_argument("--output", required=True, help="Output CSV path")
    p_data.add_argument("--rollouts", type=int, default=1000)
    p_data.add_argument("--opponents", type=int, default=1)
    p_data.add_argument("--seed", type=int, default=None)
    p_data.add_argument("--cache", type=str, default=None)
    p_data.add_argument("--jobs", type=int, default=1)
    p_data.set_defaults(func=_cmd_dataset)

    p_play = sub.add_parser("play", help="Play a CLI Texas Hold'em game")
    p_play.add_argument("--players", type=int, default=4, help="Total players including you")
    p_play.add_argument("--stack", type=int, default=1000, help="Starting stack size")
    p_play.add_argument("--small-blind", type=int, default=5)
    p_play.add_argument("--big-blind", type=int, default=10)
    p_play.add_argument("--ai-style", type=str, default="balanced", choices=["balanced", "tight", "aggressive", "random"])
    p_play.add_argument("--seed", type=int, default=None)
    p_play.add_argument("--log", type=str, default=None, help="Optional hand history log path")
    p_play.add_argument("--show-ehs", action="store_true", help="Show EHS estimate on your turns")
    p_play.add_argument("--ehs-rollouts", type=int, default=200, help="Rollouts for displayed EHS")
    p_play.add_argument("--ai-rollouts", type=int, default=200, help="Rollouts for AI decisions")
    p_play.add_argument("--max-hands", type=int, default=None, help="Stop after N hands")
    p_play.set_defaults(func=_cmd_play)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
