from __future__ import annotations

import argparse

from .engine import Engine


def main() -> None:
    parser = argparse.ArgumentParser(description="Deal a hand and evaluate winner.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    args = parser.parse_args()

    engine = Engine()
    engine.new_hand(seed=args.seed)
    engine.deal_flop()
    engine.deal_turn()
    engine.deal_river()
    winner, score_one, score_two = engine.evaluate_showdown()

    print(f"P1: {engine.hole_cards['p1']}  P2: {engine.hole_cards['p2']}")
    print(f"Board: {engine.board}")
    print(f"Winner: {winner} (scores: {score_one} vs {score_two})")


if __name__ == "__main__":
    main()
