from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

from ..member2.bucketing import compute_infoset_id
from ..poker.engine import Engine
from ..schemas import Action, ActionType


class MCCFRTrainer:
    def __init__(self) -> None:
        self.regret_sum: Dict[tuple[str, str], float] = {}
        self.strategy_sum: Dict[tuple[str, str], float] = {}
        self.iteration_count: int = 0

    def get_strategy(
        self, infoset: str, legal_actions: List[ActionType]
    ) -> Dict[ActionType, float]:
        regrets = [self.regret_sum.get((infoset, a.value), 0.0) for a in legal_actions]
        positives = [max(r, 0.0) for r in regrets]
        normalizer = sum(positives)
        if normalizer > 0:
            strategy = [r / normalizer for r in positives]
        else:
            strategy = [1.0 / len(legal_actions)] * len(legal_actions)
        return dict(zip(legal_actions, strategy))

    def _sample_action(self, engine: Engine, action: ActionType) -> Action:
        if action != ActionType.RAISE:
            return Action(action=action)

        current_player = engine.betting.current_player
        if current_player is None:
            return Action(action=ActionType.CHECK)
        min_raise_to = engine.betting.min_raise_to()
        max_raise_to = engine.betting.max_raise_to(current_player)
        if max_raise_to < min_raise_to:
            amount = max_raise_to
        else:
            amount = random.randint(min_raise_to, max_raise_to)
        return Action(action=ActionType.RAISE, amount=amount)

    def mccfr(self, engine: Engine, player: str, sampling_prob: float = 1.0) -> float:
        if engine.is_terminal():
            return float(engine.utility(player))

        legal_actions = list(engine.betting.legal_actions())
        if not legal_actions:
            return 0.0

        infoset = str(
            compute_infoset_id(
                player_id=player,
                hole_cards=engine.hole_cards.get(player, []),
                board=engine.board,
                street=engine.street.value,
                action_history=engine.betting.action_history,
                pot=engine.betting.pot,
                player_stack=engine.betting.stacks.get(player, 0),
                big_blind=engine.betting.big_blind,
            )
        )

        strategy = self.get_strategy(infoset, legal_actions)
        acting_player = engine.betting.current_player
        if acting_player is None:
            return 0.0

        actions, probs = zip(*strategy.items())
        chosen_action = random.choices(actions, probs)[0]
        chosen_prob = strategy[chosen_action]

        next_state = engine.clone()
        try:
            next_state.step(self._sample_action(next_state, chosen_action), acting_player)
        except ValueError:
            return 0.0

        util = self.mccfr(next_state, player, sampling_prob * chosen_prob)

        for action in legal_actions:
            action_util = util if action == chosen_action else 0.0
            regret = action_util - util * chosen_prob
            weight = sampling_prob / max(chosen_prob, 1e-9)
            key = (infoset, action.value)
            self.regret_sum[key] = self.regret_sum.get(key, 0.0) + weight * regret

        for action, prob in strategy.items():
            key = (infoset, action.value)
            self.strategy_sum[key] = self.strategy_sum.get(key, 0.0) + prob

        return util

    def train_from_dataset(self, dataset_path: Path, log_interval: int = 100) -> None:
        with dataset_path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f, start=1):
                hand = json.loads(line)
                engine = Engine()
                engine.load_hand(hand)
                for player in engine.players:
                    self.mccfr(engine.clone(), player)
                self.iteration_count += 1
                if i % log_interval == 0:
                    print(
                        f"[Hand {i}] Regrets={len(self.regret_sum)} Strategy={len(self.strategy_sum)}"
                    )

    def export_strategy(self, filename: Path) -> None:
        strategy_table: Dict[str, Dict[str, float]] = {}
        for (infoset, action), total in self.strategy_sum.items():
            if infoset not in strategy_table:
                strategy_table[infoset] = {}
            strategy_table[infoset][action] = total / max(1, self.iteration_count)
        with filename.open("w", encoding="utf-8") as f:
            json.dump(strategy_table, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MCCFR strategy from dataset.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).with_name("training_dataset.jsonl"),
        help="Path to JSONL training dataset",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("strategy.json"),
        help="Path to write exported strategy JSON",
    )
    parser.add_argument("--log-interval", type=int, default=50)
    args = parser.parse_args()

    trainer = MCCFRTrainer()
    trainer.train_from_dataset(args.dataset, log_interval=args.log_interval)
    trainer.export_strategy(args.output)
    print(f"Training complete. Strategy exported to {args.output}")


if __name__ == "__main__":
    main()
