from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from ..member2.bucketing import compute_infoset_id
from ..poker.engine import Engine
from ..schemas import Action, ActionType


class MCCFRTrainer:
    def __init__(self, seed: Optional[int] = None) -> None:
        self.regret_sum: Dict[tuple[str, str], float] = {}
        self.strategy_sum: Dict[tuple[str, str], float] = {}
        self.iteration_count: int = 0
        self.rng = random.Random(seed)

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
            amount = self.rng.randint(min_raise_to, max_raise_to)
        return Action(action=ActionType.RAISE, amount=amount)

    def mccfr(self, engine: Engine, player: str, sampling_prob: float = 1.0) -> float:
        if engine.is_terminal():
            return float(engine.utility(player))

        legal_actions = list(engine.betting.legal_actions())
        if not legal_actions:
            return 0.0
        acting_player = engine.betting.current_player
        if acting_player is None:
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
                bets=dict(engine.betting.contributions),
                to_call=engine.betting.to_call(acting_player) if acting_player else None,
                button_player=engine.button_player,
                active_players=list(engine.betting.players),
                schema_version="v2",
            )
        )

        strategy = self.get_strategy(infoset, legal_actions)

        actions, probs = zip(*strategy.items())
        chosen_action = self.rng.choices(actions, probs)[0]
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

    def _train_on_engine(
        self,
        engine: Engine,
        traversals_per_hand: int = 1,
    ) -> None:
        traversals = max(1, traversals_per_hand)
        for _ in range(traversals):
            for player in engine.players:
                if engine.betting.stacks.get(player, engine.betting.starting_stack) <= 0:
                    continue
                self.mccfr(engine.clone(), player)
            self.iteration_count += 1

    def _maybe_log_progress(self, processed: int, log_interval: int, label: str = "Hand") -> None:
        if log_interval <= 0:
            return
        if processed % log_interval != 0:
            return
        print(
            f"[{label} {processed}] Iter={self.iteration_count} "
            f"Regrets={len(self.regret_sum)} Strategy={len(self.strategy_sum)}"
        )

    def _load_jsonl_records(self, dataset_path: Path, max_hands: Optional[int] = None) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        with dataset_path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                raw = line.strip()
                if not raw:
                    continue
                payload = json.loads(raw)
                if not isinstance(payload, dict):
                    raise ValueError(f"Line {line_no} is not a JSON object")
                records.append(payload)
                if max_hands is not None and len(records) >= max_hands:
                    break
        return records

    def _looks_like_full_hand_record(self, record: Dict[str, Any]) -> bool:
        required_any = {"hole_cards", "stacks", "bets", "action_history"}
        return any(key in record for key in required_any)

    def _infer_players_from_hand(self, hand: Dict[str, Any]) -> tuple[str, ...]:
        player_ids: set[str] = set()

        for key in ("stacks", "bets", "hole_cards"):
            value = hand.get(key)
            if isinstance(value, dict):
                for player_id in value.keys():
                    if isinstance(player_id, str) and player_id.startswith("p"):
                        player_ids.add(player_id)

        folded = hand.get("folded_players")
        if isinstance(folded, list):
            for player_id in folded:
                if isinstance(player_id, str) and player_id.startswith("p"):
                    player_ids.add(player_id)

        if not player_ids:
            return ("p1", "p2")

        def player_sort_key(player_id: str) -> tuple[int, str]:
            suffix = player_id[1:]
            return (int(suffix), player_id) if suffix.isdigit() else (9999, player_id)

        ordered = tuple(sorted(player_ids, key=player_sort_key))
        return ordered if len(ordered) >= 2 else ("p1", "p2")

    def train_from_dataset(
        self,
        dataset_path: Path,
        log_interval: int = 100,
        *,
        epochs: int = 1,
        traversals_per_hand: int = 1,
        shuffle_each_epoch: bool = False,
        max_hands: Optional[int] = None,
    ) -> None:
        records = self._load_jsonl_records(dataset_path, max_hands=max_hands)
        if not records:
            raise ValueError(f"No records found in dataset: {dataset_path}")

        if not self._looks_like_full_hand_record(records[0]):
            raise ValueError(
                "Dataset does not look like full-hand snapshots required for MCCFR training "
                "(expected fields like hole_cards/stacks/bets/action_history). "
                "Use --self-play-hands to train from generated hands, or provide a full-hand JSONL."
            )

        safe_epochs = max(1, epochs)
        processed = 0
        for epoch in range(safe_epochs):
            epoch_records = list(records)
            if shuffle_each_epoch:
                self.rng.shuffle(epoch_records)

            for hand in epoch_records:
                engine = Engine()
                engine.players = self._infer_players_from_hand(hand)
                engine.load_hand(hand)
                self._train_on_engine(engine, traversals_per_hand=traversals_per_hand)
                processed += 1
                self._maybe_log_progress(processed, log_interval, label="Hand")

    def train_self_play(
        self,
        *,
        hands: int,
        log_interval: int = 100,
        traversals_per_hand: int = 1,
        players: int = 2,
        starting_stack: int = 1000,
        small_blind: int = 5,
        big_blind: int = 10,
        seed_base: int = 1,
    ) -> None:
        if players < 2 or players > 5:
            raise ValueError("players must be between 2 and 5")
        if hands <= 0:
            raise ValueError("hands must be > 0")

        player_ids = tuple(f"p{i}" for i in range(1, players + 1))
        for hand_index in range(hands):
            engine = Engine(players=player_ids)
            engine.betting.starting_stack = starting_stack
            engine.betting.small_blind = small_blind
            engine.betting.big_blind = big_blind
            engine.button_index = hand_index % players
            engine.new_hand(seed=seed_base + hand_index, rotate_button=False)
            self._train_on_engine(engine, traversals_per_hand=traversals_per_hand)
            self._maybe_log_progress(hand_index + 1, log_interval, label="SelfPlay")

    def export_strategy(self, filename: Path) -> None:
        strategy_table: Dict[str, Dict[str, float]] = {}
        for (infoset, action), total in self.strategy_sum.items():
            if infoset not in strategy_table:
                strategy_table[infoset] = {}
            strategy_table[infoset][action] = float(total)

        # Normalize per infoset row so probabilities sum to ~1 for runtime sampling.
        for infoset, row in strategy_table.items():
            row_total = sum(max(0.0, value) for value in row.values())
            if row_total <= 0:
                action_count = max(1, len(row))
                for action in list(row.keys()):
                    row[action] = 1.0 / action_count
                continue
            for action in list(row.keys()):
                row[action] = max(0.0, row[action]) / row_total
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
    parser.add_argument("--epochs", type=int, default=1, help="Passes over dataset (dataset mode)")
    parser.add_argument(
        "--traversals-per-hand",
        type=int,
        default=1,
        help="MCCFR traversals per hand snapshot/generated hand",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle dataset records each epoch (dataset mode)",
    )
    parser.add_argument(
        "--max-hands",
        type=int,
        default=None,
        help="Limit number of dataset records loaded (dataset mode)",
    )
    parser.add_argument("--seed", type=int, default=1, help="Training RNG seed")
    parser.add_argument(
        "--self-play-hands",
        type=int,
        default=0,
        help="Train from generated self-play starting hands instead of dataset",
    )
    parser.add_argument("--players", type=int, default=2, help="Players for self-play mode (2-5)")
    parser.add_argument("--stack", type=int, default=1000, help="Starting stack for self-play mode")
    parser.add_argument("--small-blind", type=int, default=5, help="Small blind for self-play mode")
    parser.add_argument("--big-blind", type=int, default=10, help="Big blind for self-play mode")
    args = parser.parse_args()

    trainer = MCCFRTrainer(seed=args.seed)
    if args.self_play_hands > 0:
        trainer.train_self_play(
            hands=args.self_play_hands,
            log_interval=args.log_interval,
            traversals_per_hand=args.traversals_per_hand,
            players=args.players,
            starting_stack=args.stack,
            small_blind=args.small_blind,
            big_blind=args.big_blind,
            seed_base=args.seed,
        )
    else:
        trainer.train_from_dataset(
            args.dataset,
            log_interval=args.log_interval,
            epochs=args.epochs,
            traversals_per_hand=args.traversals_per_hand,
            shuffle_each_epoch=args.shuffle,
            max_hands=args.max_hands,
        )
    trainer.export_strategy(args.output)
    print(
        f"Training complete. Iterations={trainer.iteration_count} "
        f"Regrets={len(trainer.regret_sum)} StrategyRows={len({k for k, _ in trainer.strategy_sum.keys()})} "
        f"Output={args.output}"
    )


if __name__ == "__main__":
    main()
