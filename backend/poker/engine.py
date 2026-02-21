from __future__ import annotations

import random
from dataclasses import dataclass, field
import json
from typing import Dict, List, Optional, Tuple

from .betting import BettingState
from .cards import Deck, build_deck
from .evaluator import compare_hands, evaluate_hand, hand_category
from ..schemas import Action, EventMessage, EventType, GameStatePublic, Street


HUMAN_PLAYER_ID = "p1"
AI_PLAYER_ID = "p2"
DEFAULT_PLAYERS = ("p1", "p2", "p3", "p4", "p5")
DEFAULT_HISTORY_LIMIT = 10
DEFAULT_HAND_STRENGTH_ROLLOUTS = 120
HAND_CATEGORY_ORDER = (
    "Straight Flush",
    "Four of a Kind",
    "Full House",
    "Flush",
    "Straight",
    "Three of a Kind",
    "Two Pair",
    "Pair",
    "High Card",
)


@dataclass
class Engine:
    deck: Deck = field(default_factory=Deck)
    board: List[str] = field(default_factory=list)
    hole_cards: Dict[str, List[str]] = field(default_factory=dict)
    street: Street = Street.PREFLOP
    betting: BettingState = field(default_factory=BettingState)
    players: Tuple[str, ...] = field(default_factory=lambda: DEFAULT_PLAYERS)
    button_index: int = 0
    button_player: str = HUMAN_PLAYER_ID
    sb_player: str = HUMAN_PLAYER_ID
    bb_player: str = "p2"
    pending_events: List[EventMessage] = field(default_factory=list)
    _rng: random.Random = field(default_factory=random.Random, init=False)
    _starting_stacks: Dict[str, int] = field(default_factory=dict, init=False)

    def _eligible_players_for_hand(self) -> Tuple[str, ...]:
        if not self.betting.stacks:
            return self.players
        return tuple(
            player
            for player in self.players
            if self.betting.stacks.get(player, self.betting.starting_stack) > 0
        )

    def _next_eligible_player_from_index(
        self,
        eligible_players: Tuple[str, ...],
        start_index: int,
    ) -> Optional[str]:
        if not eligible_players:
            return None
        eligible_set = set(eligible_players)
        for offset in range(len(self.players)):
            candidate = self.players[(start_index + offset) % len(self.players)]
            if candidate in eligible_set:
                return candidate
        return None

    def new_hand(
        self, seed: Optional[int] = None, rotate_button: bool = False
    ) -> Dict[str, List[str]]:
        if len(self.players) < 2 or len(self.players) > 5:
            raise ValueError("Engine supports between 2 and 5 players")
        if rotate_button:
            self.button_index = (self.button_index + 1) % len(self.players)
        self.button_index = self.button_index % len(self.players)
        hand_players = self._eligible_players_for_hand()
        if not hand_players:
            hand_players = self.players

        button_player = self._next_eligible_player_from_index(
            hand_players,
            self.button_index,
        )
        if button_player is None:
            button_player = hand_players[0]
        self.button_player = button_player
        button_hand_index = hand_players.index(button_player)
        if len(hand_players) == 2:
            sb_index = button_hand_index
            bb_index = (button_hand_index + 1) % len(hand_players)
            first_to_act = hand_players[sb_index]
        else:
            sb_index = (button_hand_index + 1) % len(hand_players)
            bb_index = (button_hand_index + 2) % len(hand_players)
            first_to_act = hand_players[(bb_index + 1) % len(hand_players)]
        self.sb_player = hand_players[sb_index]
        self.bb_player = hand_players[bb_index]
        self._rng = random.Random(seed)
        self.deck = Deck()
        self.deck.shuffle(self._rng)
        self.board = []
        self.street = Street.PREFLOP
        hand_player_set = set(hand_players)
        self.hole_cards = {
            player: self.deck.deal(2) if player in hand_player_set else []
            for player in self.players
        }
        self.betting.start_hand(
            players=hand_players,
            sb_player=hand_players[sb_index],
            bb_player=hand_players[bb_index],
            first_to_act=first_to_act,
        )
        self._starting_stacks = dict(self.betting.stacks)
        self._queue_event(
            EventType.DEAL_HOLE,
            {"street": self.street.value, "cards": []},
        )
        return self.hole_cards

    def start_next_hand(self, seed: Optional[int] = None) -> Dict[str, List[str]]:
        return self.new_hand(seed=seed, rotate_button=True)

    def deal_flop(self) -> List[str]:
        dealt = self.deck.deal(3)
        self.board.extend(dealt)
        self.street = Street.FLOP
        self._queue_event(
            EventType.DEAL_FLOP,
            {"street": self.street.value, "cards": list(dealt)},
        )
        return list(self.board)

    def deal_turn(self) -> List[str]:
        dealt = self.deck.deal(1)
        self.board.extend(dealt)
        self.street = Street.TURN
        self._queue_event(
            EventType.DEAL_TURN,
            {"street": self.street.value, "cards": list(dealt)},
        )
        return list(self.board)

    def deal_river(self) -> List[str]:
        dealt = self.deck.deal(1)
        self.board.extend(dealt)
        self.street = Street.RIVER
        self._queue_event(
            EventType.DEAL_RIVER,
            {"street": self.street.value, "cards": list(dealt)},
        )
        return list(self.board)

    def evaluate_showdown(self) -> Tuple[str, int, int]:
        if len(self.hole_cards) < 2:
            raise ValueError("Hole cards not dealt")
        if len(self.board) < 5:
            raise ValueError("Board must have 5 cards to evaluate showdown")
        self.street = Street.SHOWDOWN
        return compare_hands(
            self.hole_cards[HUMAN_PLAYER_ID],
            self.hole_cards[self._first_ai_player()],
            self.board,
        )

    def step(self, action: Action, player_id: str = HUMAN_PLAYER_ID) -> None:
        result = self.betting.step(action, player_id)

        if result.hand_over:
            self._end_hand_by_fold(result.winner)
            return

        if result.round_complete:
            if self.street == Street.PREFLOP:
                self.deal_flop()
                self.betting.start_new_round(
                    first_to_act=self._first_to_act_postflop()
                )
            elif self.street == Street.FLOP:
                self.deal_turn()
                self.betting.start_new_round(
                    first_to_act=self._first_to_act_postflop()
                )
            elif self.street == Street.TURN:
                self.deal_river()
                self.betting.start_new_round(
                    first_to_act=self._first_to_act_postflop()
                )
            elif self.street == Street.RIVER:
                self._resolve_showdown()

    def drain_events(self) -> List[EventMessage]:
        events = list(self.pending_events)
        self.pending_events.clear()
        return events

    def _queue_event(self, event: EventType, data: Optional[Dict[str, object]] = None) -> None:
        self.pending_events.append(EventMessage(event=event, data=data))

    def _end_hand_by_fold(self, winner: Optional[str]) -> None:
        pot_total = self.betting.pot
        self.betting.payout(winner, remainder_to=self.button_player)
        self.street = Street.SHOWDOWN
        self._queue_event(
            EventType.HAND_END,
            {
                "winner": winner,
                "hand_category": None,
                "pot": pot_total,
            },
        )

    def _resolve_showdown(self) -> None:
        active_players = self.betting.active_players()
        scores: Dict[str, int] = {}
        for player_id in active_players:
            scores[player_id] = evaluate_hand(
                self.hole_cards[player_id], self.board
            )
        best_score = min(scores.values())
        winners = [player for player, score in scores.items() if score == best_score]
        pot_total = self.betting.pot
        category = hand_category(best_score)
        if len(winners) == 1:
            self.betting.payout(winners[0], remainder_to=self.button_player)
            winner_field: object = winners[0]
        else:
            self.betting.payout(winners, remainder_to=self.button_player)
            winner_field = winners

        self._queue_event(
            EventType.HAND_END,
            {
                "winner": winner_field,
                "hand_category": category,
                "pot": pot_total,
            },
        )

    def resolve_showdown(self) -> None:
        self._resolve_showdown()

    def _estimate_viewer_outcomes(
        self,
        hole_cards: List[str],
        board_cards: List[str],
        n_opponents: int,
        rollouts: int = DEFAULT_HAND_STRENGTH_ROLLOUTS,
    ) -> Tuple[float, Dict[str, float]]:
        opponent_count = max(0, n_opponents)
        safe_rollouts = max(1, rollouts)

        known_cards = set(hole_cards + board_cards)
        deck = [card for card in build_deck() if card not in known_cards]
        board_cards_needed = max(0, 5 - len(board_cards))
        draw_count = board_cards_needed + (2 * opponent_count)
        default_probs = {category: 0.0 for category in HAND_CATEGORY_ORDER}
        if draw_count < 0 or draw_count > len(deck):
            return 0.0, default_probs

        total_score = 0.0
        category_counts: Dict[str, int] = {category: 0 for category in HAND_CATEGORY_ORDER}
        for _ in range(safe_rollouts):
            drawn = self._rng.sample(deck, draw_count)
            completed_board = board_cards + drawn[:board_cards_needed]
            hero_score = evaluate_hand(hole_cards, completed_board)
            hero_category = hand_category(hero_score)
            category_counts[hero_category] = category_counts.get(hero_category, 0) + 1

            if opponent_count <= 0:
                total_score += 1.0
                continue

            contenders: List[Tuple[str, int]] = [("hero", hero_score)]
            opp_start = board_cards_needed
            for i in range(opponent_count):
                opp_hole = drawn[opp_start + (i * 2): opp_start + ((i + 1) * 2)]
                contenders.append((f"opp{i}", evaluate_hand(opp_hole, completed_board)))

            best_score = min(score for _, score in contenders)
            winners = [player for player, score in contenders if score == best_score]
            if "hero" in winners:
                total_score += 1.0 / len(winners)

        category_probs = {
            category: (count * 100.0) / safe_rollouts
            for category, count in category_counts.items()
        }
        return total_score / safe_rollouts, category_probs

    def _estimate_viewer_equity(
        self,
        hole_cards: List[str],
        board_cards: List[str],
        n_opponents: int,
        rollouts: int = DEFAULT_HAND_STRENGTH_ROLLOUTS,
    ) -> float:
        equity, _ = self._estimate_viewer_outcomes(
            hole_cards=hole_cards,
            board_cards=board_cards,
            n_opponents=n_opponents,
            rollouts=rollouts,
        )
        return equity

    def _viewer_strength(
        self, viewer: Optional[str]
    ) -> Tuple[Optional[str], Optional[float], Optional[Dict[str, float]]]:
        if not viewer:
            return None, None, None
        hole_cards = self.hole_cards.get(viewer) or []
        if len(hole_cards) != 2:
            return None, None, None

        board_cards = list(self.board)
        if len(board_cards) >= 3:
            try:
                score = evaluate_hand(hole_cards, board_cards)
                label = hand_category(score)
            except Exception:
                label = "Hand"
        else:
            same_rank = hole_cards[0][0] == hole_cards[1][0]
            suited = hole_cards[0][1] == hole_cards[1][1]
            if same_rank:
                label = "Pocket Pair"
            elif suited:
                label = "Suited"
            else:
                label = "High Card"

        active_opponents = [
            player
            for player in self.betting.active_players()
            if player != viewer
        ]
        n_opponents = max(0, len(active_opponents))

        try:
            equity, category_probs = self._estimate_viewer_outcomes(
                hole_cards=hole_cards,
                board_cards=board_cards,
                n_opponents=n_opponents,
            )
        except Exception:
            return label, None, None

        rounded_probs = {
            category: round(prob, 1)
            for category, prob in category_probs.items()
        }
        return label, round(equity * 100, 1), rounded_probs

    def to_public_state(
        self,
        viewer: Optional[str] = HUMAN_PLAYER_ID,
        history_limit: int = DEFAULT_HISTORY_LIMIT,
        session_id: Optional[str] = None,
    ) -> Dict[str, object]:
        player_hand = self.hole_cards.get(viewer) if viewer in self.hole_cards else None
        hand_strength_label, hand_strength_pct, hand_category_probs = self._viewer_strength(
            viewer
        )
        revealed_hands = None
        current_player = self.betting.current_player
        to_call = self.betting.to_call(current_player) if current_player else None
        min_raise_to = self.betting.min_raise_to() if current_player else None
        max_raise_to = (
            self.betting.max_raise_to(current_player) if current_player else None
        )
        if self.street == Street.SHOWDOWN or self.betting.hand_over:
            revealed_hands = {
                player: list(cards)
                for player, cards in self.hole_cards.items()
                if len(cards) == 2
            }

        state = GameStatePublic(
            session_id=session_id,
            street=self.street,
            pot=self.betting.pot,
            community_cards=list(self.board),
            hand=player_hand,
            revealed_hands=revealed_hands,
            folded_players=sorted(self.betting.folded_players),
            stacks=dict(self.betting.stacks),
            bets=dict(self.betting.contributions),
            button_player=self.button_player,
            small_blind_player=self.sb_player,
            big_blind_player=self.bb_player,
            current_player=current_player,
            legal_actions=list(self.betting.legal_actions()),
            to_call=to_call,
            min_raise_to=min_raise_to,
            max_raise_to=max_raise_to,
            action_history=list(self.betting.action_history[-history_limit:]),
            hand_strength_label=hand_strength_label,
            hand_strength_pct=hand_strength_pct,
            hand_category_probs=hand_category_probs,
        )

        return json.loads(state.json(by_alias=True, exclude_none=True))

    def to_ai_state(self) -> Dict[str, object]:
        current_player = self.betting.current_player or self._first_ai_player()
        return {
            "street": self.street.value,
            "legal_actions": [action.value for action in self.betting.legal_actions()],
            "min_raise_to": self.betting.min_raise_to(),
            "max_raise_to": self.betting.max_raise_to(current_player),
            "to_call": self.betting.to_call(current_player),
            "stacks": dict(self.betting.stacks),
            "bets": dict(self.betting.contributions),
            "current_player": current_player,
            "big_blind": self.betting.big_blind,
            "pot": self.betting.pot,
            "community_cards": list(self.board),
            "hand": list(self.hole_cards.get(current_player, [])),
            "action_history": list(self.betting.action_history),
        }

    def _first_ai_player(self) -> str:
        for player in self.players:
            if player != HUMAN_PLAYER_ID:
                return player
        return HUMAN_PLAYER_ID

    def _small_blind_index(self) -> int:
        if len(self.players) == 2:
            return self.button_index
        return (self.button_index + 1) % len(self.players)

    def _big_blind_index(self) -> int:
        if len(self.players) == 2:
            return (self.button_index + 1) % len(self.players)
        return (self.button_index + 2) % len(self.players)

    def _first_to_act_preflop(self, bb_index: int) -> str:
        if len(self.players) == 2:
            return self.players[self._small_blind_index()]
        return self.players[(bb_index + 1) % len(self.players)]

    def _first_to_act_postflop(self) -> str:
        players_in_hand = self.betting.players or self.players
        if not players_in_hand:
            return self.button_player

        if self.button_player in players_in_hand:
            start_index = (players_in_hand.index(self.button_player) + 1) % len(players_in_hand)
        else:
            start_index = 0

        for offset in range(len(players_in_hand)):
            candidate = players_in_hand[(start_index + offset) % len(players_in_hand)]
            if candidate in self.betting.folded_players:
                continue
            if candidate in self.betting.all_in_players:
                continue
            return candidate
        return players_in_hand[start_index]

    def clone(self) -> "Engine":
        import copy

        return copy.deepcopy(self)

    def is_terminal(self) -> bool:
        return self.street == Street.SHOWDOWN or self.betting.hand_over

    def utility(self, player_id: str) -> int:
        current_stack = self.betting.stacks.get(player_id, 0)
        starting_stack = self._starting_stacks.get(player_id, self.betting.starting_stack)
        return current_stack - starting_stack

    def get_chip_change(self, player_id: str) -> int:
        return self.utility(player_id)

    def load_hand(self, hand: Dict[str, object]) -> None:
        self.deck = Deck()
        self.board = list(hand.get("board", []))
        self.hole_cards = dict(hand.get("hole_cards", {}))

        for player in self.players:
            self.hole_cards.setdefault(player, [])

        street_str = str(hand.get("street", Street.PREFLOP.value))
        try:
            self.street = Street(street_str)
        except ValueError:
            self.street = Street.PREFLOP

        stacks = dict(hand.get("stacks", {}))
        for player in self.players:
            stacks.setdefault(player, self.betting.starting_stack)
        self.betting.stacks = stacks
        self._starting_stacks = dict(stacks)

        self.betting.pot = int(hand.get("pot", 0))
        self.betting.contributions = dict(hand.get("bets", {}))
        for player in self.players:
            self.betting.contributions.setdefault(player, 0)

        raw_history = hand.get("action_history", [])
        if isinstance(raw_history, list):
            self.betting.action_history = raw_history
        else:
            self.betting.action_history = []

        raw_current = hand.get("current_player")
        if isinstance(raw_current, str) and raw_current in self.players:
            self.betting.current_player = raw_current
        else:
            self.betting.current_player = self._first_to_act_preflop(self._big_blind_index())

        self.betting.hand_over = bool(hand.get("hand_over", False))
        self.betting.folded_players = set(hand.get("folded_players", []))
        self.betting.all_in_players = set(hand.get("all_in_players", []))
        self.betting.pending_players = set(hand.get("pending_players", []))
