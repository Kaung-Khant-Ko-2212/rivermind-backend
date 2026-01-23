from __future__ import annotations

import random
from dataclasses import dataclass, field
import json
from typing import Dict, List, Optional, Tuple

from .betting import BettingState
from .cards import Deck
from .evaluator import compare_hands, evaluate_hand, hand_category
from ..schemas import Action, EventMessage, EventType, GameStatePublic, Street


HUMAN_PLAYER_ID = "p1"
AI_PLAYER_ID = "p2"
DEFAULT_PLAYERS = ("p1", "p2", "p3", "p4", "p5")
DEFAULT_HISTORY_LIMIT = 10


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
    bb_player: str = "p2"
    pending_events: List[EventMessage] = field(default_factory=list)
    _rng: random.Random = field(default_factory=random.Random, init=False)

    def new_hand(
        self, seed: Optional[int] = None, rotate_button: bool = False
    ) -> Dict[str, List[str]]:
        if len(self.players) < 2 or len(self.players) > 5:
            raise ValueError("Engine supports between 2 and 5 players")
        if rotate_button:
            self.button_index = (self.button_index + 1) % len(self.players)
        self.button_index = self.button_index % len(self.players)
        self.button_player = self.players[self.button_index]
        sb_index = self._small_blind_index()
        bb_index = self._big_blind_index()
        self.bb_player = self.players[bb_index]
        self._rng = random.Random(seed)
        self.deck = Deck()
        self.deck.shuffle(self._rng)
        self.board = []
        self.street = Street.PREFLOP
        self.hole_cards = {player: self.deck.deal(2) for player in self.players}
        first_to_act = self._first_to_act_preflop(bb_index)
        self.betting.start_hand(
            players=self.players,
            sb_player=self.players[sb_index],
            bb_player=self.players[bb_index],
            first_to_act=first_to_act,
        )
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

    def to_public_state(
        self,
        viewer: Optional[str] = HUMAN_PLAYER_ID,
        history_limit: int = DEFAULT_HISTORY_LIMIT,
        session_id: Optional[str] = None,
    ) -> Dict[str, object]:
        player_hand = self.hole_cards.get(viewer) if viewer in self.hole_cards else None

        state = GameStatePublic(
            session_id=session_id,
            street=self.street,
            pot=self.betting.pot,
            community_cards=list(self.board),
            hand=player_hand,
            stacks=dict(self.betting.stacks),
            bets=dict(self.betting.contributions),
            current_player=self.betting.current_player,
            legal_actions=list(self.betting.legal_actions()),
            action_history=list(self.betting.action_history[-history_limit:]),
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
        start_index = self._small_blind_index()
        for offset in range(len(self.players)):
            candidate = self.players[(start_index + offset) % len(self.players)]
            if candidate in self.betting.folded_players:
                continue
            if candidate in self.betting.all_in_players:
                continue
            return candidate
        return self.players[start_index]
