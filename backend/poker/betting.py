from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from ..schemas import Action, ActionRecord, ActionType


@dataclass
class StepResult:
    round_complete: bool
    hand_over: bool
    winner: Optional[str] = None


@dataclass
class BettingState:
    players: Tuple[str, ...] = ("p1", "p2")
    small_blind: int = 5
    big_blind: int = 10
    starting_stack: int = 1000
    stacks: Dict[str, int] = field(default_factory=dict)
    contributions: Dict[str, int] = field(default_factory=dict)
    pot: int = 0
    current_bet: int = 0
    last_raise_size: int = 0
    current_player: Optional[str] = None
    pending_players: Set[str] = field(default_factory=set)
    action_history: List[ActionRecord] = field(default_factory=list)
    hand_over: bool = False
    winner: Optional[str] = None
    folded_players: Set[str] = field(default_factory=set)
    all_in_players: Set[str] = field(default_factory=set)

    def start_hand(
        self,
        players: Tuple[str, ...],
        sb_player: str,
        bb_player: str,
        first_to_act: str,
    ) -> None:
        self.players = players
        if not self.stacks:
            self.stacks = {player: self.starting_stack for player in self.players}
        else:
            self.stacks = {
                player: max(0, int(chips))
                for player, chips in self.stacks.items()
            }
            for player in self.players:
                self.stacks.setdefault(player, self.starting_stack)

        self.contributions = {player: 0 for player in self.stacks}
        self.pot = 0
        self.current_bet = 0
        self.last_raise_size = self.big_blind
        self.pending_players = set()
        self.action_history = []
        self.hand_over = False
        self.winner = None
        self.folded_players = set()
        self.all_in_players = set()

        if len(self.players) < 2:
            self.hand_over = True
            self.winner = self.players[0] if self.players else None
            self.current_player = None
            return

        self._post_blind(sb_player, self.small_blind)
        self._post_blind(bb_player, self.big_blind)
        self.current_bet = max(
            (self.contributions[player] for player in self.players),
            default=0,
        )
        self.all_in_players = {player for player in self.players if self.stacks[player] == 0}
        self.pending_players = set(self.active_players()) - set(self.all_in_players)
        self.current_player = first_to_act
        if self.current_player in self.all_in_players:
            self.current_player = self._next_player(self.current_player)

    def start_new_round(self, first_to_act: str) -> None:
        self.contributions = {player: 0 for player in self.players}
        self.current_bet = 0
        self.last_raise_size = self.big_blind
        self.pending_players = set(self.active_players()) - set(self.all_in_players)
        self.current_player = first_to_act

    def legal_actions(self) -> List[ActionType]:
        if self.hand_over or self.current_player is None:
            return []

        to_call = self.to_call(self.current_player)
        stack = self.stacks[self.current_player]
        actions = [ActionType.FOLD]

        if to_call == 0:
            actions.append(ActionType.CHECK)
        else:
            if stack > 0:
                actions.append(ActionType.CALL)

        if stack > to_call and self._can_raise(self.current_player):
            actions.append(ActionType.RAISE)

        return actions

    def to_call(self, player_id: str) -> int:
        return max(0, self.current_bet - self.contributions.get(player_id, 0))

    def min_raise_to(self) -> int:
        return self._min_raise_to()

    def max_raise_to(self, player_id: str) -> int:
        return self.contributions[player_id] + self.stacks[player_id]

    def step(self, action: Action, player_id: str) -> StepResult:
        if self.hand_over:
            raise ValueError("Hand is over")
        if player_id != self.current_player:
            raise ValueError("Not this player's turn")
        if player_id in self.folded_players:
            raise ValueError("Player has already folded")

        to_call = self.to_call(player_id)
        round_complete = False

        if action.action == ActionType.FOLD:
            self._record_action(player_id, action)
            self.folded_players.add(player_id)
            self.pending_players.discard(player_id)
            active = self.active_players()
            if len(active) == 1:
                self.hand_over = True
                self.winner = active[0]
                self.current_player = None
                return StepResult(
                    round_complete=False, hand_over=True, winner=self.winner
                )
            round_complete = self._round_complete()

        elif action.action == ActionType.CHECK:
            if to_call != 0:
                raise ValueError("Cannot check when facing a bet")
            self._record_action(player_id, action)
            self._complete_player_action(player_id)
            round_complete = self._round_complete()

        elif action.action == ActionType.CALL:
            if to_call == 0:
                raise ValueError("Cannot call when there is no bet")
            self._record_action(player_id, action)
            call_amount = min(to_call, self.stacks[player_id])
            self._apply_chips(player_id, call_amount)
            if self.stacks[player_id] == 0:
                self.all_in_players.add(player_id)
            self._complete_player_action(player_id)
            round_complete = self._round_complete()

        elif action.action == ActionType.RAISE:
            if action.amount is None:
                raise ValueError("amount is required for raise")
            self._apply_raise(player_id, action.amount)
            self._record_action(player_id, action)
            round_complete = False

        else:
            raise ValueError("Unsupported action")

        if not self.hand_over and not round_complete:
            self.current_player = self._next_player(player_id)
            if self.current_player is None:
                round_complete = True
        elif round_complete:
            self.current_player = None

        return StepResult(
            round_complete=round_complete, hand_over=self.hand_over, winner=self.winner
        )

    def _apply_chips(self, player_id: str, amount: int) -> None:
        if amount > self.stacks[player_id]:
            raise ValueError("Not enough chips to call")
        self.stacks[player_id] -= amount
        self.contributions[player_id] += amount
        self.pot += amount

    def _apply_raise(self, player_id: str, raise_to: int) -> None:
        if raise_to <= self.current_bet:
            raise ValueError("Raise must increase the current bet")

        player_stack = self.stacks[player_id]
        contributed = self.contributions[player_id]
        required = raise_to - contributed
        if required > player_stack:
            raise ValueError("Raise exceeds stack")

        min_raise_to = self._min_raise_to()
        all_in = required == player_stack
        if raise_to < min_raise_to and not all_in:
            raise ValueError("Raise below minimum")

        raise_size = raise_to - self.current_bet
        if not (all_in and raise_to < min_raise_to):
            self.last_raise_size = raise_size
        self.current_bet = raise_to
        self._apply_chips(player_id, required)
        if self.stacks[player_id] == 0:
            self.all_in_players.add(player_id)
        self.pending_players = set(self.active_players()) - {player_id} - set(
            self.all_in_players
        )

    def _can_raise(self, player_id: str) -> bool:
        contributed = self.contributions[player_id]
        stack = self.stacks[player_id]
        return contributed + stack > self.current_bet

    def _min_raise_to(self) -> int:
        if self.current_bet == 0:
            return self.last_raise_size
        return self.current_bet + self.last_raise_size

    def active_players(self) -> List[str]:
        return [player for player in self.players if player not in self.folded_players]

    def _next_player(self, current_player: str) -> Optional[str]:
        if not self.players:
            return None
        start_index = self.players.index(current_player)
        for offset in range(1, len(self.players) + 1):
            candidate = self.players[(start_index + offset) % len(self.players)]
            if candidate in self.folded_players:
                continue
            if candidate in self.all_in_players:
                continue
            if candidate in self.pending_players:
                return candidate
        return None

    def _post_blind(self, player_id: str, amount: int) -> None:
        blind_amount = min(amount, self.stacks[player_id])
        self.stacks[player_id] -= blind_amount
        self.contributions[player_id] += blind_amount
        self.pot += blind_amount

    def _record_action(self, player_id: str, action: Action) -> None:
        self.action_history.append(ActionRecord(player_id=player_id, action=action))

    def _complete_player_action(self, player_id: str) -> None:
        self.pending_players.discard(player_id)

    def _round_complete(self) -> bool:
        return len(self.pending_players) == 0

    def payout(self, winner: Optional[object], remainder_to: Optional[str] = None) -> None:
        if winner is None or winner == "tie":
            winners = list(self.active_players())
        elif isinstance(winner, list):
            winners = winner
        else:
            winners = [winner]

        split = self.pot // len(winners)
        remainder = self.pot % len(winners)
        for player in winners:
            self.stacks[player] += split
        if remainder:
            recipient = remainder_to or winners[0]
            self.stacks[recipient] += remainder
        self.pot = 0
        self.hand_over = True
        self.winner = winner
