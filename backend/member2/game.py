from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Iterable, Optional

from .cards import int_to_card
from .evaluator import best_hand_rank
from .ehs import EHSEngine


@dataclass
class Player:
    name: str
    stack: int
    is_human: bool = False
    ai_style: str = "balanced"
    cards: list[int] = field(default_factory=list)
    folded: bool = False
    all_in: bool = False
    bet_round: int = 0
    contrib: int = 0

    def reset_for_hand(self) -> None:
        self.cards = []
        self.folded = False
        self.all_in = False
        self.bet_round = 0
        self.contrib = 0


@dataclass
class Pot:
    amount: int
    eligible: list[int]


class PokerGame:
    def __init__(
        self,
        players: list[Player],
        small_blind: int,
        big_blind: int,
        rng_seed: Optional[int] = None,
        log_path: Optional[str] = None,
        show_ehs: bool = False,
        ehs_rollouts: int = 200,
        ai_rollouts: int = 200,
    ) -> None:
        if small_blind <= 0 or big_blind <= 0:
            raise ValueError("Blinds must be positive")
        if big_blind < small_blind:
            raise ValueError("Big blind must be >= small blind")
        if len(players) < 2:
            raise ValueError("Need at least 2 players")
        self.players = players
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.dealer_index = 0
        self.rng = random.Random(rng_seed)
        self.board: list[int] = []
        self.pot = 0
        self.current_bet = 0
        self.last_raise = big_blind
        self.log_path = log_path
        self.show_ehs = show_ehs
        self.ehs_engine = EHSEngine(rollouts=ehs_rollouts, n_opponents=1, seed=rng_seed)
        self.ai_engine = EHSEngine(rollouts=ai_rollouts, n_opponents=1, seed=rng_seed)

    def log(self, message: str) -> None:
        print(message)
        if self.log_path:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(message + "\n")

    def active_indices(self) -> list[int]:
        return [i for i, p in enumerate(self.players) if p.stack > 0]

    def in_hand_indices(self) -> list[int]:
        return [i for i, p in enumerate(self.players) if p.cards and not p.folded]

    def next_active_index(self, start: int) -> int:
        if not self.active_indices():
            return start
        idx = start
        while True:
            idx = (idx + 1) % len(self.players)
            if self.players[idx].stack > 0:
                return idx

    def next_in_hand_index(self, start: int) -> int:
        idx = start
        for _ in range(len(self.players)):
            idx = (idx + 1) % len(self.players)
            p = self.players[idx]
            if p.cards and not p.folded and not p.all_in:
                return idx
        return start

    def reset_hand(self) -> None:
        self.board = []
        self.pot = 0
        self.current_bet = 0
        self.last_raise = self.big_blind
        for p in self.players:
            p.reset_for_hand()

    def shuffle_deck(self) -> list[int]:
        deck = list(range(52))
        self.rng.shuffle(deck)
        return deck

    def deal_hole(self, deck: list[int]) -> None:
        active = self.active_indices()
        for _ in range(2):
            for idx in active:
                self.players[idx].cards.append(deck.pop())

    def post_blind(self, idx: int, amount: int) -> None:
        p = self.players[idx]
        to_post = min(amount, p.stack)
        p.stack -= to_post
        p.bet_round += to_post
        p.contrib += to_post
        self.pot += to_post
        if p.stack == 0:
            p.all_in = True
        self.log(f"{p.name} posts blind {to_post}")

    def post_blinds(self) -> tuple[int, int]:
        sb_idx = self.next_active_index(self.dealer_index)
        bb_idx = self.next_active_index(sb_idx)
        self.post_blind(sb_idx, self.small_blind)
        self.post_blind(bb_idx, self.big_blind)
        self.current_bet = max(self.current_bet, self.players[bb_idx].bet_round)
        return sb_idx, bb_idx

    def betting_round(self, start_index: int) -> bool:
        active_to_act = [
            i
            for i, p in enumerate(self.players)
            if p.cards and not p.folded and not p.all_in
        ]
        if len(self.in_hand_indices()) <= 1 or not active_to_act:
            return False

        to_act = len(active_to_act)
        idx = start_index

        while to_act > 0:
            p = self.players[idx]
            if p.cards and not p.folded and not p.all_in:
                action, amount = self.decide_action(p)
                if action == "fold":
                    p.folded = True
                    self.log(f"{p.name} folds")
                    to_act -= 1
                elif action == "check":
                    self.log(f"{p.name} checks")
                    to_act -= 1
                elif action == "call":
                    self.collect_bet(p, amount)
                    self.log(f"{p.name} calls {amount}")
                    to_act -= 1
                elif action == "raise":
                    prev_bet = self.current_bet
                    self.collect_bet(p, amount)
                    self.current_bet = p.bet_round
                    self.last_raise = max(self.current_bet - prev_bet, self.big_blind)
                    self.log(f"{p.name} raises to {p.bet_round}")
                    to_act = len(
                        [
                            i
                            for i, pl in enumerate(self.players)
                            if pl.cards and not pl.folded and not pl.all_in and i != idx
                        ]
                    )
                elif action == "all_in":
                    prev_bet = self.current_bet
                    self.collect_bet(p, amount)
                    if p.bet_round > prev_bet:
                        self.current_bet = p.bet_round
                        self.last_raise = max(self.current_bet - prev_bet, self.big_blind)
                        to_act = len(
                            [
                                i
                                for i, pl in enumerate(self.players)
                                if pl.cards and not pl.folded and not pl.all_in and i != idx
                            ]
                        )
                    else:
                        to_act -= 1
                    self.log(f"{p.name} is all-in for {amount}")

                if len(self.in_hand_indices()) <= 1:
                    return True

            if to_act == 0:
                break
            if not [
                pl
                for pl in self.players
                if pl.cards and not pl.folded and not pl.all_in
            ]:
                break
            idx = self.next_in_hand_index(idx)

        return False

    def collect_bet(self, player: Player, amount: int) -> None:
        amount = min(amount, player.stack)
        player.stack -= amount
        player.bet_round += amount
        player.contrib += amount
        self.pot += amount
        if player.stack == 0:
            player.all_in = True

    def decide_action(self, player: Player) -> tuple[str, int]:
        to_call = max(0, self.current_bet - player.bet_round)

        if player.is_human:
            return self.human_action(player, to_call)
        return self.ai_action(player, to_call)

    def human_action(self, player: Player, to_call: int) -> tuple[str, int]:
        self.show_state(player, to_call)
        while True:
            action = input("Action [f=fold, c=call/check, r=raise, a=all-in, h=help]: ").strip().lower()
            if action in {"h", "help"}:
                self.show_help()
                continue
            if action in {"f", "fold"}:
                return "fold", 0
            if action in {"c", "call", "check"}:
                if to_call == 0:
                    return "check", 0
                return "call", min(to_call, player.stack)
            if action in {"a", "all", "all-in"}:
                return "all_in", player.stack
            if action in {"r", "raise", "bet"}:
                min_total = self.current_bet + max(self.last_raise, self.big_blind)
                if self.current_bet == 0:
                    min_total = self.big_blind
                max_total = player.bet_round + player.stack
                if max_total <= self.current_bet:
                    self.log("You cannot raise with your current stack.")
                    continue
                amount = self.prompt_raise_amount(min_total, max_total)
                raise_amount = amount - player.bet_round
                if raise_amount <= 0:
                    self.log("Invalid raise amount.")
                    continue
                if amount >= player.bet_round + player.stack:
                    return "all_in", player.stack
                return "raise", raise_amount
            self.log("Invalid action.")

    def show_state(self, player: Player, to_call: int) -> None:
        self.log("=" * 40)
        self.log(f"Board: {cards_to_str(self.board)}")
        self.log(f"Your hand: {cards_to_str(player.cards)}")
        self.log(f"Pot: {self.pot}")
        self.log(f"To call: {to_call}")
        self.log(f"Your stack: {player.stack}")
        if self.show_ehs:
            opp_count = max(1, len([p for p in self.players if p.cards and not p.folded]) - 1)
            self.ehs_engine.n_opponents = opp_count
            ehs = self.ehs_engine.compute(player.cards, self.board).ehs
            self.log(f"Estimated hand strength: {ehs:.3f}")
        self.log("=" * 40)

    def show_help(self) -> None:
        self.log("f=fold, c=call/check, r=raise/bet, a=all-in")

    def prompt_raise_amount(self, min_total: int, max_total: int) -> int:
        while True:
            raw = input(f"Raise to (min {min_total}, max {max_total}): ").strip().lower()
            if raw in {"a", "all"}:
                return max_total
            try:
                amount = int(raw)
            except ValueError:
                self.log("Enter a number or 'all'.")
                continue
            if amount < min_total:
                self.log("Amount below minimum raise.")
                continue
            if amount > max_total:
                self.log("Amount above max stack.")
                continue
            return amount

    def ai_action(self, player: Player, to_call: int) -> tuple[str, int]:
        active_players = [p for p in self.players if p.cards and not p.folded]
        opp_count = max(1, len(active_players) - 1)
        self.ai_engine.n_opponents = opp_count
        ehs = self.ai_engine.compute(player.cards, self.board).ehs

        pot_odds = 0.0
        if to_call > 0:
            pot_odds = to_call / (self.pot + to_call)

        style = player.ai_style
        fold_threshold = 0.25
        raise_threshold = 0.6
        bluff_chance = 0.02

        if style == "tight":
            fold_threshold = 0.35
            raise_threshold = 0.7
            bluff_chance = 0.01
        elif style == "aggressive":
            fold_threshold = 0.2
            raise_threshold = 0.5
            bluff_chance = 0.05
        elif style == "random":
            return self.ai_random_action(player, to_call)

        if to_call > 0 and ehs < pot_odds and ehs < fold_threshold:
            return "fold", 0

        if ehs > raise_threshold or (self.rng.random() < bluff_chance and to_call == 0):
            max_total = player.bet_round + player.stack
            min_total = self.current_bet + max(self.last_raise, self.big_blind)
            if self.current_bet == 0:
                min_total = self.big_blind
            target = min(max_total, max(min_total, self.current_bet + self.last_raise))
            raise_amount = target - player.bet_round
            if raise_amount >= player.stack:
                return "all_in", player.stack
            if raise_amount > 0:
                return "raise", raise_amount

        if to_call == 0:
            return "check", 0
        return "call", min(to_call, player.stack)

    def ai_random_action(self, player: Player, to_call: int) -> tuple[str, int]:
        if to_call > 0 and self.rng.random() < 0.5:
            return "fold", 0
        if to_call == 0 and self.rng.random() < 0.3:
            max_total = player.bet_round + player.stack
            min_total = self.current_bet + max(self.last_raise, self.big_blind)
            if self.current_bet == 0:
                min_total = self.big_blind
            target = min(max_total, min_total)
            raise_amount = target - player.bet_round
            if raise_amount >= player.stack:
                return "all_in", player.stack
            return "raise", raise_amount
        if to_call == 0:
            return "check", 0
        return "call", min(to_call, player.stack)

    def deal_flop(self, deck: list[int]) -> None:
        self.board.extend([deck.pop(), deck.pop(), deck.pop()])
        self.log(f"Flop: {cards_to_str(self.board)}")

    def deal_turn(self, deck: list[int]) -> None:
        self.board.append(deck.pop())
        self.log(f"Turn: {cards_to_str(self.board)}")

    def deal_river(self, deck: list[int]) -> None:
        self.board.append(deck.pop())
        self.log(f"River: {cards_to_str(self.board)}")

    def reset_bets(self) -> None:
        self.current_bet = 0
        self.last_raise = self.big_blind
        for p in self.players:
            p.bet_round = 0

    def showdown(self) -> None:
        self.log("Showdown")
        pots = build_side_pots(self.players)
        for pot in pots:
            elig_players = [self.players[i] for i in pot.eligible]
            best = None
            winners: list[Player] = []
            for p in elig_players:
                rank = best_hand_rank(p.cards + self.board)
                if best is None or rank > best:
                    best = rank
                    winners = [p]
                elif rank == best:
                    winners.append(p)
            share = pot.amount // len(winners)
            remainder = pot.amount % len(winners)
            for i, winner in enumerate(winners):
                award = share + (1 if i < remainder else 0)
                winner.stack += award
                self.log(f"{winner.name} wins {award}")
        for p in self.players:
            if p.cards:
                self.log(f"{p.name} shows {cards_to_str(p.cards)}")

    def award_pot_to_last(self) -> None:
        remaining = [p for p in self.players if p.cards and not p.folded]
        if len(remaining) == 1:
            remaining[0].stack += self.pot
            self.log(f"{remaining[0].name} wins pot {self.pot}")

    def play_hand(self) -> bool:
        active = self.active_indices()
        if len(active) < 2:
            return False

        self.reset_hand()
        deck = self.shuffle_deck()
        self.deal_hole(deck)
        sb_idx, bb_idx = self.post_blinds()

        self.log(f"Dealer: {self.players[self.dealer_index].name}")

        start = self.next_in_hand_index(bb_idx)
        if self.betting_round(start):
            self.award_pot_to_last()
            return True
        self.reset_bets()

        if len(self.in_hand_indices()) <= 1:
            self.award_pot_to_last()
            return True

        self.deal_flop(deck)
        start = self.next_in_hand_index(self.dealer_index)
        if self.betting_round(start):
            self.award_pot_to_last()
            return True
        self.reset_bets()

        if len(self.in_hand_indices()) <= 1:
            self.award_pot_to_last()
            return True

        self.deal_turn(deck)
        start = self.next_in_hand_index(self.dealer_index)
        if self.betting_round(start):
            self.award_pot_to_last()
            return True
        self.reset_bets()

        if len(self.in_hand_indices()) <= 1:
            self.award_pot_to_last()
            return True

        self.deal_river(deck)
        start = self.next_in_hand_index(self.dealer_index)
        if self.betting_round(start):
            self.award_pot_to_last()
            return True
        self.reset_bets()

        self.showdown()
        return True

    def rotate_dealer(self) -> None:
        if self.active_indices():
            self.dealer_index = self.next_active_index(self.dealer_index)


def build_side_pots(players: list[Player]) -> list[Pot]:
    contributions = [p.contrib for p in players if p.contrib > 0]
    if not contributions:
        return []
    levels = sorted(set(contributions))
    pots: list[Pot] = []
    prev = 0
    for level in levels:
        in_level = [i for i, p in enumerate(players) if p.contrib >= level]
        amount = (level - prev) * len(in_level)
        eligible = [i for i in in_level if not players[i].folded]
        pots.append(Pot(amount=amount, eligible=eligible))
        prev = level
    return pots


def cards_to_str(cards: Iterable[int]) -> str:
    return " ".join(int_to_card(c) for c in cards)


def run_cli_game(
    num_players: int,
    starting_stack: int,
    small_blind: int,
    big_blind: int,
    ai_style: str,
    rng_seed: Optional[int],
    log_path: Optional[str],
    show_ehs: bool,
    ehs_rollouts: int,
    ai_rollouts: int,
    max_hands: Optional[int],
) -> None:
    players = [Player(name="You", stack=starting_stack, is_human=True)]
    for i in range(1, num_players):
        players.append(Player(name=f"AI{i}", stack=starting_stack, ai_style=ai_style))

    game = PokerGame(
        players=players,
        small_blind=small_blind,
        big_blind=big_blind,
        rng_seed=rng_seed,
        log_path=log_path,
        show_ehs=show_ehs,
        ehs_rollouts=ehs_rollouts,
        ai_rollouts=ai_rollouts,
    )

    hand = 1
    while True:
        active = [p for p in game.players if p.stack > 0]
        if len(active) < 2:
            game.log("Game over")
            break
        if max_hands is not None and hand > max_hands:
            game.log("Reached max hands")
            break
        game.log(f"\n--- Hand {hand} ---")
        game.play_hand()
        game.log("Stacks: " + ", ".join(f"{p.name}:{p.stack}" for p in game.players))
        game.rotate_dealer()
        hand += 1
        cont = input("Play next hand? [Y/n]: ").strip().lower()
        if cont in {"n", "no", "q", "quit"}:
            break
