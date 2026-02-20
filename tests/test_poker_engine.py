import pytest

from backend.poker.cards import Deck, build_deck
from backend.poker.engine import Engine, HUMAN_PLAYER_ID
from backend.schemas import GameStatePublic

treys = pytest.importorskip("treys")
from backend.poker.evaluator import compare_hands


def test_deck_has_52_unique_cards() -> None:
    deck = build_deck()
    assert len(deck) == 52
    assert len(set(deck)) == 52


def test_dealing_removes_cards_from_deck() -> None:
    deck = Deck()
    initial = len(deck.cards)
    dealt = deck.deal(5)
    assert len(dealt) == 5
    assert len(deck.cards) == initial - 5
    assert not set(dealt).intersection(deck.cards)


def test_five_player_hand_deals_all_players() -> None:
    engine = Engine()
    engine.new_hand(seed=10)
    assert len(engine.hole_cards) == 5
    for cards in engine.hole_cards.values():
        assert len(cards) == 2


def test_deterministic_dealing_with_seed() -> None:
    engine_one = Engine(players=(HUMAN_PLAYER_ID, "p2"))
    engine_two = Engine(players=(HUMAN_PLAYER_ID, "p2"))

    engine_one.new_hand(seed=42)
    engine_two.new_hand(seed=42)

    engine_one.deal_flop()
    engine_one.deal_turn()
    engine_one.deal_river()

    engine_two.deal_flop()
    engine_two.deal_turn()
    engine_two.deal_river()

    assert engine_one.hole_cards == engine_two.hole_cards
    assert engine_one.board == engine_two.board


def test_evaluator_known_matchup() -> None:
    board = ["2c", "7d", "9h", "Jc", "Qd"]
    hole_one = ["As", "Ah"]
    hole_two = ["Ks", "Kh"]

    winner, score_one, score_two = compare_hands(hole_one, hole_two, board)
    assert winner == "p1"
    assert score_one < score_two


def test_public_state_matches_schema_and_hides_ai_cards() -> None:
    engine = Engine(players=(HUMAN_PLAYER_ID, "p2"))
    engine.new_hand(seed=7)
    engine.deal_flop()
    engine.deal_turn()
    engine.deal_river()

    public_state = engine.to_public_state(viewer=HUMAN_PLAYER_ID)
    parsed = GameStatePublic.parse_obj(public_state)
    assert parsed.hand == engine.hole_cards[HUMAN_PLAYER_ID]
    assert public_state["player_hand"] == engine.hole_cards[HUMAN_PLAYER_ID]

    ai_view = engine.to_public_state(viewer="observer")
    parsed_ai = GameStatePublic.parse_obj(ai_view)
    assert parsed_ai.hand is None
    assert "player_hand" not in ai_view


def test_busted_player_not_dealt_next_hand() -> None:
    engine = Engine(players=(HUMAN_PLAYER_ID, "p2", "p3"))
    engine.new_hand(seed=11)
    engine.betting.stacks = {HUMAN_PLAYER_ID: 0, "p2": 1000, "p3": 1000}

    engine.start_next_hand(seed=12)

    assert engine.hole_cards[HUMAN_PLAYER_ID] == []
    assert HUMAN_PLAYER_ID not in engine.betting.players
    assert set(engine.betting.players) == {"p2", "p3"}
    assert HUMAN_PLAYER_ID not in engine.betting.active_players()


def test_single_remaining_player_ends_hand_without_readding_busted_seat() -> None:
    engine = Engine(players=(HUMAN_PLAYER_ID, "p2"))
    engine.new_hand(seed=13)
    engine.betting.stacks = {HUMAN_PLAYER_ID: 1000, "p2": 0}

    engine.start_next_hand(seed=14)

    assert engine.betting.players == (HUMAN_PLAYER_ID,)
    assert engine.betting.hand_over is True
    assert engine.betting.current_player is None
    assert engine.hole_cards["p2"] == []
