import pytest

from backend.poker.engine import AI_PLAYER_ID, Engine, HUMAN_PLAYER_ID
from backend.schemas import Action, ActionType, Street


def _reach_flop(engine: Engine) -> None:
    engine.step(Action(action=ActionType.CALL), player_id=HUMAN_PLAYER_ID)
    engine.step(Action(action=ActionType.CHECK), player_id=AI_PLAYER_ID)


def test_check_check_advances_to_flop() -> None:
    engine = Engine(players=(HUMAN_PLAYER_ID, AI_PLAYER_ID))
    engine.new_hand(seed=1)

    _reach_flop(engine)
    assert engine.street == Street.FLOP
    assert len(engine.board) == 3


def test_bet_call_advances_to_turn() -> None:
    engine = Engine(players=(HUMAN_PLAYER_ID, AI_PLAYER_ID))
    engine.new_hand(seed=2)
    _reach_flop(engine)

    engine.step(Action(action=ActionType.RAISE, amount=20), player_id=AI_PLAYER_ID)
    engine.step(Action(action=ActionType.CALL), player_id=HUMAN_PLAYER_ID)

    assert engine.street == Street.TURN
    assert len(engine.board) == 4


def test_bet_fold_ends_hand() -> None:
    engine = Engine(players=(HUMAN_PLAYER_ID, AI_PLAYER_ID))
    engine.new_hand(seed=3)
    _reach_flop(engine)

    engine.step(Action(action=ActionType.RAISE, amount=20), player_id=AI_PLAYER_ID)
    engine.step(Action(action=ActionType.FOLD), player_id=HUMAN_PLAYER_ID)

    assert engine.street == Street.SHOWDOWN
    assert engine.betting.hand_over is True
    assert engine.betting.pot == 0
    assert engine.betting.stacks[AI_PLAYER_ID] == 1010
    assert sum(engine.betting.stacks.values()) == 2000


def test_raise_below_minimum_is_invalid() -> None:
    engine = Engine(players=(HUMAN_PLAYER_ID, AI_PLAYER_ID))
    engine.new_hand(seed=4)

    with pytest.raises(ValueError):
        engine.step(Action(action=ActionType.RAISE, amount=15), player_id=HUMAN_PLAYER_ID)


def test_raise_above_stack_is_invalid() -> None:
    engine = Engine(players=(HUMAN_PLAYER_ID, AI_PLAYER_ID))
    engine.new_hand(seed=5)

    with pytest.raises(ValueError):
        engine.step(Action(action=ActionType.RAISE, amount=2000), player_id=HUMAN_PLAYER_ID)


def test_turns_alternate_and_round_closes() -> None:
    engine = Engine(players=(HUMAN_PLAYER_ID, AI_PLAYER_ID))
    engine.new_hand(seed=6)

    assert engine.betting.current_player == HUMAN_PLAYER_ID
    engine.step(Action(action=ActionType.CALL), player_id=HUMAN_PLAYER_ID)
    assert engine.betting.current_player == AI_PLAYER_ID
    engine.step(Action(action=ActionType.CHECK), player_id=AI_PLAYER_ID)
    assert engine.street == Street.FLOP
    assert engine.betting.current_player == AI_PLAYER_ID


def test_showdown_payout_correct() -> None:
    engine = Engine(players=(HUMAN_PLAYER_ID, AI_PLAYER_ID))
    engine.new_hand(seed=7)

    engine.hole_cards = {
        HUMAN_PLAYER_ID: ["As", "Ah"],
        AI_PLAYER_ID: ["Ks", "Kh"],
    }
    engine.board = ["2c", "7d", "9h", "Jc", "Qd"]
    engine.street = Street.RIVER
    engine.betting.pot = 20
    engine.betting.stacks = {HUMAN_PLAYER_ID: 990, AI_PLAYER_ID: 990}

    engine.resolve_showdown()
    assert engine.betting.pot == 0
    assert engine.betting.stacks[HUMAN_PLAYER_ID] == 1010
    assert sum(engine.betting.stacks.values()) == 2000
