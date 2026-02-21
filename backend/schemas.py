from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, ValidationError, confloat, conint, root_validator, validator


class ActionType(str, Enum):
    CHECK = "check"
    CALL = "call"
    FOLD = "fold"
    RAISE = "raise"


class Street(str, Enum):
    PREFLOP = "preflop"
    FLOP = "flop"
    TURN = "turn"
    RIVER = "river"
    SHOWDOWN = "showdown"


class EventType(str, Enum):
    DEAL_HOLE = "DEAL_HOLE"
    DEAL_FLOP = "DEAL_FLOP"
    DEAL_TURN = "DEAL_TURN"
    DEAL_RIVER = "DEAL_RIVER"
    SHOWDOWN = "SHOWDOWN"
    HAND_END = "HAND_END"
    NEW_HAND = "NEW_HAND"
    TABLE_END = "TABLE_END"


def _validate_action_amount(action: ActionType, amount: Optional[int]) -> None:
    if action == ActionType.RAISE:
        if amount is None:
            raise ValueError("amount is required for raise")
    else:
        if amount is not None:
            raise ValueError("amount is only valid for raise")


class Action(BaseModel):
    action: ActionType
    amount: Optional[conint(ge=1)] = None

    @root_validator(skip_on_failure=True)
    def validate_amount(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        _validate_action_amount(values.get("action"), values.get("amount"))
        return values


class ClientMessage(BaseModel):
    type: str
    action: ActionType = Field(alias="val")
    amount: Optional[conint(ge=1)] = None

    @validator("action", pre=True)
    def normalize_deal_action(cls, value: Any) -> Any:
        if isinstance(value, str) and value.lower() == "deal":
            return ActionType.CALL
        return value

    @root_validator(skip_on_failure=True)
    def validate_move(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        message_type = values.get("type")
        if message_type != "MOVE":
            raise ValueError("type must be MOVE for client actions")
        _validate_action_amount(values.get("action"), values.get("amount"))
        return values

    class Config:
        allow_population_by_field_name = True


class ActionRecord(BaseModel):
    player_id: str
    action: Action


class GameStatePublic(BaseModel):
    session_id: Optional[str] = None
    street: Street
    pot: conint(ge=0) = 0
    community_cards: List[str] = Field(default_factory=list)
    hand: Optional[List[str]] = Field(default=None, alias="player_hand")
    revealed_hands: Optional[Dict[str, List[str]]] = None
    folded_players: List[str] = Field(default_factory=list)
    stacks: Dict[str, conint(ge=0)]
    bets: Dict[str, conint(ge=0)]
    button_player: Optional[str] = None
    small_blind_player: Optional[str] = None
    big_blind_player: Optional[str] = None
    current_player: Optional[str] = None
    legal_actions: List[ActionType] = Field(default_factory=list)
    to_call: Optional[conint(ge=0)] = None
    min_raise_to: Optional[conint(ge=0)] = None
    max_raise_to: Optional[conint(ge=0)] = None
    action_history: List[ActionRecord] = Field(default_factory=list, alias="history")
    hand_strength_pct: Optional[confloat(ge=0, le=100)] = None
    hand_strength_label: Optional[str] = None
    hand_category_probs: Optional[Dict[str, confloat(ge=0, le=100)]] = None
    awaiting_hand_continue: bool = False

    class Config:
        allow_population_by_field_name = True


class ErrorMessage(BaseModel):
    code: Optional[str] = None
    message: str
    details: Optional[List[str]] = None


class EventMessage(BaseModel):
    event: EventType
    data: Optional[Dict[str, Any]] = None


ServerPayload = Union[GameStatePublic, ErrorMessage, EventMessage]


class ServerMessage(BaseModel):
    type: str
    payload: ServerPayload

    @root_validator(skip_on_failure=True)
    def validate_payload(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        message_type = values.get("type")
        payload = values.get("payload")
        if message_type == "STATE" and not isinstance(payload, GameStatePublic):
            raise ValueError("payload must be GameStatePublic for STATE")
        if message_type == "ERROR" and not isinstance(payload, ErrorMessage):
            raise ValueError("payload must be ErrorMessage for ERROR")
        if message_type == "EVENT" and not isinstance(payload, EventMessage):
            raise ValueError("payload must be EventMessage for EVENT")
        if message_type not in {"STATE", "ERROR", "EVENT"}:
            raise ValueError("type must be STATE, ERROR, or EVENT for server messages")
        return values


def format_validation_error(error: ValidationError) -> ErrorMessage:
    details = []
    for entry in error.errors():
        location = ".".join(str(part) for part in entry.get("loc", []))
        message = entry.get("msg", "Invalid value")
        if location:
            details.append(f"{location}: {message}")
        else:
            details.append(message)
    return ErrorMessage(code="VALIDATION_ERROR", message="Invalid message", details=details or None)
