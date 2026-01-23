import json
import logging
import time
from typing import Any, Dict, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from .config import AppConfig
from .logging_setup import configure_logging
from .schemas import (
    Action,
    ClientMessage,
    ErrorMessage,
    EventMessage,
    EventType,
    GameStatePublic,
    ServerMessage,
    format_validation_error,
)
from .ai.policy import get_ai_action
from .session_store import SessionStore
from .training.replay_buffer import ReplayBuffer


configure_logging()
logger = logging.getLogger("backend.websocket")

app = FastAPI()
store = SessionStore()
config = AppConfig.from_env()
replay_buffer: Optional[ReplayBuffer] = (
    ReplayBuffer(capacity=config.replay_capacity) if config.replay_enabled else None
)


async def send_server_message(websocket: WebSocket, message: ServerMessage) -> None:
    await websocket.send_json(message.dict(by_alias=True))


def _parse_json_payload(raw_text: str) -> Dict[str, Any]:
    payload = json.loads(raw_text)
    if not isinstance(payload, dict):
        raise ValueError("Message must be a JSON object")
    return payload


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()

    session_id = websocket.query_params.get("session_id")
    player_id = websocket.query_params.get("player_id") or "p1"
    session, created = store.get_or_create(session_id)
    logger.info(
        "WebSocket connected session_id=%s created=%s",
        session.session_id,
        created,
    )

    if player_id not in session.engine.players:
        await send_server_message(
            websocket,
            ServerMessage(
                type="ERROR",
                payload=ErrorMessage(
                    message="Invalid player_id",
                    details=[f"{player_id} is not a valid seat"],
                ),
            ),
        )
        await websocket.close()
        return

    store.register_socket(session.session_id, player_id, websocket)

    if created:
        session.engine.new_hand()

    await _broadcast_update(session)
    await _run_ai_turns(session, replay_buffer)

    try:
        while True:
            raw_text = await websocket.receive_text()
            try:
                payload = _parse_json_payload(raw_text)
            except (json.JSONDecodeError, ValueError) as exc:
                await send_server_message(
                    websocket,
                    ServerMessage(
                        type="ERROR",
                        payload=ErrorMessage(message="Invalid JSON", details=[str(exc)]),
                    ),
                )
                continue

            store.touch(session.session_id)

            try:
                client_message = ClientMessage.parse_obj(payload)
            except ValidationError as exc:
                await send_server_message(
                    websocket,
                    ServerMessage(
                        type="ERROR",
                        payload=format_validation_error(exc),
                    ),
                )
                continue

            action = Action(action=client_message.action, amount=client_message.amount)
            acting_player = session.engine.betting.current_player or "p1"
            if acting_player != player_id:
                await send_server_message(
                    websocket,
                    ServerMessage(
                        type="ERROR",
                        payload=ErrorMessage(
                            message="Not your turn",
                            details=[
                                f"Current player is {acting_player}",
                            ],
                        ),
                    ),
                )
                continue
            try:
                action_street = session.engine.street.value
                session.engine.step(action, player_id=acting_player)
            except ValueError as exc:
                await send_server_message(
                    websocket,
                    ServerMessage(
                        type="ERROR",
                        payload=ErrorMessage(message="Invalid action", details=[str(exc)]),
                    ),
                )
                continue

            _record_experience(
                replay_buffer, session.session_id, acting_player, action, action_street
            )

            await _broadcast_update(session)
            await _run_ai_turns(session, replay_buffer)
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected session_id=%s", session.session_id)
        store.remove_socket(session.session_id, player_id)
    except Exception:
        logger.exception("WebSocket error session_id=%s", session.session_id)
        store.remove_socket(session.session_id, player_id)


async def _broadcast_update(session) -> None:
    events = session.engine.drain_events()
    await _broadcast_events(session, events)

    if session.engine.betting.hand_over:
        session.engine.start_next_hand()
        await _broadcast_new_hand(session)
        events = session.engine.drain_events()
        await _broadcast_events(session, events)

    await _broadcast_state(session)


async def _broadcast_events(session, events) -> None:
    if not events:
        return
    for player_id, socket in list(session.player_sockets.items()):
        for event in events:
            await _safe_send(socket, ServerMessage(type="EVENT", payload=event))


async def _broadcast_new_hand(session) -> None:
    for player_id, socket in list(session.player_sockets.items()):
        new_hand_event = EventMessage(
            event=EventType.NEW_HAND,
            data={
                "player_hand": session.engine.hole_cards.get(player_id, []),
                "button": session.engine.button_player,
            },
        )
        await _safe_send(socket, ServerMessage(type="EVENT", payload=new_hand_event))


async def _broadcast_state(session) -> None:
    for player_id, socket in list(session.player_sockets.items()):
        updated_state = GameStatePublic.parse_obj(
            session.engine.to_public_state(
                viewer=player_id, session_id=session.session_id
            )
        )
        await _safe_send(socket, ServerMessage(type="STATE", payload=updated_state))


async def _safe_send(websocket: WebSocket, message: ServerMessage) -> None:
    try:
        await send_server_message(websocket, message)
    except Exception:
        return


async def _run_ai_turns(session, buffer: Optional[ReplayBuffer]) -> None:
    max_actions = max(10, len(session.engine.players) * 4)
    actions_taken = 0
    while (
        not session.engine.betting.hand_over
        and session.engine.betting.current_player
        and session.engine.betting.current_player not in session.human_players
        and actions_taken < max_actions
    ):
        ai_state = session.engine.to_ai_state()
        ai_action = get_ai_action(ai_state)
        ai_player = session.engine.betting.current_player
        try:
            ai_street = session.engine.street.value
            session.engine.step(ai_action, player_id=ai_player)
        except ValueError:
            break
        else:
            _record_experience(
                buffer,
                session.session_id,
                ai_player,
                ai_action,
                ai_street,
            )
            await _broadcast_update(session)
            actions_taken += 1


def _record_experience(
    buffer: Optional[ReplayBuffer],
    session_id: str,
    player_id: str,
    action: Action,
    street: str,
) -> None:
    if buffer is None:
        return
    buffer.add(
        {
            "timestamp": time.time(),
            "street": street,
            "player_to_act": player_id,
            "infoset_id": f"{session_id}:{street}:{player_id}",
            "action_taken": action.action.value,
            "amount": action.amount,
            "outcome": None,
        }
    )
