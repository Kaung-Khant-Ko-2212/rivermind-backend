import json
import logging
import time
import asyncio
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError

from .config import AppConfig
from .logging_setup import configure_logging
from .schemas import (
    Action,
    ActionType,
    ClientMessage,
    ErrorMessage,
    EventMessage,
    EventType,
    GameStatePublic,
    ServerMessage,
    format_validation_error,
)
from .ai.policy import get_ai_action
from .session_store import SEAT_ORDER, SessionStore
from .training.replay_buffer import ReplayBuffer
from .member2.bucketing import compute_infoset_id


configure_logging()
logger = logging.getLogger("backend.websocket")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
store = SessionStore()
config = AppConfig.from_env()
replay_buffer: Optional[ReplayBuffer] = (
    ReplayBuffer(capacity=config.replay_capacity) if config.replay_enabled else None
)
TRACE_ENABLED = config.game_trace
TURN_DELAY_SECONDS = config.ai_turn_delay_ms / 1000.0


class CreateTableRequest(BaseModel):
    user_key: Optional[str] = None


class JoinTableRequest(BaseModel):
    user_key: Optional[str] = None


class StartTableRequest(BaseModel):
    player_id: str


def _trace(session_id: str, message: str) -> None:
    if not TRACE_ENABLED:
        return
    print(f"[GAME][session={session_id}] {message}", flush=True)


async def _sleep_between_turns(session_id: str, reason: str) -> None:
    if TURN_DELAY_SECONDS <= 0:
        return
    _trace(
        session_id,
        f"TURN_DELAY reason={reason} ms={int(TURN_DELAY_SECONDS * 1000)}",
    )
    await asyncio.sleep(TURN_DELAY_SECONDS)


def _audit_chips(session) -> None:
    stacks = session.engine.betting.stacks
    total_chips = sum(stacks.values())
    expected_total = session.engine.betting.starting_stack * len(session.engine.players)
    if total_chips != expected_total:
        logger.warning(
            "Chip audit mismatch session_id=%s total=%s expected=%s stacks=%s",
            session.session_id,
            total_chips,
            expected_total,
            stacks,
        )
        _trace(
            session.session_id,
            f"CHIP_AUDIT_MISMATCH total={total_chips} expected={expected_total} stacks={stacks}",
        )
        return
    _trace(
        session.session_id,
        f"CHIP_AUDIT_OK total={total_chips} stacks={stacks}",
    )


@app.get("/health")
async def healthcheck() -> Dict[str, str]:
    return {"status": "ok"}


def _table_status_payload(session) -> Dict[str, Any]:
    return {
        "table_id": session.session_id,
        "mode": session.mode,
        "started": session.started,
        "ended": session.table_ended,
        "winners": list(session.table_winners),
        "host_player_id": session.host_player_id,
        "joined_players": sorted(session.joined_players),
        "seats": [
            {
                "seat": seat,
                "joined": seat in session.joined_players,
                "connected": seat in session.human_players,
                "is_host": seat == session.host_player_id,
            }
            for seat in SEAT_ORDER
        ],
    }


@app.post("/tables/create")
async def create_table(payload: CreateTableRequest) -> Dict[str, Any]:
    session = store.create_multiplayer_table(host_user_key=payload.user_key)
    _trace(session.session_id, "TABLE_CREATED mode=multi host=p1")
    return {
        "table_id": session.session_id,
        "player_id": session.host_player_id,
        "status": _table_status_payload(session),
    }


@app.get("/tables/{table_id}")
async def get_table_status(table_id: str) -> Dict[str, Any]:
    session = store.get(table_id)
    if not session or session.mode != "multi":
        raise HTTPException(status_code=404, detail="Table not found")
    return _table_status_payload(session)


@app.post("/tables/{table_id}/join")
async def join_table(table_id: str, payload: JoinTableRequest) -> Dict[str, Any]:
    try:
        seat = store.join_multiplayer_table(table_id, user_key=payload.user_key)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    session = store.get(table_id)
    if not session:
        raise HTTPException(status_code=404, detail="Table not found")
    _trace(session.session_id, f"TABLE_JOIN seat={seat} joined={sorted(session.joined_players)}")
    return {
        "table_id": session.session_id,
        "player_id": seat,
        "status": _table_status_payload(session),
    }


@app.post("/tables/{table_id}/start")
async def start_table(table_id: str, payload: StartTableRequest) -> Dict[str, Any]:
    try:
        session = store.start_multiplayer_table(table_id, requester_player_id=payload.player_id)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    _trace(session.session_id, f"TABLE_STARTED by={payload.player_id}")
    return _table_status_payload(session)


async def send_server_message(websocket: WebSocket, message: ServerMessage) -> None:
    await websocket.send_json(message.dict(by_alias=True))


def _parse_json_payload(raw_text: str) -> Dict[str, Any]:
    payload = json.loads(raw_text)
    if not isinstance(payload, dict):
        raise ValueError("Message must be a JSON object")
    return payload


async def _advance_to_next_hand(session) -> None:
    session.awaiting_hand_continue = False
    session.engine.start_next_hand()
    _trace(
        session.session_id,
        f"NEXT_HAND_STARTED button={session.engine.button_player} current={session.engine.betting.current_player}",
    )
    await _broadcast_new_hand(session)
    events = session.engine.drain_events()
    await _broadcast_events(session, events)
    await _broadcast_state(session)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()

    session_id = websocket.query_params.get("session_id")
    player_id = websocket.query_params.get("player_id") or "p1"
    mode = (websocket.query_params.get("mode") or "single").strip().lower()
    if mode not in {"single", "multi"}:
        mode = "single"
    existing_session = store.get(session_id) if session_id else None
    if existing_session and existing_session.mode == "multi":
        mode = "multi"

    if mode == "multi":
        if not session_id:
            await send_server_message(
                websocket,
                ServerMessage(
                    type="ERROR",
                    payload=ErrorMessage(
                        code="MISSING_TABLE_ID",
                        message="Missing table_id (session_id) for multiplayer",
                    ),
                ),
            )
            await websocket.close()
            return
        session = existing_session or store.get(session_id)
        created = False
        if not session:
            await send_server_message(
                websocket,
                ServerMessage(
                    type="ERROR",
                    payload=ErrorMessage(
                        code="TABLE_NOT_FOUND",
                        message="Table not found",
                        details=[f"session_id={session_id}"],
                    ),
                ),
            )
            await websocket.close()
            return
        if session.mode != "multi":
            await send_server_message(
                websocket,
                ServerMessage(
                    type="ERROR",
                    payload=ErrorMessage(
                        code="INVALID_TABLE_MODE",
                        message="session_id does not reference a multiplayer table",
                    ),
                ),
            )
            await websocket.close()
            return
    else:
        if session_id and session_id.upper().startswith("TBL-"):
            await send_server_message(
                websocket,
                ServerMessage(
                    type="ERROR",
                    payload=ErrorMessage(
                        code="INVALID_SINGLE_SESSION_ID",
                        message="Table-style session_id requires multiplayer mode",
                        details=[f"session_id={session_id}", "Use mode=multi for TBL-* ids"],
                    ),
                ),
            )
            await websocket.close()
            return
        session, created = store.get_or_create(session_id, mode="single")

    logger.info(
        "WebSocket connected session_id=%s created=%s",
        session.session_id,
        created,
    )
    _trace(
        session.session_id,
        f"CONNECT player={player_id} created={created}",
    )

    if player_id not in session.engine.players:
        await send_server_message(
            websocket,
            ServerMessage(
                type="ERROR",
                payload=ErrorMessage(
                    code="INVALID_PLAYER_ID",
                    message="Invalid player_id",
                    details=[f"{player_id} is not a valid seat"],
                ),
            ),
        )
        await websocket.close()
        return

    if session.mode == "multi":
        if player_id not in session.joined_players:
            await send_server_message(
                websocket,
                ServerMessage(
                    type="ERROR",
                    payload=ErrorMessage(
                        code="SEAT_NOT_JOINED",
                        message="Seat is not part of this table",
                        details=[f"{player_id} has not joined table {session.session_id}"],
                    ),
                ),
            )
            await websocket.close()
            return
        if not session.started:
            await send_server_message(
                websocket,
                ServerMessage(
                    type="ERROR",
                    payload=ErrorMessage(
                        code="TABLE_NOT_STARTED",
                        message="Host has not started this table yet",
                    ),
                ),
            )
            await websocket.close()
            return

    store.register_socket(session.session_id, player_id, websocket)
    _trace(
        session.session_id,
        f"HUMANS now={sorted(session.human_players)}",
    )

    if session.mode == "single" and (created or not session.started):
        session.engine.new_hand()
        session.started = True
        _trace(
            session.session_id,
            f"NEW_HAND street={session.engine.street.value} button={session.engine.button_player} current={session.engine.betting.current_player} pot={session.engine.betting.pot}",
        )

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
                        payload=ErrorMessage(
                            code="INVALID_JSON",
                            message="Invalid JSON",
                            details=[str(exc)],
                        ),
                    ),
                )
                continue

            store.touch(session.session_id)

            if session.mode == "multi" and session.table_ended:
                await send_server_message(
                    websocket,
                    ServerMessage(
                        type="ERROR",
                        payload=ErrorMessage(
                            code="TABLE_ENDED",
                            message="This table has ended",
                            details=["Create a new table to continue playing"],
                        ),
                    ),
                )
                continue

            message_type = str(payload.get("type") or "").strip().upper()
            if message_type == "CONTINUE":
                if session.mode == "multi" and session.table_ended:
                    await send_server_message(
                        websocket,
                        ServerMessage(
                            type="ERROR",
                            payload=ErrorMessage(
                                code="TABLE_ENDED",
                                message="This table has ended",
                                details=["Create a new table to continue playing"],
                            ),
                        ),
                    )
                    continue
                if not session.engine.betting.hand_over:
                    await send_server_message(
                        websocket,
                        ServerMessage(
                            type="ERROR",
                            payload=ErrorMessage(
                                code="HAND_NOT_OVER",
                                message="Cannot continue yet",
                                details=["The current hand is still in progress"],
                            ),
                        ),
                    )
                    continue
                if not session.awaiting_hand_continue:
                    await send_server_message(
                        websocket,
                        ServerMessage(
                            type="ERROR",
                            payload=ErrorMessage(
                                code="HAND_CONTINUE_NOT_READY",
                                message="Hand is not waiting for continue",
                            ),
                        ),
                    )
                    continue
                _trace(session.session_id, f"HAND_CONTINUE by={player_id}")
                await _advance_to_next_hand(session)
                await _run_ai_turns(session, replay_buffer)
                continue

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
                            code="NOT_YOUR_TURN",
                            message="Not your turn",
                            details=[
                                f"Current player is {acting_player}",
                            ],
                        ),
                    ),
                )
                continue
            _trace(
                session.session_id,
                f"HUMAN_MOVE player={acting_player} action={action.action.value} amount={action.amount} street={session.engine.street.value}",
            )
            try:
                action_street = session.engine.street.value
                session.engine.step(action, player_id=acting_player)
                hand_ended_from_move = session.engine.betting.hand_over
            except ValueError as exc:
                await send_server_message(
                    websocket,
                    ServerMessage(
                        type="ERROR",
                        payload=ErrorMessage(
                            code="INVALID_ACTION",
                            message="Invalid action",
                            details=[str(exc)],
                        ),
                    ),
                )
                _trace(
                    session.session_id,
                    f"HUMAN_MOVE_REJECTED player={acting_player} action={action.action.value} amount={action.amount} error={exc}",
                )
                continue

            _record_experience(
                replay_buffer, session.session_id, acting_player, action, action_street, session.engine
            )
            _trace(
                session.session_id,
                f"POST_HUMAN_MOVE street={session.engine.street.value} pot={session.engine.betting.pot} current={session.engine.betting.current_player} legal={[a.value for a in session.engine.betting.legal_actions()]}",
            )

            await _broadcast_update(session)
            if not hand_ended_from_move:
                await _sleep_between_turns(session.session_id, "human_move")
            await _run_ai_turns(session, replay_buffer)
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected session_id=%s", session.session_id)
        _trace(session.session_id, f"DISCONNECT player={player_id}")
        store.remove_socket(session.session_id, player_id)
    except RuntimeError as exc:
        # Starlette can surface closed sockets as RuntimeError once send() has failed.
        if "WebSocket is not connected" in str(exc):
            logger.info("WebSocket disconnected session_id=%s", session.session_id)
            _trace(session.session_id, f"DISCONNECT player={player_id} reason=not_connected")
            store.remove_socket(session.session_id, player_id)
            return
        logger.exception("WebSocket runtime error session_id=%s", session.session_id)
        _trace(
            session.session_id,
            f"ERROR player={player_id} runtime websocket exception",
        )
        store.remove_socket(session.session_id, player_id)
    except Exception:
        logger.exception("WebSocket error session_id=%s", session.session_id)
        _trace(session.session_id, f"ERROR player={player_id} unexpected websocket exception")
        store.remove_socket(session.session_id, player_id)


async def _broadcast_update(session) -> None:
    funded_players = []
    table_should_end = False
    if session.engine.betting.hand_over:
        funded_players = [
            player_id
            for player_id, chips in session.engine.betting.stacks.items()
            if chips > 0
        ]
        table_should_end = session.mode == "multi" and len(funded_players) <= 1
        if not table_should_end and not session.awaiting_hand_continue:
            session.awaiting_hand_continue = True
            _trace(
                session.session_id,
                "HAND_WAITING_FOR_CONTINUE",
            )

    events = session.engine.drain_events()
    await _broadcast_events(session, events)
    await _broadcast_state(session)

    if session.engine.betting.hand_over:
        _audit_chips(session)
        if table_should_end:
            session.awaiting_hand_continue = False
            if not session.table_ended:
                session.table_ended = True
                session.table_winners = list(funded_players)
                winner_desc = funded_players[0] if funded_players else "none"
                _trace(
                    session.session_id,
                    f"TABLE_END winner={winner_desc} stacks={session.engine.betting.stacks}",
                )
                end_event = EventMessage(
                    event=EventType.TABLE_END,
                    data={
                        "winners": list(funded_players),
                        "stacks": dict(session.engine.betting.stacks),
                    },
                )
                await _broadcast_events(session, [end_event])
            await _broadcast_state(session)
            return
        return


async def _broadcast_events(session, events) -> None:
    if not events:
        return
    for event in events:
        _trace(
            session.session_id,
            f"EVENT name={event.event.value} data={event.data}",
        )
    for player_id, socket in list(session.player_sockets.items()):
        for event in events:
            delivered = await _safe_send(socket, ServerMessage(type="EVENT", payload=event))
            if not delivered:
                store.remove_socket(session.session_id, player_id)
                _trace(session.session_id, f"DROP_SOCKET player={player_id} reason=send_failed")
                break


async def _broadcast_new_hand(session) -> None:
    _trace(
        session.session_id,
        f"NEW_HAND_BROADCAST button={session.engine.button_player} current={session.engine.betting.current_player} pot={session.engine.betting.pot}",
    )
    for player_id, socket in list(session.player_sockets.items()):
        new_hand_event = EventMessage(
            event=EventType.NEW_HAND,
            data={
                "player_hand": session.engine.hole_cards.get(player_id, []),
                "button": session.engine.button_player,
                "small_blind_player": session.engine.sb_player,
                "big_blind_player": session.engine.bb_player,
                "current_player": session.engine.betting.current_player,
            },
        )
        delivered = await _safe_send(socket, ServerMessage(type="EVENT", payload=new_hand_event))
        if not delivered:
            store.remove_socket(session.session_id, player_id)
            _trace(session.session_id, f"DROP_SOCKET player={player_id} reason=send_failed")


async def _broadcast_state(session) -> None:
    _trace(
        session.session_id,
        f"STATE street={session.engine.street.value} pot={session.engine.betting.pot} current={session.engine.betting.current_player} legal={[a.value for a in session.engine.betting.legal_actions()]}",
    )
    for player_id, socket in list(session.player_sockets.items()):
        state_payload = session.engine.to_public_state(
            viewer=player_id, session_id=session.session_id
        )
        state_payload["awaiting_hand_continue"] = bool(session.awaiting_hand_continue)
        updated_state = GameStatePublic.parse_obj(
            state_payload
        )
        delivered = await _safe_send(socket, ServerMessage(type="STATE", payload=updated_state))
        if not delivered:
            store.remove_socket(session.session_id, player_id)
            _trace(session.session_id, f"DROP_SOCKET player={player_id} reason=send_failed")


async def _safe_send(websocket: WebSocket, message: ServerMessage) -> bool:
    try:
        await send_server_message(websocket, message)
        return True
    except WebSocketDisconnect:
        return False
    except RuntimeError as exc:
        # Raised if the websocket has already transitioned to a closed state.
        if "WebSocket is not connected" in str(exc) or 'Cannot call "send" once a close message has been sent.' in str(exc):
            return False
        return False
    except OSError:
        return False
    except Exception:
        return False


async def _run_ai_turns(session, buffer: Optional[ReplayBuffer]) -> None:
    if session.mode == "multi" and session.table_ended:
        return

    def _fallback_ai_action(engine) -> Optional[Action]:
        legal = engine.betting.legal_actions()
        for candidate in (ActionType.CHECK, ActionType.CALL, ActionType.FOLD, ActionType.RAISE):
            if candidate in legal:
                if candidate == ActionType.RAISE:
                    return Action(action=ActionType.RAISE, amount=engine.betting.min_raise_to())
                return Action(action=candidate)
        return None

    def _find_next_eligible_player() -> Optional[str]:
        betting = session.engine.betting
        current = betting.current_player

        if current and current not in betting.folded_players and current not in betting.all_in_players:
            return current

        if current:
            try:
                candidate = betting._next_player(current)  # type: ignore[attr-defined]
            except Exception:
                candidate = None
            if candidate:
                return candidate

        for player in session.engine.players:
            if player in betting.pending_players and player not in betting.folded_players and player not in betting.all_in_players:
                return player
        return None

    def _advance_without_actor() -> bool:
        # Repair invalid actor seat first.
        next_player = _find_next_eligible_player()
        if next_player:
            if session.engine.betting.current_player != next_player:
                _trace(
                    session.session_id,
                    f"TURN_REPAIRED previous={session.engine.betting.current_player} next={next_player}",
                )
                session.engine.betting.current_player = next_player
                return True
            return False

        # No eligible actor: run out remaining streets until showdown.
        street = session.engine.street.value
        if street == "preflop":
            session.engine.deal_flop()
            session.engine.betting.start_new_round(
                first_to_act=session.engine._first_to_act_postflop()  # type: ignore[attr-defined]
            )
            _trace(session.session_id, "AUTO_PROGRESS preflop->flop (no eligible actor)")
            return True
        if street == "flop":
            session.engine.deal_turn()
            session.engine.betting.start_new_round(
                first_to_act=session.engine._first_to_act_postflop()  # type: ignore[attr-defined]
            )
            _trace(session.session_id, "AUTO_PROGRESS flop->turn (no eligible actor)")
            return True
        if street == "turn":
            session.engine.deal_river()
            session.engine.betting.start_new_round(
                first_to_act=session.engine._first_to_act_postflop()  # type: ignore[attr-defined]
            )
            _trace(session.session_id, "AUTO_PROGRESS turn->river (no eligible actor)")
            return True
        if street == "river":
            session.engine.resolve_showdown()
            _trace(session.session_id, "AUTO_PROGRESS river->showdown (no eligible actor)")
            return True
        return False

    max_actions = max(10, len(session.engine.players) * 4)
    actions_taken = 0
    while not session.engine.betting.hand_over and actions_taken < max_actions:
        if _advance_without_actor():
            hand_ended_from_auto = session.engine.betting.hand_over
            await _broadcast_update(session)
            if not hand_ended_from_auto:
                await _sleep_between_turns(session.session_id, "auto_progress")
            continue

        if not session.engine.betting.current_player:
            break
        human_controlled_players = set(session.human_players)
        if session.mode == "multi":
            human_controlled_players.update(session.joined_players)
        if session.engine.betting.current_player in human_controlled_players:
            break

        ai_state = session.engine.to_ai_state()
        ai_player = session.engine.betting.current_player
        try:
            ai_action = get_ai_action(ai_state)
        except Exception as exc:
            logger.warning(
                "AI action generation failed session_id=%s player=%s error=%s",
                session.session_id,
                ai_player,
                exc,
            )
            _trace(
                session.session_id,
                f"AI_ACTION_BUILD_FAILED player={ai_player} error={exc}",
            )
            ai_action = _fallback_ai_action(session.engine)
            if ai_action is None:
                _trace(
                    session.session_id,
                    f"AI_ACTION_BUILD_FAILED_NO_FALLBACK player={ai_player}",
                )
                break
        _trace(
            session.session_id,
            f"AI_MOVE player={ai_player} action={ai_action.action.value} amount={ai_action.amount} street={session.engine.street.value}",
        )
        try:
            ai_street = session.engine.street.value
            session.engine.step(ai_action, player_id=ai_player)
            hand_ended_from_ai = session.engine.betting.hand_over
        except ValueError as exc:
            logger.warning(
                "AI action rejected session_id=%s player=%s action=%s error=%s",
                session.session_id,
                ai_player,
                ai_action.dict(),
                exc,
            )
            _trace(
                session.session_id,
                f"AI_MOVE_REJECTED player={ai_player} action={ai_action.action.value} amount={ai_action.amount} error={exc}",
            )
            fallback = _fallback_ai_action(session.engine)
            if fallback is None:
                _trace(
                    session.session_id,
                    f"AI_MOVE_REJECTED_NO_FALLBACK player={ai_player}",
                )
                break
            try:
                ai_street = session.engine.street.value
                session.engine.step(fallback, player_id=ai_player)
                hand_ended_from_ai = session.engine.betting.hand_over
            except ValueError:
                _trace(
                    session.session_id,
                    f"AI_FALLBACK_REJECTED player={ai_player} action={fallback.action.value} amount={fallback.amount}",
                )
                break
            else:
                _trace(
                    session.session_id,
                    f"AI_FALLBACK_APPLIED player={ai_player} action={fallback.action.value} amount={fallback.amount}",
                )
                _record_experience(
                    buffer,
                    session.session_id,
                    ai_player,
                    fallback,
                    ai_street,
                    session.engine,
                )
                await _broadcast_update(session)
                actions_taken += 1
                if not hand_ended_from_ai:
                    await _sleep_between_turns(session.session_id, "ai_move_fallback")
        else:
            _trace(
                session.session_id,
                f"AI_MOVE_APPLIED player={ai_player} action={ai_action.action.value} amount={ai_action.amount} next={session.engine.betting.current_player}",
            )
            _record_experience(
                buffer,
                session.session_id,
                ai_player,
                ai_action,
                ai_street,
                session.engine,
            )
            await _broadcast_update(session)
            actions_taken += 1
            if not hand_ended_from_ai:
                await _sleep_between_turns(session.session_id, "ai_move")


def _record_experience(
    buffer: Optional[ReplayBuffer],
    session_id: str,
    player_id: str,
    action: Action,
    street: str,
    engine,
) -> None:
    if buffer is None:
        return
    
    # Get state information for bucketing
    hole_cards = engine.hole_cards.get(player_id, [])
    board = list(engine.board)
    action_history = engine.betting.action_history[:-1] if engine.betting.action_history else []  # Exclude current action
    pot = engine.betting.pot
    player_stack = engine.betting.stacks.get(player_id, 0)
    big_blind = engine.betting.big_blind
    
    # Compute bucketed infoset ID
    infoset_id = compute_infoset_id(
        player_id=player_id,
        hole_cards=hole_cards,
        board=board,
        street=street,
        action_history=action_history,
        pot=pot,
        player_stack=player_stack,
        big_blind=big_blind,
    )
    
    buffer.add(
        {
            "timestamp": time.time(),
            "street": street,
            "player_to_act": player_id,
            "infoset_id": infoset_id,
            "action_taken": action.action.value,
            "amount": action.amount,
            "outcome": None,
        }
    )
