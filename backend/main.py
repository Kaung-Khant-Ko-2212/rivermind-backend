import json
import logging
import time
import asyncio
import random
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
# Frontend hand intro sequence currently shows Dealer -> (2s) -> SB -> (2s) -> BB
# and then waits another 2s before actions resume.
HAND_INTRO_BLOCK_SECONDS = 6.0


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


def _start_hand_intro_block(session) -> None:
    session.hand_intro_block_until = time.time() + max(0.0, HAND_INTRO_BLOCK_SECONDS)
    _trace(
        session.session_id,
        f"HAND_INTRO_BLOCK start until={round(session.hand_intro_block_until, 3)} secs={HAND_INTRO_BLOCK_SECONDS}",
    )


def _hand_intro_wait_remaining(session) -> float:
    return max(0.0, float(getattr(session, "hand_intro_block_until", 0.0) or 0.0) - time.time())


async def _wait_for_hand_intro_if_needed(session) -> None:
    remaining = _hand_intro_wait_remaining(session)
    if remaining <= 0:
        return
    _trace(
        session.session_id,
        f"HAND_INTRO_WAIT ms={int(remaining * 1000)}",
    )
    await asyncio.sleep(remaining)


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


def _ensure_tendency_row(session, player_id: str) -> Dict[str, int]:
    stats = session.player_tendency_stats.get(player_id)
    if stats is None:
        stats = {
            "actions": 0,
            "folds": 0,
            "checks": 0,
            "calls": 0,
            "raises": 0,
            "preflop_actions": 0,
            "preflop_vpip_opportunities": 0,
            "preflop_vpip": 0,
            "facing_bet_samples": 0,
            "fold_vs_bet": 0,
            "call_vs_bet": 0,
            "raise_vs_bet": 0,
        }
        session.player_tendency_stats[player_id] = stats
    return stats


def _record_human_tendency(
    session,
    *,
    player_id: str,
    action: Action,
    street: str,
    to_call_before: int,
    legal_actions_before: list[ActionType],
) -> None:
    stats = _ensure_tendency_row(session, player_id)
    action_name = action.action.value

    stats["actions"] += 1
    if action_name == "fold":
        stats["folds"] += 1
    elif action_name == "check":
        stats["checks"] += 1
    elif action_name == "call":
        stats["calls"] += 1
    elif action_name == "raise":
        stats["raises"] += 1

    if street == "preflop":
        stats["preflop_actions"] += 1
        # VPIP opportunity approximated as having a voluntary decision preflop.
        # Excludes free checks in BB when to_call == 0.
        if to_call_before > 0 or ActionType.RAISE in legal_actions_before:
            stats["preflop_vpip_opportunities"] += 1
            if action.action in {ActionType.CALL, ActionType.RAISE}:
                stats["preflop_vpip"] += 1

    if to_call_before > 0:
        stats["facing_bet_samples"] += 1
        if action.action == ActionType.FOLD:
            stats["fold_vs_bet"] += 1
        elif action.action == ActionType.CALL:
            stats["call_vs_bet"] += 1
        elif action.action == ActionType.RAISE:
            stats["raise_vs_bet"] += 1


def _build_opponent_profile(session, ai_player: str) -> Optional[Dict[str, Any]]:
    relevant_rows: list[Dict[str, int]] = []
    for player_id, stats in session.player_tendency_stats.items():
        if player_id == ai_player:
            continue
        if not isinstance(stats, dict):
            continue
        relevant_rows.append(stats)

    if not relevant_rows:
        return None

    totals: Dict[str, int] = {}
    for row in relevant_rows:
        for key, value in row.items():
            totals[key] = totals.get(key, 0) + int(value or 0)

    actions = max(0, totals.get("actions", 0))
    facing_bet = max(0, totals.get("facing_bet_samples", 0))
    preflop_vpip_opp = max(0, totals.get("preflop_vpip_opportunities", 0))

    def safe_rate(num: int, den: int) -> float:
        return float(num) / float(den) if den > 0 else 0.0

    calls = totals.get("calls", 0)
    raises = totals.get("raises", 0)
    checks = totals.get("checks", 0)
    folds = totals.get("folds", 0)
    continue_vs_bet = totals.get("call_vs_bet", 0) + totals.get("raise_vs_bet", 0)

    profile = {
        "tracked_players": len(relevant_rows),
        "samples": actions,
        "facing_bet_samples": facing_bet,
        "fold_rate": safe_rate(folds, actions),
        "check_rate": safe_rate(checks, actions),
        "call_rate": safe_rate(calls, actions),
        "raise_rate": safe_rate(raises, actions),
        "aggression_rate": safe_rate(raises, max(1, calls + checks)),
        "fold_vs_bet_rate": safe_rate(totals.get("fold_vs_bet", 0), facing_bet),
        "continue_vs_bet_rate": safe_rate(continue_vs_bet, facing_bet),
        "raise_vs_bet_rate": safe_rate(totals.get("raise_vs_bet", 0), facing_bet),
        "vpip_samples": preflop_vpip_opp,
        "vpip_rate": safe_rate(totals.get("preflop_vpip", 0), preflop_vpip_opp),
    }
    return profile if profile["samples"] > 0 else None


def _raise_size_candidates(engine, player_id: str) -> list[int]:
    min_raise_to = engine.betting.min_raise_to()
    max_raise_to = engine.betting.max_raise_to(player_id)
    if max_raise_to < min_raise_to:
        return [max_raise_to]
    if max_raise_to == min_raise_to:
        return [min_raise_to]

    contribution = int(engine.betting.contributions.get(player_id, 0) or 0)
    to_call = int(engine.betting.to_call(player_id) or 0)
    pot = int(engine.betting.pot or 0)

    raw_targets = [
        min_raise_to,
        contribution + to_call + max(1, pot // 2),
        contribution + to_call + max(1, pot),
        max_raise_to,
    ]
    targets: list[int] = []
    seen: set[int] = set()
    for target in raw_targets:
        clamped = max(min_raise_to, min(max_raise_to, int(target)))
        if clamped in seen:
            continue
        seen.add(clamped)
        targets.append(clamped)
    return targets or [min_raise_to]


def _should_use_lookahead(engine, ai_player: str) -> bool:
    legal = engine.betting.legal_actions()
    if len(legal) <= 1:
        return False
    active_players = engine.betting.active_players()
    if len(active_players) > 2:
        return False
    big_blind = max(1, int(engine.betting.big_blind or 10))
    pot = max(0, int(engine.betting.pot or 0))
    to_call = max(0, int(engine.betting.to_call(ai_player) or 0))
    stack = max(0, int(engine.betting.stacks.get(ai_player, 0) or 0))
    if ActionType.RAISE not in legal and to_call == 0:
        return False
    return (
        pot >= 8 * big_blind
        or to_call >= 4 * big_blind
        or (to_call > 0 and to_call * 2 >= max(1, pot))
        or stack <= 20 * big_blind
    )


def _simulate_hand_to_terminal(
    engine,
    *,
    target_player: str,
    rng: random.Random,
    target_opponent_profile: Optional[Dict[str, Any]],
    max_actions: int = 120,
) -> float:
    def _find_next_eligible_player_local() -> Optional[str]:
        betting = engine.betting
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

        for player in engine.players:
            if player in betting.pending_players and player not in betting.folded_players and player not in betting.all_in_players:
                return player
        return None

    def _advance_without_actor_local() -> bool:
        next_player = _find_next_eligible_player_local()
        if next_player:
            if engine.betting.current_player != next_player:
                engine.betting.current_player = next_player
                return True
            return False

        street = engine.engine.street.value if hasattr(engine, "engine") else engine.street.value
        # engine is a poker Engine, not session; the hasattr check is defensive.
        if street == "preflop":
            engine.deal_flop()
            engine.betting.start_new_round(first_to_act=engine._first_to_act_postflop())  # type: ignore[attr-defined]
            return True
        if street == "flop":
            engine.deal_turn()
            engine.betting.start_new_round(first_to_act=engine._first_to_act_postflop())  # type: ignore[attr-defined]
            return True
        if street == "turn":
            engine.deal_river()
            engine.betting.start_new_round(first_to_act=engine._first_to_act_postflop())  # type: ignore[attr-defined]
            return True
        if street == "river":
            engine.resolve_showdown()
            return True
        return False

    actions_taken = 0
    while not engine.betting.hand_over and actions_taken < max_actions:
        if _advance_without_actor_local():
            continue
        actor = engine.betting.current_player
        if not actor:
            break

        sim_state = engine.to_ai_state()
        if target_opponent_profile and actor == target_player:
            sim_state["opponent_profile"] = target_opponent_profile
        sim_action = get_ai_action(sim_state, rng=rng)
        try:
            engine.step(sim_action, player_id=actor)
        except ValueError:
            # Fallback to a safe legal action inside rollout.
            legal = engine.betting.legal_actions()
            fallback = None
            for candidate in (ActionType.CHECK, ActionType.CALL, ActionType.FOLD, ActionType.RAISE):
                if candidate in legal:
                    if candidate == ActionType.RAISE:
                        fallback = Action(action=ActionType.RAISE, amount=engine.betting.min_raise_to())
                    else:
                        fallback = Action(action=candidate)
                    break
            if fallback is None:
                break
            try:
                engine.step(fallback, player_id=actor)
            except ValueError:
                break
        actions_taken += 1

    return float(engine.get_chip_change(target_player))


def _choose_ai_action_with_lookahead(
    session,
    *,
    ai_player: str,
    base_state: Dict[str, Any],
    opponent_profile: Optional[Dict[str, Any]],
) -> Action:
    base_action = get_ai_action(base_state)
    if not _should_use_lookahead(session.engine, ai_player):
        return base_action

    legal = session.engine.betting.legal_actions()
    candidates: list[Action] = []
    seen_keys: set[tuple[str, Optional[int]]] = set()

    def add_candidate(action: Action) -> None:
        key = (action.action.value, int(action.amount) if action.amount is not None else None)
        if key in seen_keys:
            return
        seen_keys.add(key)
        candidates.append(action)

    add_candidate(base_action)

    for action_type in legal:
        if action_type != ActionType.RAISE:
            add_candidate(Action(action=action_type))
            continue
        for amount in _raise_size_candidates(session.engine, ai_player):
            add_candidate(Action(action=ActionType.RAISE, amount=amount))

    if len(candidates) <= 1:
        return base_action

    rollout_count = 3
    best_action = base_action
    best_score = float("-inf")
    base_score = float("-inf")
    seed_base = int(time.time() * 1000) & 0x7FFFFFFF

    for idx, candidate in enumerate(candidates):
        scores: list[float] = []
        for rollout_idx in range(rollout_count):
            sim_engine = session.engine.clone()
            try:
                sim_engine.step(candidate, player_id=ai_player)
            except ValueError:
                continue
            rollout_rng = random.Random(seed_base + idx * 97 + rollout_idx * 997)
            score = _simulate_hand_to_terminal(
                sim_engine,
                target_player=ai_player,
                rng=rollout_rng,
                target_opponent_profile=opponent_profile,
            )
            scores.append(score)
        if not scores:
            continue
        avg_score = sum(scores) / len(scores)
        if candidate.action == base_action.action and candidate.amount == base_action.amount:
            base_score = avg_score
        if avg_score > best_score:
            best_score = avg_score
            best_action = candidate

    # Require a small edge to override the base policy to reduce rollout noise.
    bb = max(1, int(session.engine.betting.big_blind or 10))
    if best_action.action == base_action.action and best_action.amount == base_action.amount:
        return base_action
    if base_score != float("-inf") and best_score < base_score + (0.25 * bb):
        return base_action

    _trace(
        session.session_id,
        (
            "AI_LOOKAHEAD_OVERRIDE "
            f"player={ai_player} "
            f"base={base_action.action.value}:{base_action.amount} "
            f"chosen={best_action.action.value}:{best_action.amount} "
            f"base_score={round(base_score, 2) if base_score != float('-inf') else 'na'} "
            f"best_score={round(best_score, 2)}"
        ),
    )
    return best_action


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
    _start_hand_intro_block(session)
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
            intro_wait_remaining = _hand_intro_wait_remaining(session)
            if intro_wait_remaining > 0:
                await send_server_message(
                    websocket,
                    ServerMessage(
                        type="ERROR",
                        payload=ErrorMessage(
                            code="HAND_INTRO_IN_PROGRESS",
                            message="Hand intro sequence in progress",
                            details=[f"wait_ms={int(intro_wait_remaining * 1000)}"],
                        ),
                    ),
                )
                continue
            _trace(
                session.session_id,
                f"HUMAN_MOVE player={acting_player} action={action.action.value} amount={action.amount} street={session.engine.street.value}",
            )
            action_street = session.engine.street.value
            to_call_before = session.engine.betting.to_call(acting_player)
            legal_actions_before = list(session.engine.betting.legal_actions())
            try:
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
            _record_human_tendency(
                session,
                player_id=acting_player,
                action=action,
                street=action_street,
                to_call_before=to_call_before,
                legal_actions_before=legal_actions_before,
            )

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
                "small_blind": session.engine.betting.small_blind,
                "big_blind": session.engine.betting.big_blind,
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
    await _wait_for_hand_intro_if_needed(session)

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

        ai_player = session.engine.betting.current_player
        ai_state = session.engine.to_ai_state()
        opponent_profile = _build_opponent_profile(session, ai_player)
        if opponent_profile:
            ai_state["opponent_profile"] = opponent_profile
        try:
            ai_action = _choose_ai_action_with_lookahead(
                session,
                ai_player=ai_player,
                base_state=ai_state,
                opponent_profile=opponent_profile,
            )
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
        bets=dict(engine.betting.contributions),
        to_call=engine.betting.to_call(player_id),
        button_player=engine.button_player,
        active_players=list(engine.betting.players),
        schema_version="v2",
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
