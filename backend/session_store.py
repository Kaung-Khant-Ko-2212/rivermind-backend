from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Dict, Optional, Tuple, Any, Set, List
from uuid import uuid4

from .poker.engine import Engine

SEAT_ORDER: List[str] = ["p1", "p2", "p3", "p4", "p5"]


@dataclass
class SessionData:
    session_id: str
    engine: Engine
    last_seen: float
    mode: str = "single"
    host_player_id: str = "p1"
    started: bool = False
    player_sockets: Dict[str, Any] = field(default_factory=dict)
    human_players: Set[str] = field(default_factory=set)
    joined_players: Set[str] = field(default_factory=set)
    seat_owners: Dict[str, str] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    table_ended: bool = False
    table_winners: List[str] = field(default_factory=list)
    awaiting_hand_continue: bool = False


class SessionStore:
    def __init__(self, ttl_seconds: int = 1800) -> None:
        self._sessions: Dict[str, SessionData] = {}
        self._ttl_seconds = ttl_seconds

    def _generate_table_id(self) -> str:
        while True:
            candidate = f"TBL-{uuid4().hex[:8].upper()}"
            if candidate not in self._sessions:
                return candidate

    def _create_session(
        self,
        session_id: str,
        current_time: float,
        mode: str,
        host_player_id: str = "p1",
    ) -> SessionData:
        session = SessionData(
            session_id=session_id,
            engine=Engine(),
            last_seen=current_time,
            mode=mode,
            host_player_id=host_player_id,
            started=False,
            created_at=current_time,
        )
        if mode == "multi":
            session.joined_players.add(host_player_id)
        self._sessions[session_id] = session
        return session

    def create_multiplayer_table(
        self,
        host_user_key: Optional[str] = None,
        now: Optional[float] = None,
    ) -> SessionData:
        current_time = now if now is not None else time.time()
        self._cleanup_expired(current_time)
        session_id = self._generate_table_id()
        session = self._create_session(
            session_id=session_id,
            current_time=current_time,
            mode="multi",
            host_player_id="p1",
        )
        if host_user_key:
            session.seat_owners["p1"] = host_user_key
        return session

    def get(self, session_id: str, now: Optional[float] = None) -> Optional[SessionData]:
        current_time = now if now is not None else time.time()
        self._cleanup_expired(current_time)
        session = self._sessions.get(session_id)
        if session:
            session.last_seen = current_time
        return session

    def get_or_create(
        self,
        session_id: Optional[str],
        now: Optional[float] = None,
        mode: str = "single",
    ) -> Tuple[SessionData, bool]:
        current_time = now if now is not None else time.time()
        self._cleanup_expired(current_time)

        if session_id and session_id in self._sessions:
            session = self._sessions[session_id]
            session.last_seen = current_time
            return session, False

        if not session_id:
            session_id = uuid4().hex
        session = self._create_session(
            session_id=session_id,
            current_time=current_time,
            mode=mode,
            host_player_id="p1",
        )
        return session, True

    def join_multiplayer_table(
        self,
        session_id: str,
        user_key: Optional[str] = None,
        now: Optional[float] = None,
    ) -> str:
        current_time = now if now is not None else time.time()
        self._cleanup_expired(current_time)
        session = self._sessions.get(session_id)
        if not session:
            raise ValueError("Table not found")
        if session.mode != "multi":
            raise ValueError("Not a multiplayer table")
        if session.table_ended:
            raise ValueError("Table has ended")
        session.last_seen = current_time

        if user_key:
            for seat, owner in session.seat_owners.items():
                if owner == user_key:
                    return seat

        for seat in SEAT_ORDER:
            if seat not in session.joined_players:
                session.joined_players.add(seat)
                if user_key:
                    session.seat_owners[seat] = user_key
                return seat
        raise ValueError("Table is full")

    def start_multiplayer_table(
        self,
        session_id: str,
        requester_player_id: str,
        now: Optional[float] = None,
    ) -> SessionData:
        current_time = now if now is not None else time.time()
        self._cleanup_expired(current_time)
        session = self._sessions.get(session_id)
        if not session:
            raise ValueError("Table not found")
        if session.mode != "multi":
            raise ValueError("Not a multiplayer table")
        if session.table_ended:
            raise ValueError("Table has ended")
        if requester_player_id != session.host_player_id:
            raise PermissionError("Only host can start the table")
        session.last_seen = current_time
        if not session.started:
            session.engine.new_hand()
            session.started = True
        return session

    def register_socket(self, session_id: str, player_id: str, websocket: Any) -> None:
        session = self._sessions.get(session_id)
        if not session:
            return
        session.player_sockets[player_id] = websocket
        session.human_players.add(player_id)

    def remove_socket(self, session_id: str, player_id: str) -> None:
        session = self._sessions.get(session_id)
        if not session:
            return
        if session.player_sockets.get(player_id) is not None:
            session.player_sockets.pop(player_id, None)
        session.human_players.discard(player_id)

    def touch(self, session_id: str, now: Optional[float] = None) -> None:
        current_time = now if now is not None else time.time()
        self._cleanup_expired(current_time)
        session = self._sessions.get(session_id)
        if session:
            session.last_seen = current_time

    def _cleanup_expired(self, now: float) -> None:
        expired = [
            session_id
            for session_id, session in self._sessions.items()
            if now - session.last_seen > self._ttl_seconds
        ]
        for session_id in expired:
            del self._sessions[session_id]
