from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Dict, Optional, Tuple, Any, Set
from uuid import uuid4

from .poker.engine import Engine


@dataclass
class SessionData:
    session_id: str
    engine: Engine
    last_seen: float
    player_sockets: Dict[str, Any] = field(default_factory=dict)
    human_players: Set[str] = field(default_factory=set)


class SessionStore:
    def __init__(self, ttl_seconds: int = 1800) -> None:
        self._sessions: Dict[str, SessionData] = {}
        self._ttl_seconds = ttl_seconds

    def get_or_create(
        self, session_id: Optional[str], now: Optional[float] = None
    ) -> Tuple[SessionData, bool]:
        current_time = now if now is not None else time.time()
        self._cleanup_expired(current_time)

        if session_id and session_id in self._sessions:
            session = self._sessions[session_id]
            session.last_seen = current_time
            return session, False

        if not session_id:
            session_id = uuid4().hex
        else:
            session_id = uuid4().hex

        session = SessionData(
            session_id=session_id, engine=Engine(), last_seen=current_time
        )
        self._sessions[session_id] = session
        return session, True

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
