from fastapi.testclient import TestClient

from backend.main import app, store


def _recv_until_state(websocket):
    while True:
        message = websocket.receive_json()
        if message["type"] == "STATE":
            return message


def _drive_to_flop(websocket, max_steps: int = 40):
    state = _recv_until_state(websocket)
    while state["payload"]["current_player"] != "p1":
        state = _recv_until_state(websocket)
    steps = 0
    while state["payload"]["street"] != "flop" and steps < max_steps:
        if state["payload"]["current_player"] == "p1":
            websocket.send_json({"type": "MOVE", "val": "call"})
        state = _recv_until_state(websocket)
        steps += 1
    return state


def test_websocket_initial_state_and_move_val() -> None:
    client = TestClient(app)
    with client.websocket_connect("/ws") as websocket:
        initial = _recv_until_state(websocket)
        assert initial["type"] == "STATE"
        assert "session_id" in initial["payload"]
        response = _drive_to_flop(websocket)
        assert response["payload"]["street"] == "flop"
        assert len(response["payload"]["community_cards"]) == 3


def test_websocket_malformed_json_returns_error_and_continues() -> None:
    client = TestClient(app)
    with client.websocket_connect("/ws") as websocket:
        _recv_until_state(websocket)

        websocket.send_text("{bad json")
        error = websocket.receive_json()
        assert error["type"] == "ERROR"

        response = _drive_to_flop(websocket)
        assert response["type"] == "STATE"


def test_websocket_reconnect_preserves_state() -> None:
    client = TestClient(app)
    with client.websocket_connect("/ws") as websocket:
        initial = _recv_until_state(websocket)
        session_id = initial["payload"]["session_id"]

        flop_state = _drive_to_flop(websocket)
        board = flop_state["payload"]["community_cards"]
        street = flop_state["payload"]["street"]

    with client.websocket_connect(f"/ws?session_id={session_id}") as websocket:
        resumed = _recv_until_state(websocket)
        assert resumed["payload"]["session_id"] == session_id
        assert resumed["payload"]["community_cards"] == board
        assert resumed["payload"]["street"] == street


def test_websocket_deal_flop_event() -> None:
    client = TestClient(app)
    with client.websocket_connect("/ws") as websocket:
        saw_flop_event = False
        attempts = 0

        while not saw_flop_event and attempts < 10:
            message = websocket.receive_json()
            if message["type"] == "EVENT":
                if message["payload"]["event"] == "DEAL_FLOP":
                    saw_flop_event = True
                    break
            elif message["type"] == "STATE":
                if message["payload"]["current_player"] == "p1":
                    websocket.send_json({"type": "MOVE", "val": "call"})
                    attempts += 1

        assert saw_flop_event is True


def test_multiplayer_join_after_start_can_connect() -> None:
    store._sessions.clear()  # type: ignore[attr-defined]
    client = TestClient(app)

    create_response = client.post("/tables/create", json={"user_key": "host-user"})
    assert create_response.status_code == 200
    create_payload = create_response.json()
    table_id = create_payload["table_id"]
    host_player_id = create_payload["player_id"]

    start_response = client.post(
        f"/tables/{table_id}/start",
        json={"player_id": host_player_id},
    )
    assert start_response.status_code == 200
    assert start_response.json()["started"] is True

    join_response = client.post(
        f"/tables/{table_id}/join",
        json={"user_key": "late-user"},
    )
    assert join_response.status_code == 200
    join_payload = join_response.json()
    assert join_payload["player_id"] == "p2"
    assert join_payload["status"]["started"] is True

    with client.websocket_connect(
        f"/ws?mode=multi&session_id={table_id}&player_id={join_payload['player_id']}"
    ) as websocket:
        state = _recv_until_state(websocket)
        assert state["type"] == "STATE"
        assert state["payload"]["session_id"] == table_id


def test_single_mode_rejects_table_style_session_id() -> None:
    client = TestClient(app)
    with client.websocket_connect("/ws?mode=single&session_id=TBL-INVALID&player_id=p1") as websocket:
        error = websocket.receive_json()
        assert error["type"] == "ERROR"
        assert error["payload"]["code"] == "INVALID_SINGLE_SESSION_ID"


def test_multiplayer_table_end_emits_event_and_rejects_moves() -> None:
    store._sessions.clear()  # type: ignore[attr-defined]
    client = TestClient(app)

    create_response = client.post("/tables/create", json={"user_key": "host-user"})
    assert create_response.status_code == 200
    table_id = create_response.json()["table_id"]

    start_response = client.post(
        f"/tables/{table_id}/start",
        json={"player_id": "p1"},
    )
    assert start_response.status_code == 200

    session = store.get(table_id)
    assert session is not None
    session.engine.betting.stacks = {
        "p1": 1000,
        "p2": 0,
        "p3": 0,
        "p4": 0,
        "p5": 0,
    }
    session.engine.betting.hand_over = True
    session.engine.betting.current_player = None

    with client.websocket_connect(f"/ws?mode=multi&session_id={table_id}&player_id=p1") as websocket:
        messages = [websocket.receive_json() for _ in range(4)]
        events = [msg["payload"]["event"] for msg in messages if msg["type"] == "EVENT"]

        assert "TABLE_END" in events
        assert "NEW_HAND" not in events

        websocket.send_json({"type": "MOVE", "val": "call"})
        error = websocket.receive_json()
        assert error["type"] == "ERROR"
        assert error["payload"]["code"] == "TABLE_ENDED"

    session_after = store.get(table_id)
    assert session_after is not None
    assert session_after.table_ended is True
    assert session_after.table_winners == ["p1"]
