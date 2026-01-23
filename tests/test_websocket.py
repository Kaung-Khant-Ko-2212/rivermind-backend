from fastapi.testclient import TestClient

from backend.main import app


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
