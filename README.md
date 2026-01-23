# Texas Hold'em Backend

## Run the server

```bash
uvicorn backend.main:app --reload
```

## Manual WebSocket test

Using a WebSocket client such as `websocat`:

```bash
websocat ws://127.0.0.1:8000/ws
```

On connect, you should receive a `STATE` message. Send a MOVE:

```json
{"type":"MOVE","val":"call"}
```

The server should respond with another `STATE` message. The canonical field name
is `action`, but the server also accepts `val` for compatibility.

## Simple UI (browser)

Open a basic test UI from `ui/index.html`:

```bash
python -m http.server 8080
```

Then visit `http://127.0.0.1:8080/ui/` in your browser and click Connect.
The UI will store your `session_id` in localStorage and reuse it on reconnect.

## Session persistence

The initial `STATE` message includes a `session_id`. Persist this on the client
(for example, using `localStorage`) and reconnect with
`ws://127.0.0.1:8000/ws?session_id=...` to resume the same game state.

## Multi-human seats (minimal)

The server supports up to 5 seats. To connect as a specific player, pass
`player_id`:

```
ws://127.0.0.1:8000/ws?session_id=...&player_id=p2
```

If a seat has no connected human, the AI will act for that seat.

## Team responsibilities (file map)

Member 1: Game Engine Architect
- `backend/poker/cards.py` (deck, shuffle, dealing helpers)
- `backend/poker/evaluator.py` (Treys hand evaluation)
- `backend/poker/betting.py` (betting state machine)
- `backend/poker/engine.py` (street flow + state serialization)
- `backend/schemas.py` (public state schema alignment)

Member 4: Infrastructure & Bridge Lead
- `backend/main.py` (FastAPI WebSocket server)
- `backend/session_store.py` (session management + TTL)
- `backend/training/replay_buffer.py` (experience replay buffer)
- `backend/config.py` (feature flags + buffer config)
- `backend/ai/policy.py` (AI action hook for WS loop)
- `backend/protocol.md` (message/event contract)
