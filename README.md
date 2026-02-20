# Texas Hold'em Backend

## Run the server

```bash
uvicorn backend.main:app --reload
```

You can configure backend runtime values using `.env` at the backend project root
(`Member 1 + Member 2 + Member 4/.env`). See `.env.example`.

To use Member 3 strategy policy from shell:

```bash
set AI_MODE=strategy
uvicorn backend.main:app --reload
```

Optional strategy file override:

```bash
set AI_STRATEGY_PATH=backend/ai/strategy.json
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

## Multiplayer table flow

Use REST endpoints to create/join/start the table, then connect WebSocket in
`mode=multi`.

1. Create table (host gets seat `p1`):

```bash
curl -X POST http://127.0.0.1:8000/tables/create ^
  -H "Content-Type: application/json" ^
  -d "{\"user_key\":\"host-user\"}"
```

2. Join table (returns assigned seat):

```bash
curl -X POST http://127.0.0.1:8000/tables/TBL-XXXXXXX/join ^
  -H "Content-Type: application/json" ^
  -d "{\"user_key\":\"joiner-user\"}"
```

3. Start table (host only):

```bash
curl -X POST http://127.0.0.1:8000/tables/TBL-XXXXXXX/start ^
  -H "Content-Type: application/json" ^
  -d "{\"player_id\":\"p1\"}"
```

4. Connect seat via WebSocket:

```
ws://127.0.0.1:8000/ws?mode=multi&session_id=TBL-XXXXXXX&player_id=p2
```

Notes:
- Server supports seats `p1`..`p5`.
- Seats without a joined human are AI-controlled.
- Late join after table start is supported; newly joined seats can take over AI.

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

Member 3: CFR Scientist (integrated)
- `backend/ai/strategy.json` (exported strategy table used at runtime)
- `backend/ai/training_dataset.jsonl` (training dataset)
- `backend/ai/trainingdata.py` (MCCFR trainer/export script)
