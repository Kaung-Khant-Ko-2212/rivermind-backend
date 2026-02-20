# Texas Hold'em WebSocket Protocol

This document defines the canonical JSON contract used by the React client and
FastAPI backend.

## Connection

WebSocket endpoint:

```
ws://127.0.0.1:8000/ws?mode=<single|multi>&session_id=<optional>&player_id=<optional>
```

- `mode`: defaults to `single`. Use `multi` for multiplayer tables.
- `session_id`: reuse previous game state on reconnect.
- `player_id`: seat identity (`p1`..`p5`). Defaults to `p1`.

### Multiplayer table setup (required before `mode=multi`)

1. `POST /tables/create` -> returns `table_id` and host seat `p1`.
2. `POST /tables/{table_id}/join` -> returns assigned seat (`p2`..`p5`).
3. `POST /tables/{table_id}/start` -> host starts the table.
4. Connect websocket with `mode=multi`, `session_id=<table_id>`, `player_id=<seat>`.

If a socket connects with `mode=multi` but seat/table preconditions are not met,
server sends `ERROR` and closes the socket.

## Client -> Server

### MOVE

Canonical payload:

```json
{
  "type": "MOVE",
  "action": "call"
}
```

Raise example:

```json
{
  "type": "MOVE",
  "action": "raise",
  "amount": 120
}
```

Compatibility note:
- Backend currently accepts `val` as an alias for `action`:
  `{ "type": "MOVE", "val": "call" }`.
- New clients should always send `action`.

Validation rules:
- `type` must be `MOVE`.
- `action` must be one of: `check`, `call`, `fold`, `raise`.
- `amount` is required only for `raise`.
- `amount` is interpreted as **raise-to total** for this betting round.

## Server -> Client

### STATE

```json
{
  "type": "STATE",
  "payload": {
    "session_id": "1b1f7e6d8a2b4b6a9f7b6e3b6a9f7b6e",
    "street": "flop",
    "pot": 150,
    "community_cards": ["Ah", "Td", "7s"],
    "player_hand": ["Kc", "Kh"],
    "stacks": {
      "p1": 850,
      "p2": 1000
    },
    "bets": {
      "p1": 50,
      "p2": 50
    },
    "current_player": "p2",
    "legal_actions": ["check", "raise", "fold"],
    "history": [
      {
        "player_id": "p1",
        "action": {
          "action": "raise",
          "amount": 50
        }
      }
    ]
  }
}
```

Field notes:
- `player_hand` is only present for the viewer seat.
- `history` is the recent action sequence (bounded on backend).

### EVENT

```json
{
  "type": "EVENT",
  "payload": {
    "event": "DEAL_FLOP",
    "data": {
      "street": "flop",
      "cards": ["Ah", "Td", "7s"]
    }
  }
}
```

Known event names:
- `DEAL_HOLE`
- `DEAL_FLOP`
- `DEAL_TURN`
- `DEAL_RIVER`
- `SHOWDOWN`
- `HAND_END`
- `NEW_HAND`
- `TABLE_END` (multiplayer table complete; no further moves accepted)

### ERROR

```json
{
  "type": "ERROR",
  "payload": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid message",
    "details": ["amount: amount is required for raise"]
  }
}
```

`code` is optional but recommended for client-side handling.
