# Project Status & Handoff Notes

This file summarizes what the current codebase already contributes (by role) and what other members should do next, based on `Roles.pdf`.

## Current Contributions (This Directory)

### Member 1: Game Engine Architect
Implemented in this repo:
- Deck + shuffle + dealing helpers: `backend/poker/cards.py`
- Hand evaluation (Treys): `backend/poker/evaluator.py`
- Betting state machine (2-5 players, blinds, turn order, min-raise): `backend/poker/betting.py`
- Engine orchestration, street flow, showdown, serialization: `backend/poker/engine.py`
- JSON schema alignment for frontend: `backend/schemas.py`

Status: **Complete for Phases 1-3 (with documented simplifications).**

### Member 4: Infrastructure & Bridge Lead
Implemented in this repo:
- FastAPI WebSocket server: `backend/main.py`
- Session management with TTL: `backend/session_store.py`
- Replay buffer infra: `backend/training/replay_buffer.py`
- Config flags: `backend/config.py`
- AI action hook (stub policy): `backend/ai/policy.py`
- Protocol docs: `backend/protocol.md`
- Simple test UI: `ui/index.html`

Status: **Complete for Phases 1-3 (with AI stub).**

## What Other Members Should Do Next

### Member 2: Cartographer (Abstraction Lead)
Needed work:
- Implement EHS calculator (Monte Carlo): new module recommended under `backend/analysis/` or `backend/training/`.
- Implement bucketing (information abstraction) and a mapping table from live cards to bucket ids.
- Expose a stable `infoset_id` function to replace the placeholder in `backend/main.py` replay logging.

Suggested integration points:
- `backend/training/replay_buffer.py` can store bucketed infosets.
- Add a utility module for `compute_infoset_id(...)` and call it from `_record_experience()` in `backend/main.py`.

### Member 3: CFR Scientist (AI Lead)
Needed work:
- Implement MCCFR (external sampling CFR).
- Training loop to update regrets and strategies.
- Export a strategy profile (lookup table or model).

Suggested integration points:
- Replace `backend/ai/policy.py` with a real strategy lookup.
- Use replay buffer if needed for supervised/analysis steps.
- Add a loader for exported strategy artifacts.

### Member 5: UX & Integration Developer
Needed work:
- React UI (Table / Card / Controls / Pot) with Tailwind CSS.
- WebSocket listeners for `STATE` and `EVENT` (DEAL_* / HAND_END / NEW_HAND).
- Action handlers to send `{ "type": "MOVE", "action": "...", "amount": n }`.

Suggested integration points:
- Use the protocol examples in `backend/protocol.md`.
- The UI can start with the simple test UI pattern in `ui/index.html`.

## Notes & Assumptions
- Engine supports 2-5 players with fixed blinds (SB=5, BB=10) and starting stacks 1000.
- Seats are `p1`..`p5`. Use `player_id` in the WebSocket query to connect multiple humans.
- Raise `amount` is treated as "raise to" on the current street with min-raise rules.
- AI policy is a stub (random or passive mode).
- Session persistence uses `session_id` in `STATE`, store in localStorage for reconnect.

## Quick Run
1) Server: `uvicorn backend.main:app --reload`
2) UI: `python -m http.server 8080` -> open `http://127.0.0.1:8080/ui/`
