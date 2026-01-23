# Codebase Overview (TOUNDERSTAND.md)

This file explains what each implemented file in this directory does, at a high level.

## Top-Level
- `README.md`  
  How to run the server, test via WebSocket, use the simple UI, and persist sessions.
- `FORPROJECT.md`  
  Role-based status and handoff notes mapped to the project plan.
- `requirements.txt`  
  Python dependencies for server + tests (FastAPI, uvicorn, pydantic, treys, pytest, httpx).

## Backend Core
- `backend/__init__.py`  
  Marks `backend` as a package.
- `backend/main.py`  
  FastAPI app with `/ws` WebSocket endpoint.  
  Handles:
  - session creation/resume
  - initial STATE
  - MOVE validation
  - calling the engine
  - AI hook for one move after human
  - EVENT + STATE emission
  - optional replay buffer recording
- `backend/logging_setup.py`  
  Basic logging configuration used by the server.
- `backend/config.py`  
  Config flags sourced from env (replay buffer, AI mode/seed).
- `backend/session_store.py`  
  In-memory session storage with TTL cleanup and `last_seen`.
- `backend/schemas.py`  
  Pydantic models for:
  - client MOVE messages
  - server STATE / EVENT / ERROR messages
  - game state schema
  Includes aliases like `player_hand` and `history`, plus validation helpers.
- `backend/protocol.md`  
  Message and event examples for WebSocket communication.

## Poker Engine
- `backend/poker/__init__.py`  
  Marks poker engine package.
- `backend/poker/cards.py`  
  Card helpers and `Deck` (build, shuffle, deal).
- `backend/poker/evaluator.py`  
  Treys wrapper for hand evaluation + hand category labels.
- `backend/poker/betting.py`  
  Betting state machine (2-5 players):
  - blinds
  - turn order
  - legal actions
  - min-raise rules
  - round completion + payout
- `backend/poker/engine.py`  
  Orchestrates the game:
  - deals cards by street
  - advances betting rounds
  - resolves showdown/folds
  - queues DEAL_* / HAND_END events
  - serializes public state for UI
- `backend/poker/cli.py`  
  Small CLI to deal a hand and print a showdown result.

## AI Hook
- `backend/ai/__init__.py`  
  Marks AI package.
- `backend/ai/policy.py`  
  Stub AI policy:
  - chooses a random legal action (or passive mode)
  - produces valid raise amounts
  - supports AI_MODE / AI_SEED env control

## Training / Replay Buffer
- `backend/training/__init__.py`  
  Marks training package.
- `backend/training/replay_buffer.py`  
  ReplayBuffer:
  - add / sample
  - max capacity with eviction
  - save/load JSONL

## Simple Test UI
- `ui/index.html`  
  Lightweight browser UI to connect to WebSocket and send actions.  
  Shows state, cards, stacks, and a live log.

## Tests
- `tests/conftest.py`  
  Adds project root to `sys.path` and sets AI_MODE to passive for deterministic tests.
- `tests/test_schemas.py`  
  Validates schema parsing and error formatting.
- `tests/test_poker_engine.py`  
  Deck correctness, deterministic dealing, evaluator sanity, public state validation.
- `tests/test_betting.py`  
  Betting round logic, min-raise rules, fold/showdown payouts.
- `tests/test_ai_policy.py`  
  Ensures AI returns legal actions and valid raise sizes.
- `tests/test_replay_buffer.py`  
  Replay buffer add/sample/evict/save/load behavior.
- `tests/test_websocket.py`  
  WebSocket integration: connect, STATE, events, reconnect, and AI loop safety.

## Notes
- Engine supports 2-5 players with fixed blinds (SB=5, BB=10).
- Seats are `p1`..`p5`. Use `player_id` in the WebSocket query for multi-human tests.
- Raise `amount` is treated as "raise to" on the current street.
- AI is a stub (random or passive), intended to be replaced by Member 3's strategy.
