# Member 2B: Bucketing System Documentation

This document provides comprehensive documentation for the information abstraction (bucketing) system implemented for Member 2B tasks.

## Table of Contents

1. [Bucket Definition Document](#bucket-definition-document)
2. [Mapping Function and Lookup Table](#mapping-function-and-lookup-table)
3. [Stage-Specific Abstraction Summary](#stage-specific-abstraction-summary)
4. [Infoset ID Format](#infoset-id-format)
5. [Usage Examples](#usage-examples)

---

## 1. Bucket Definition Document

### 1.1 Hole Card Buckets

Hole cards are bucketed based on:
- **Pair status**: Pocket pairs vs. non-pairs
- **Rank values**: High cards (A, K, Q, J, T, 9, 8) vs. mid/low cards
- **Suit matching**: Suited vs. unsuited

#### Bucket Categories

| Bucket Pattern | Description | Examples |
|---------------|-------------|----------|
| `PP_AA` | Pocket Aces | As Ah, Ad Ac |
| `PP_KK` | Pocket Kings | Ks Kh, Kd Kc |
| `PP_QQ` | Pocket Queens | Qs Qh, Qd Qc |
| `PP_JJ` | Pocket Jacks | Js Jh, Jd Jc |
| `PP_TT` | Pocket Tens | Ts Th, Td Tc |
| `PP_99` | Pocket Nines | 9s 9h, 9d 9c |
| `PP_88` | Pocket Eights | 8s 8h, 8d 8c |
| `PP_77` | Pocket Sevens | 7s 7h, 7d 7c |
| `PP_66` | Pocket Sixes | 6s 6h, 6d 6c |
| `PP_55` | Pocket Fives | 5s 5h, 5d 5c |
| `PP_44` | Pocket Fours | 4s 4h, 4d 4c |
| `PP_33` | Pocket Threes | 3s 3h, 3d 3c |
| `PP_22` | Pocket Twos | 2s 2h, 2d 2c |
| `SUITED_AK` | Ace-King suited | As Ks, Ah Kh |
| `UNSUITED_AK` | Ace-King offsuit | As Kd, Ah Kc |
| `SUITED_AQ` | Ace-Queen suited | As Qs, Ah Qh |
| `UNSUITED_AQ` | Ace-Queen offsuit | As Qd, Ah Qc |
| `SUITED_AJ` | Ace-Jack suited | As Js, Ah Jh |
| `UNSUITED_AJ` | Ace-Jack offsuit | As Jd, Ah Jc |
| `SUITED_AT` | Ace-Ten suited | As Ts, Ah Th |
| `UNSUITED_AT` | Ace-Ten offsuit | As Td, Ah Tc |
| `SUITED_A9` | Ace-Nine suited | As 9s, Ah 9h |
| `UNSUITED_A9` | Ace-Nine offsuit | As 9d, Ah 9c |
| `SUITED_A8` | Ace-Eight suited | As 8s, Ah 8h |
| `UNSUITED_A8` | Ace-Eight offsuit | As 8d, Ah 8c |
| `SUITED_KQ` | King-Queen suited | Ks Qs, Kh Qh |
| `UNSUITED_KQ` | King-Queen offsuit | Kd Qc, Kc Qd |
| `SUITED_KJ` | King-Jack suited | Ks Js, Kh Jh |
| `UNSUITED_KJ` | King-Jack offsuit | Kd Jc, Kc Jd |
| `SUITED_KT` | King-Ten suited | Ks Ts, Kh Th |
| `UNSUITED_KT` | King-Ten offsuit | Kd Tc, Kc Td |
| `SUITED_K9` | King-Nine suited | Ks 9s, Kh 9h |
| `UNSUITED_K9` | King-Nine offsuit | Kd 9c, Kc 9d |
| `SUITED_K8` | King-Eight suited | Ks 8s, Kh 8h |
| `UNSUITED_K8` | King-Eight offsuit | Kd 8c, Kc 8d |
| `SUITED_QJ` | Queen-Jack suited | Qs Js, Qh Jh |
| `UNSUITED_QJ` | Queen-Jack offsuit | Qd Jc, Qc Jd |
| `SUITED_QT` | Queen-Ten suited | Qs Ts, Qh Th |
| `UNSUITED_QT` | Queen-Ten offsuit | Qd Tc, Qc Td |
| `SUITED_Q9` | Queen-Nine suited | Qs 9s, Qh 9h |
| `UNSUITED_Q9` | Queen-Nine offsuit | Qd 9c, Qc 9d |
| `SUITED_Q8` | Queen-Eight suited | Qs 8s, Qh 8h |
| `UNSUITED_Q8` | Queen-Eight offsuit | Qd 8c, Qc 8d |
| `SUITED_JT` | Jack-Ten suited | Js Ts, Jh Th |
| `UNSUITED_JT` | Jack-Ten offsuit | Jd Tc, Jc Td |
| `SUITED_J9` | Jack-Nine suited | Js 9s, Jh 9h |
| `UNSUITED_J9` | Jack-Nine offsuit | Jd 9c, Jc 9d |
| `SUITED_J8` | Jack-Eight suited | Js 8s, Jh 8h |
| `UNSUITED_J8` | Jack-Eight offsuit | Jd 8c, Jc 8d |
| `SUITED_T9` | Ten-Nine suited | Ts 9s, Th 9h |
| `UNSUITED_T9` | Ten-Nine offsuit | Td 9c, Tc 9d |
| `SUITED_T8` | Ten-Eight suited | Ts 8s, Th 8h |
| `UNSUITED_T8` | Ten-Eight offsuit | Td 8c, Tc 8d |
| `SUITED_98` | Nine-Eight suited | 9s 8s, 9h 8h |
| `UNSUITED_98` | Nine-Eight offsuit | 9d 8c, 9c 8d |
| `SUITED_MID` | Suited mid cards (7-6) | 7s 6s, 7h 6h |
| `UNSUITED_MID` | Unsuited mid cards (7-6) | 7d 6c, 7c 6d |
| `SUITED_LOW` | Suited low cards (≤5) | 5s 4s, 3h 2h |
| `UNSUITED_LOW` | Unsuited low cards (≤5) | 5d 4c, 3c 2d |

**Notes:**
- Cards are normalized so the higher rank comes first
- Pairs are always bucketed by rank (AA, KK, QQ, etc.)
- High card combinations (rank ≥ 8) are bucketed by specific ranks
- Mid cards (rank 6-7) are bucketed as `MID`
- Low cards (rank ≤ 5) are bucketed as `LOW`

### 1.2 Board Buckets

Board cards are bucketed based on:
- **Street**: Preflop, Flop, Turn, River
- **Suit distribution**: Rainbow, Two-tone, Monotone
- **Rank distribution**: Paired vs. unpaired
- **Card ranks**: High cards vs. low cards

#### Flop Buckets (3 cards)

| Bucket Pattern | Description | Examples |
|---------------|-------------|----------|
| `FLOP_RAINBOW` | Three different suits | As Kd Qc |
| `FLOP_RAINBOW_PAIRED` | Rainbow with pair | As Ad Kc |
| `FLOP_RAINBOW_HIGH` | Rainbow with 2+ high cards | As Kd Qc |
| `FLOP_RAINBOW_LOW` | Rainbow with no high cards | 7c 6d 5h |
| `FLOP_TWO_TONE` | Two cards same suit | As Ks Qd |
| `FLOP_TWO_TONE_PAIRED` | Two-tone with pair | As Ad Ks |
| `FLOP_TWO_TONE_HIGH` | Two-tone with 2+ high cards | As Ks Qd |
| `FLOP_TWO_TONE_LOW` | Two-tone with no high cards | 7s 6s 5d |
| `FLOP_MONOTONE` | All three same suit | As Ks Qs |
| `FLOP_MONOTONE_PAIRED` | Monotone with pair | As Ad Ks |
| `FLOP_MONOTONE_HIGH` | Monotone with 2+ high cards | As Ks Qs |
| `FLOP_MONOTONE_LOW` | Monotone with no high cards | 7s 6s 5s |

#### Turn Buckets (4 cards)

| Bucket Pattern | Description | Examples |
|---------------|-------------|----------|
| `TURN_RAINBOW` | Four different suits | As Kd Qc Jh |
| `TURN_RAINBOW_PAIRED` | Rainbow with pair | As Ad Kc Qh |
| `TURN_TWO_TONE` | Two cards same suit | As Ks Qd Jh |
| `TURN_TWO_TONE_PAIRED` | Two-tone with pair | As Ad Ks Qh |
| `TURN_FLUSH_DRAW` | Three cards same suit | As Ks Qs Jd |
| `TURN_FLUSH_DRAW_PAIRED` | Flush draw with pair | As Ad Ks Qs |

#### River Buckets (5 cards)

| Bucket Pattern | Description | Examples |
|---------------|-------------|----------|
| `RIVER_RAINBOW` | Five different suits | As Kd Qc Jh Td |
| `RIVER_RAINBOW_PAIRED` | Rainbow with pair | As Ad Kc Qh Td |
| `RIVER_FLUSH_DRAW` | Four cards same suit | As Ks Qs Js Td |
| `RIVER_FLUSH_DRAW_PAIRED` | Flush draw with pair | As Ad Ks Qs Js |
| `RIVER_FLUSH` | All five same suit | As Ks Qs Js Ts |
| `RIVER_FLUSH_PAIRED` | Flush with pair | As Ad Ks Qs Js |

**Notes:**
- High cards = A, K, Q, J, T (rank ≥ 10)
- Paired = at least one pair on board
- Flush draw = 3-4 cards of same suit
- Flush = 5 cards of same suit

### 1.3 Pot Size Buckets

Pot size is bucketed relative to the big blind (BB).

| Bucket | Pot Size (in BB) | Example (BB=10) |
|--------|-----------------|-----------------|
| `POT_TINY` | < 5 BB | 0-49 chips |
| `POT_SMALL` | 5-20 BB | 50-199 chips |
| `POT_MEDIUM` | 20-50 BB | 200-499 chips |
| `POT_LARGE` | 50-100 BB | 500-999 chips |
| `POT_HUGE` | ≥ 100 BB | ≥ 1000 chips |

### 1.4 Stack Ratio Buckets

Stack size is bucketed relative to the big blind (BB).

| Bucket | Stack Size (in BB) | Example (BB=10) |
|--------|-------------------|-----------------|
| `STACK_SHORT` | ≤ 20 BB | ≤ 200 chips |
| `STACK_SHALLOW` | 20-50 BB | 201-499 chips |
| `STACK_MEDIUM` | 50-100 BB | 500-999 chips |
| `STACK_DEEP` | > 100 BB | ≥ 1000 chips |

### 1.5 Betting Sequence Buckets

Betting sequences are bucketed by:
- **Street**: Preflop, Flop, Turn, River
- **Action pattern**: Last 2-3 actions in sequence

| Bucket Pattern | Description | Examples |
|---------------|-------------|----------|
| `{STREET}_NO_ACTION` | No actions yet this street | `PREFLOP_NO_ACTION` |
| `{STREET}_check` | Single check | `FLOP_check` |
| `{STREET}_call` | Single call | `PREFLOP_call` |
| `{STREET}_raise` | Single raise | `FLOP_raise` |
| `{STREET}_fold` | Single fold | `TURN_fold` |
| `{STREET}_check_check` | Two checks | `FLOP_check_check` |
| `{STREET}_call_call` | Two calls | `PREFLOP_call_call` |
| `{STREET}_raise_call` | Raise then call | `FLOP_raise_call` |
| `{STREET}_call_raise` | Call then raise | `PREFLOP_call_raise` |
| `{STREET}_check_raise` | Check then raise | `FLOP_check_raise` |
| `{STREET}_raise_raise` | Two raises | `PREFLOP_raise_raise` |

**Notes:**
- Uses last 2-3 actions from action history
- Actions are: `check`, `call`, `raise`, `fold`
- Street prefix is uppercase (PREFLOP, FLOP, TURN, RIVER)

---

## 2. Mapping Function and Lookup Table

### 2.1 Mapping Functions

The bucketing system uses several mapping functions to convert raw game state into bucket IDs:

#### `bucket_hole_cards(hole_cards: List[str]) -> str`

**Input**: List of 2 card strings (e.g., `["As", "Ah"]`)  
**Output**: Hole card bucket ID (e.g., `"PP_AA"`)

**Mapping Logic**:
1. Extract ranks and suits from cards
2. Normalize: higher rank first
3. Check if pair → bucket as `PP_{RANK}{RANK}`
4. Check if suited → prefix with `SUITED_` or `UNSUITED_`
5. If rank ≥ 8 → bucket by specific ranks (e.g., `AK`, `KQ`)
6. If rank 6-7 → bucket as `MID`
7. If rank ≤ 5 → bucket as `LOW`

**Lookup Table** (simplified):

```python
# Pairs
if rank1 == rank2:
    return f"PP_{rank_name}{rank_name}"

# High cards (rank >= 8)
if rank1 >= 8 and rank2 >= 8:
    if suited:
        return f"SUITED_{rank1_name}{rank2_name}"
    else:
        return f"UNSUITED_{rank1_name}{rank2_name}"

# Mid cards (rank 6-7)
if rank1 >= 6:
    if suited:
        return "SUITED_MID"
    else:
        return "UNSUITED_MID"

# Low cards (rank <= 5)
if suited:
    return "SUITED_LOW"
else:
    return "UNSUITED_LOW"
```

#### `bucket_board(board: List[str]) -> str`

**Input**: List of board cards (0-5 cards)  
**Output**: Board bucket ID (e.g., `"FLOP_RAINBOW_HIGH"`)

**Mapping Logic**:
1. Count cards → determine street (0=preflop, 3=flop, 4=turn, 5=river)
2. Count suits → determine texture (rainbow/two-tone/monotone/flush)
3. Count ranks → check for pairs
4. Count high cards (rank ≥ 10) → high/low classification
5. Combine into bucket ID

**Lookup Table** (simplified):

```python
# Preflop
if len(board) == 0:
    return "PREFLOP"

# Flop (3 cards)
if len(board) == 3:
    suit_count = max(suit_counts)
    rank_count = max(rank_counts)
    high_count = sum(1 for r in ranks if r >= 10)
    
    texture = "RAINBOW" if suit_count == 1 else \
              "TWO_TONE" if suit_count == 2 else "MONOTONE"
    
    if rank_count >= 2:
        texture += "_PAIRED"
    if high_count >= 2:
        texture += "_HIGH"
    elif high_count == 0:
        texture += "_LOW"
    
    return f"FLOP_{texture}"

# Turn (4 cards)
if len(board) == 4:
    suit_count = max(suit_counts)
    rank_count = max(rank_counts)
    
    texture = "RAINBOW" if suit_count == 1 else \
              "TWO_TONE" if suit_count == 2 else "FLUSH_DRAW"
    
    if rank_count >= 2:
        texture += "_PAIRED"
    
    return f"TURN_{texture}"

# River (5 cards)
if len(board) == 5:
    suit_count = max(suit_counts)
    rank_count = max(rank_counts)
    
    texture = "RAINBOW" if suit_count == 1 else \
              "FLUSH_DRAW" if suit_count == 4 else "FLUSH"
    
    if rank_count >= 2:
        texture += "_PAIRED"
    
    return f"RIVER_{texture}"
```

#### `bucket_pot_size(pot: int, big_blind: int) -> str`

**Input**: Pot size in chips, big blind size  
**Output**: Pot bucket ID (e.g., `"POT_SMALL"`)

**Mapping Logic**:
```python
pot_in_bb = pot / big_blind

if pot_in_bb < 5:
    return "POT_TINY"
elif pot_in_bb < 20:
    return "POT_SMALL"
elif pot_in_bb < 50:
    return "POT_MEDIUM"
elif pot_in_bb < 100:
    return "POT_LARGE"
else:
    return "POT_HUGE"
```

#### `bucket_stack_ratio(player_stack: int, pot: int, big_blind: int) -> str`

**Input**: Player stack, pot size, big blind size  
**Output**: Stack bucket ID (e.g., `"STACK_MEDIUM"`)

**Mapping Logic**:
```python
stack_in_bb = player_stack / big_blind

if stack_in_bb > 100:
    return "STACK_DEEP"
elif stack_in_bb > 50:
    return "STACK_MEDIUM"
elif stack_in_bb > 20:
    return "STACK_SHALLOW"
else:
    return "STACK_SHORT"
```

#### `bucket_betting_sequence(action_history: List, current_street: str) -> str`

**Input**: List of action records, current street name  
**Output**: Betting sequence bucket ID (e.g., `"FLOP_raise_call"`)

**Mapping Logic**:
1. Extract last 2-3 actions from history
2. Convert action types to strings (check, call, raise, fold)
3. Join with underscores
4. Prefix with uppercase street name

---

## 3. Stage-Specific Abstraction Summary

The abstraction granularity varies by street (stage) to balance information preservation with state space reduction.

### 3.1 Preflop Abstraction

**Granularity**: **Most Detailed**

- **Hole Cards**: Full granularity
  - All pocket pairs distinguished (AA, KK, QQ, etc.)
  - All high card combinations distinguished (AK, AQ, AJ, etc.)
  - Suited/unsuited distinction maintained
  - Mid/low cards coarsely bucketed

- **Board**: No board (PREFLOP)

- **Betting**: Action patterns preserved (last 2-3 actions)

- **Pot/Stack**: Standard buckets

**Rationale**: Preflop decisions are critical and depend heavily on starting hand strength. Maximum granularity preserves important distinctions.

**Example Infoset IDs**:
```
p1:PREFLOP:PP_AA:NO_BOARD:PREFLOP_raise_call:POT_TINY:STACK_MEDIUM
p1:PREFLOP:SUITED_AK:NO_BOARD:PREFLOP_call:POT_TINY:STACK_MEDIUM
p1:PREFLOP:UNSUITED_72:NO_BOARD:PREFLOP_fold:POT_TINY:STACK_MEDIUM
```

### 3.2 Flop Abstraction

**Granularity**: **Detailed**

- **Hole Cards**: Same as preflop (full granularity)

- **Board**: Texture-based abstraction
  - Suit distribution: Rainbow, Two-tone, Monotone
  - Paired vs. unpaired
  - High vs. low cards (2+ high cards = HIGH, 0 high cards = LOW)
  - Specific ranks not preserved (only texture matters)

- **Betting**: Action patterns preserved

- **Pot/Stack**: Standard buckets

**Rationale**: Flop texture (wet/dry, paired/unpaired) is more important than specific ranks. Suit distribution indicates flush potential.

**Example Infoset IDs**:
```
p1:FLOP:PP_AA:FLOP_RAINBOW_HIGH:FLOP_check_raise:POT_SMALL:STACK_MEDIUM
p1:FLOP:SUITED_AK:FLOP_TWO_TONE_PAIRED:FLOP_call:POT_SMALL:STACK_MEDIUM
p1:FLOP:UNSUITED_72:FLOP_MONOTONE_LOW:FLOP_fold:POT_SMALL:STACK_MEDIUM
```

### 3.3 Turn Abstraction

**Granularity**: **Moderate**

- **Hole Cards**: Same as preflop (full granularity)

- **Board**: Texture-based abstraction
  - Suit distribution: Rainbow, Two-tone, Flush draw
  - Paired vs. unpaired
  - High/low distinction removed (less important on turn)

- **Betting**: Action patterns preserved

- **Pot/Stack**: Standard buckets

**Rationale**: Turn decisions focus on flush draws and made hands. High/low distinction less critical than on flop.

**Example Infoset IDs**:
```
p1:TURN:PP_AA:TURN_FLUSH_DRAW:TURN_raise_call:POT_MEDIUM:STACK_MEDIUM
p1:TURN:SUITED_AK:TURN_RAINBOW_PAIRED:TURN_check:POT_MEDIUM:STACK_MEDIUM
p1:TURN:UNSUITED_72:TURN_TWO_TONE:TURN_fold:POT_MEDIUM:STACK_MEDIUM
```

### 3.4 River Abstraction

**Granularity**: **Moderate**

- **Hole Cards**: Same as preflop (full granularity)

- **Board**: Texture-based abstraction
  - Suit distribution: Rainbow, Flush draw, Flush
  - Paired vs. unpaired
  - High/low distinction removed

- **Betting**: Action patterns preserved

- **Pot/Stack**: Standard buckets

**Rationale**: River decisions focus on made hands and flush potential. Texture matters more than specific ranks.

**Example Infoset IDs**:
```
p1:RIVER:PP_AA:RIVER_FLUSH:RIVER_raise:POT_LARGE:STACK_MEDIUM
p1:RIVER:SUITED_AK:RIVER_RAINBOW_PAIRED:RIVER_call:POT_LARGE:STACK_MEDIUM
p1:RIVER:UNSUITED_72:RIVER_FLUSH_DRAW_PAIRED:RIVER_fold:POT_LARGE:STACK_MEDIUM
```

### 3.5 Abstraction Granularity Comparison

| Stage | Hole Cards | Board | Betting | Overall Granularity |
|-------|-----------|-------|---------|---------------------|
| **Preflop** | Full (all pairs, high combos) | None | Full (2-3 actions) | **Highest** |
| **Flop** | Full (all pairs, high combos) | Texture + High/Low | Full (2-3 actions) | **High** |
| **Turn** | Full (all pairs, high combos) | Texture (no High/Low) | Full (2-3 actions) | **Moderate** |
| **River** | Full (all pairs, high combos) | Texture (no High/Low) | Full (2-3 actions) | **Moderate** |

### 3.6 Design Rationale

1. **Hole Cards**: Always full granularity because hand strength is critical at all stages
2. **Board**: Texture abstraction increases as more cards are revealed (specific ranks less important)
3. **Betting**: Always preserve action patterns (important for opponent modeling)
4. **Pot/Stack**: Standard buckets across all stages (relative sizes matter more than absolute)

---

## 4. Infoset ID Format

The complete infoset ID combines all buckets into a single string:

```
player_id:STREET:HOLE_BUCKET:BOARD_BUCKET:BETTING_BUCKET:POT_BUCKET:STACK_BUCKET
```

### Example Infoset IDs

**Preflop**:
```
p1:PREFLOP:PP_AA:NO_BOARD:PREFLOP_raise_call:POT_TINY:STACK_MEDIUM
```

**Flop**:
```
p1:FLOP:SUITED_AK:FLOP_RAINBOW_HIGH:FLOP_check_raise:POT_SMALL:STACK_MEDIUM
```

**Turn**:
```
p1:TURN:PP_KK:TURN_FLUSH_DRAW:TURN_call:POT_MEDIUM:STACK_SHALLOW
```

**River**:
```
p1:RIVER:UNSUITED_AQ:RIVER_FLUSH_PAIRED:RIVER_raise:POT_LARGE:STACK_DEEP
```

### Infoset ID Components

1. **player_id**: Player identifier (e.g., "p1", "p2")
2. **STREET**: Uppercase street name (PREFLOP, FLOP, TURN, RIVER)
3. **HOLE_BUCKET**: Hole card bucket (e.g., "PP_AA", "SUITED_AK")
4. **BOARD_BUCKET**: Board texture bucket (e.g., "FLOP_RAINBOW_HIGH")
5. **BETTING_BUCKET**: Betting sequence bucket (e.g., "FLOP_raise_call")
6. **POT_BUCKET**: Pot size bucket (e.g., "POT_SMALL")
7. **STACK_BUCKET**: Stack size bucket (e.g., "STACK_MEDIUM")

---

## 5. Usage Examples

### Example 1: Compute Infoset ID

```python
from backend.member2.bucketing import compute_infoset_id

infoset = compute_infoset_id(
    player_id="p1",
    hole_cards=["As", "Ah"],
    board=["Kd", "Qc", "Jh"],
    street="flop",
    action_history=[],
    pot=150,
    player_stack=850,
    big_blind=10
)

print(infoset)
# Output: p1:FLOP:PP_AA:FLOP_RAINBOW_HIGH:FLOP_NO_ACTION:POT_SMALL:STACK_MEDIUM
```

### Example 2: Individual Bucketing

```python
from backend.member2.bucketing import (
    bucket_hole_cards,
    bucket_board,
    bucket_pot_size,
    bucket_stack_ratio
)

# Hole cards
hole_bucket = bucket_hole_cards(["As", "Ks"])
print(hole_bucket)  # Output: SUITED_AK

# Board
board_bucket = bucket_board(["As", "Kd", "Qc"])
print(board_bucket)  # Output: FLOP_RAINBOW_HIGH

# Pot size
pot_bucket = bucket_pot_size(150, 10)
print(pot_bucket)  # Output: POT_SMALL

# Stack ratio
stack_bucket = bucket_stack_ratio(850, 150, 10)
print(stack_bucket)  # Output: STACK_MEDIUM
```

### Example 3: Integration with Game Engine

The infoset ID is automatically computed in `backend/main.py` when recording experiences:

```python
# In _record_experience function
infoset_id = compute_infoset_id(
    player_id=player_id,
    hole_cards=engine.hole_cards.get(player_id, []),
    board=list(engine.board),
    street=street,
    action_history=engine.betting.action_history[:-1],
    pot=engine.betting.pot,
    player_stack=engine.betting.stacks.get(player_id, 0),
    big_blind=engine.betting.big_blind,
)
```

---

## Summary

The Member 2B bucketing system provides:

1. **Comprehensive bucket definitions** for all game state components
2. **Efficient mapping functions** that convert raw state to bucket IDs
3. **Stage-specific abstractions** that balance information preservation with state space reduction
4. **Stable infoset IDs** that enable effective strategy learning and opponent modeling

The system reduces the state space from billions of possible states to thousands of abstracted information sets, making it feasible to learn and apply poker strategies using techniques like CFR (Counterfactual Regret Minimization).

