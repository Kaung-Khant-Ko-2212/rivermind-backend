from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple


def _get_rank(card: str) -> int:
    """Get rank value (2=2, A=14) from card string."""
    rank_ch = card[0]
    rank_map = {
        "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9,
        "T": 10, "J": 11, "Q": 12, "K": 13, "A": 14
    }
    return rank_map.get(rank_ch, 0)


def _get_suit(card: str) -> str:
    """Get suit from card string."""
    return card[1]


def bucket_hole_cards(hole_cards: List[str]) -> str:
    """
    Bucket hole cards into abstract categories.
    Returns a bucket ID string like 'PP_AA', 'SUITED_AK', 'UNSUITED_AK', etc.
    """
    if len(hole_cards) != 2:
        return "INVALID"
    
    card1, card2 = hole_cards
    rank1, rank2 = _get_rank(card1), _get_rank(card2)
    suit1, suit2 = _get_suit(card1), _get_suit(card2)
    
    # Normalize: higher rank first
    if rank1 < rank2:
        rank1, rank2 = rank2, rank1
    
    is_pair = rank1 == rank2
    is_suited = suit1 == suit2
    
    if is_pair:
        rank_names = {14: "A", 13: "K", 12: "Q", 11: "J", 10: "T"}
        rank_name = rank_names.get(rank1, str(rank1))
        return f"PP_{rank_name}{rank_name}"
    
    # High card combinations
    rank_names = {14: "A", 13: "K", 12: "Q", 11: "J", 10: "T", 9: "9", 8: "8"}
    rank1_name = rank_names.get(rank1, "LOW")
    rank2_name = rank_names.get(rank2, "LOW")
    
    # Only bucket high cards (A, K, Q, J, T, 9, 8)
    if rank1 >= 8:
        prefix = "SUITED" if is_suited else "UNSUITED"
        return f"{prefix}_{rank1_name}{rank2_name}"
    
    # Low cards get coarser buckets
    if rank1 >= 6:
        prefix = "SUITED" if is_suited else "UNSUITED"
        return f"{prefix}_MID"
    
    prefix = "SUITED" if is_suited else "UNSUITED"
    return f"{prefix}_LOW"


def bucket_board(board: List[str]) -> str:
    """
    Bucket board cards by texture.
    Returns bucket ID like 'FLOP_RAINBOW', 'FLOP_TWO_TONE', 'TURN_PAIRED', etc.
    """
    if not board:
        return "PREFLOP"
    
    board_size = len(board)
    ranks = [_get_rank(card) for card in board]
    suits = [_get_suit(card) for card in board]
    
    # Count suits
    suit_counts = {}
    for suit in suits:
        suit_counts[suit] = suit_counts.get(suit, 0) + 1
    max_suit_count = max(suit_counts.values()) if suit_counts else 0
    
    # Count ranks (for pairs, trips, etc.)
    rank_counts = {}
    for rank in ranks:
        rank_counts[rank] = rank_counts.get(rank, 0) + 1
    max_rank_count = max(rank_counts.values()) if rank_counts else 0
    
    # High cards (A, K, Q, J, T)
    high_cards = sum(1 for r in ranks if r >= 10)

    def _suit_texture() -> str:
        if board_size == 3:
            if max_suit_count == 3:
                return "MONOTONE"
            if max_suit_count == 2:
                return "TWO_TONE"
            return "RAINBOW"
        if board_size == 4:
            if max_suit_count >= 4:
                return "FOUR_FLUSH"
            if max_suit_count == 3:
                return "FLUSH_DRAW"
            if max_suit_count == 2:
                return "TWO_TONE"
            return "RAINBOW"
        if board_size >= 5:
            if max_suit_count >= 5:
                return "FLUSH_BOARD"
            if max_suit_count == 4:
                return "FOUR_FLUSH"
            if max_suit_count == 3:
                return "THREE_FLUSH"
            return "RAINBOW"
        return "UNKNOWN"

    def _pair_texture() -> str:
        if max_rank_count >= 4:
            return "QUADS_BOARD"
        if max_rank_count == 3 and len(rank_counts) <= board_size - 2:
            return "FULL_BOARD"
        if max_rank_count == 3:
            return "TRIPS_BOARD"
        if list(rank_counts.values()).count(2) >= 2:
            return "TWO_PAIR_BOARD"
        if max_rank_count == 2:
            return "PAIRED"
        return "UNPAIRED"

    def _highness() -> str:
        if high_cards >= 3:
            return "HIGH_HEAVY"
        if high_cards >= 1:
            return "MIXED"
        return "LOW"

    def _connectivity() -> str:
        if len(ranks) < 3:
            return "NOCONN"
        unique = sorted(set(ranks))
        if 14 in unique:
            unique = sorted(set(unique + [1]))
        max_run = 1
        run = 1
        for i in range(1, len(unique)):
            if unique[i] - unique[i - 1] == 1:
                run += 1
                max_run = max(max_run, run)
            else:
                run = 1
        span = max(unique) - min(unique) if unique else 99
        if max_run >= 4:
            return "STRAIGHTY"
        if max_run == 3 or span <= 5:
            return "CONNECTED"
        if span <= 8:
            return "SEMI_CONNECTED"
        return "DISCONNECTED"

    prefix = {3: "FLOP", 4: "TURN", 5: "RIVER"}.get(board_size, f"BOARD_{board_size}")
    return "_".join(
        [
            prefix,
            _suit_texture(),
            _pair_texture(),
            _connectivity(),
            _highness(),
        ]
    )


def _extract_action_type(record: Any) -> Optional[str]:
    try:
        if hasattr(record, "action"):
            action_obj = getattr(record, "action")
            if hasattr(action_obj, "action"):
                action_enum = getattr(action_obj, "action")
                return getattr(action_enum, "value", str(action_enum)).lower()
            return str(action_obj).lower()
        if isinstance(record, dict):
            action_obj = record.get("action", {})
            if isinstance(action_obj, dict):
                return str(action_obj.get("action", "unknown")).lower()
            return str(action_obj).lower()
    except (AttributeError, KeyError, TypeError, ValueError):
        return None
    return None


def bucket_betting_sequence(action_history: List, current_street: str) -> str:
    """
    Bucket betting sequence by action pattern.
    Returns bucket ID like 'PREFLOP_RAISE_CALL', 'FLOP_CHECK_CHECK', etc.
    """
    street_upper = current_street.upper()
    if not action_history:
        return f"{street_upper}_NO_ACTION"
    
    actions: List[str] = []
    for record in action_history[-10:]:
        action_type = _extract_action_type(record)
        if action_type:
            actions.append(action_type)
    
    if not actions:
        return f"{street_upper}_NO_ACTION"
    
    raises = sum(1 for a in actions if a == "raise")
    calls = sum(1 for a in actions if a == "call")
    checks = sum(1 for a in actions if a == "check")
    folds = sum(1 for a in actions if a == "fold")

    if raises >= 3:
        aggression = "RAISE_WAR"
    elif raises == 2:
        aggression = "RERAISED"
    elif raises == 1:
        aggression = "SINGLE_RAISE"
    else:
        aggression = "UNRAISED"

    opener = actions[0].upper() if actions else "NONE"
    tail = "-".join(a[:1].upper() for a in actions[-4:])
    shape = f"N{len(actions)}_R{raises}_C{calls}_K{checks}_F{folds}"
    return f"{street_upper}_{aggression}_{opener}_{tail}_{shape}"


def bucket_pot_size(pot: int, big_blind: int = 10) -> str:
    """
    Bucket pot size relative to big blind.
    Returns bucket ID like 'POT_SMALL', 'POT_MEDIUM', 'POT_LARGE', etc.
    """
    if big_blind == 0:
        return "POT_UNKNOWN"
    
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


def bucket_stack_ratio(player_stack: int, pot: int, big_blind: int = 10) -> str:
    """
    Bucket effective stack ratio.
    Returns bucket ID like 'STACK_DEEP', 'STACK_MEDIUM', 'STACK_SHALLOW', etc.
    """
    if big_blind == 0:
        return "STACK_UNKNOWN"
    
    stack_in_bb = player_stack / big_blind
    
    if stack_in_bb > 100:
        return "STACK_DEEP"
    elif stack_in_bb > 50:
        return "STACK_MEDIUM"
    elif stack_in_bb > 20:
        return "STACK_SHALLOW"
    else:
        return "STACK_SHORT"


def bucket_spr(player_stack: int, pot: int) -> str:
    """
    Bucket stack-to-pot ratio (SPR), a stronger planning signal than raw stack alone.
    """
    if pot <= 0:
        return "SPR_INF"
    spr = player_stack / max(1, pot)
    if spr < 1:
        return "SPR_LT1"
    if spr < 2:
        return "SPR_1_2"
    if spr < 4:
        return "SPR_2_4"
    if spr < 8:
        return "SPR_4_8"
    if spr < 16:
        return "SPR_8_16"
    return "SPR_16P"


def bucket_to_call_pressure(to_call: Optional[int], pot: int, big_blind: int = 10) -> str:
    """
    Bucket call pressure by absolute and pot-relative cost.
    """
    if to_call is None:
        return "CALL_UNKNOWN"
    if to_call <= 0:
        return "CALL_FREE"
    if pot <= 0:
        if big_blind <= 0:
            return "CALL_POSTED"
        return "CALL_SMALL" if to_call <= big_blind else "CALL_PRESSURE"

    ratio = to_call / max(1, pot)
    if ratio <= 0.10:
        return "CALL_TINY"
    if ratio <= 0.25:
        return "CALL_SMALL"
    if ratio <= 0.50:
        return "CALL_MEDIUM"
    if ratio <= 1.00:
        return "CALL_LARGE"
    return "CALL_ALLINISH"


def bucket_position(
    player_id: str,
    button_player: Optional[str] = None,
    active_players: Optional[Iterable[str]] = None,
) -> str:
    """
    Bucket seat position/role relative to button.
    """
    ordered_players: List[str] = []
    if active_players is not None:
        ordered_players = [p for p in active_players if isinstance(p, str)]
    if not ordered_players:
        # Fallback to a stable seat ordering guess.
        ordered_players = [f"p{i}" for i in range(1, 6)]

    if player_id not in ordered_players:
        ordered_players.append(player_id)

    # Keep seat order stable (`p1`..`p5`) when possible.
    def _seat_sort_key(seat: str) -> Tuple[int, str]:
        suffix = seat[1:]
        return (int(suffix), seat) if seat.startswith("p") and suffix.isdigit() else (9999, seat)

    ordered_players = sorted(dict.fromkeys(ordered_players), key=_seat_sort_key)
    n_players = len(ordered_players)

    if n_players <= 1:
        return "POS_SOLO"

    if button_player not in ordered_players:
        button_player = ordered_players[0]

    btn_idx = ordered_players.index(button_player)
    player_idx = ordered_players.index(player_id)
    offset = (player_idx - btn_idx) % n_players

    if n_players == 2:
        return "POS_BTN" if offset == 0 else "POS_BB"

    if offset == 0:
        return "POS_BTN"
    if offset == 1:
        return "POS_SB"
    if offset == 2:
        return "POS_BB"

    remaining_after_blinds = n_players - 3
    if remaining_after_blinds <= 1:
        return "POS_UTG"

    rel = offset - 3
    if rel == remaining_after_blinds - 1:
        return "POS_CO"
    if rel <= 0:
        return "POS_UTG"
    if rel <= max(1, remaining_after_blinds // 2):
        return "POS_MP"
    return "POS_HJ"


def compute_infoset_id(
    player_id: str,
    hole_cards: List[str],
    board: List[str],
    street: str,
    action_history: List,
    pot: int,
    player_stack: int,
    big_blind: int = 10,
    bets: Optional[Dict[str, int]] = None,
    to_call: Optional[int] = None,
    button_player: Optional[str] = None,
    active_players: Optional[Iterable[str]] = None,
    schema_version: str = "v2",
) -> str:
    """
    Compute a stable infoset ID by combining all abstractions.
    
    Args:
        player_id: The player making the decision
        hole_cards: Player's hole cards (2 cards)
        board: Community board cards
        street: Current street (preflop, flop, turn, river)
        action_history: List of action records
        pot: Current pot size
        player_stack: Player's remaining stack
        big_blind: Big blind size (default 10)
    
    Returns:
        A stable infoset ID string like:
        'p1:PREFLOP:PP_AA:PREFLOP:PREFLOP_RAISE_CALL:POT_SMALL:STACK_DEEP'
    """
    hole_bucket = bucket_hole_cards(hole_cards) if hole_cards else "NO_HOLE"
    board_bucket = bucket_board(board) if board else "NO_BOARD"
    betting_bucket = bucket_betting_sequence(action_history, street)
    pot_bucket = bucket_pot_size(pot, big_blind)
    stack_bucket = bucket_stack_ratio(player_stack, pot, big_blind)

    if schema_version.lower() in {"v1", "legacy"}:
        return ":".join(
            [
                player_id,
                street.upper(),
                hole_bucket,
                board_bucket,
                betting_bucket,
                pot_bucket,
                stack_bucket,
            ]
        )

    active_player_list = None
    if active_players is not None:
        active_player_list = [p for p in active_players if isinstance(p, str)]

    pos_bucket = bucket_position(
        player_id=player_id,
        button_player=button_player,
        active_players=active_player_list,
    )
    spr_bucket = bucket_spr(player_stack=player_stack, pot=pot)
    pressure_bucket = bucket_to_call_pressure(to_call=to_call, pot=pot, big_blind=big_blind)

    contribution_bucket = "BET_UNKNOWN"
    if isinstance(bets, dict):
        try:
            contribution = int(bets.get(player_id, 0) or 0)
            if contribution <= 0:
                contribution_bucket = "BET_0"
            elif contribution <= big_blind:
                contribution_bucket = "BET_SMALL"
            elif contribution <= max(1, big_blind * 3):
                contribution_bucket = "BET_MED"
            else:
                contribution_bucket = "BET_LARGE"
        except (TypeError, ValueError):
            contribution_bucket = "BET_UNKNOWN"

    active_count = 0
    if active_player_list is not None:
        active_count = len(active_player_list)
    if active_count <= 0 and isinstance(bets, dict) and bets:
        active_count = len(bets)
    if active_count <= 0:
        active_count = 2
    table_bucket = f"N{min(5, max(2, active_count))}"

    infoset_parts = [
        "IS2",
        player_id,
        table_bucket,
        street.upper(),
        pos_bucket,
        hole_bucket,
        board_bucket,
        betting_bucket,
        pot_bucket,
        stack_bucket,
        spr_bucket,
        pressure_bucket,
        contribution_bucket,
    ]

    return ":".join(infoset_parts)

