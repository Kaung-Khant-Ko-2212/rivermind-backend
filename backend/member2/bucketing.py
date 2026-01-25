from __future__ import annotations

from typing import Dict, List, Optional, Tuple


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
    
    # Texture classification
    if board_size == 3:  # Flop
        if max_suit_count == 3:
            texture = "MONOTONE"
        elif max_suit_count == 2:
            texture = "TWO_TONE"
        else:
            texture = "RAINBOW"
        
        if max_rank_count >= 2:
            texture += "_PAIRED"
        
        if high_cards >= 2:
            texture += "_HIGH"
        elif high_cards == 0:
            texture += "_LOW"
        
        return f"FLOP_{texture}"
    
    elif board_size == 4:  # Turn
        if max_suit_count >= 3:
            texture = "FLUSH_DRAW"
        elif max_suit_count == 2:
            texture = "TWO_TONE"
        else:
            texture = "RAINBOW"
        
        if max_rank_count >= 2:
            texture += "_PAIRED"
        
        return f"TURN_{texture}"
    
    elif board_size == 5:  # River
        if max_suit_count >= 5:
            texture = "FLUSH"
        elif max_suit_count >= 4:
            texture = "FLUSH_DRAW"
        else:
            texture = "RAINBOW"
        
        if max_rank_count >= 2:
            texture += "_PAIRED"
        
        return f"RIVER_{texture}"
    
    return f"BOARD_{board_size}"


def bucket_betting_sequence(action_history: List, current_street: str) -> str:
    """
    Bucket betting sequence by action pattern.
    Returns bucket ID like 'PREFLOP_RAISE_CALL', 'FLOP_CHECK_CHECK', etc.
    """
    street_upper = current_street.upper()
    if not action_history:
        return f"{street_upper}_NO_ACTION"
    
    # Get actions for current street only
    # For simplicity, we'll use the last few actions as pattern
    actions = []
    for record in action_history[-6:]:  # Last 6 actions
        try:
            if hasattr(record, 'action'):
                # ActionRecord object with action field
                if hasattr(record.action, 'action'):
                    # Action object with action field (ActionType enum)
                    action_type = record.action.action.value
                else:
                    action_type = str(record.action)
                actions.append(action_type)
            elif isinstance(record, dict):
                # Dictionary format
                action_obj = record.get('action', {})
                if isinstance(action_obj, dict):
                    action_type = action_obj.get('action', 'unknown')
                else:
                    action_type = str(action_obj)
                actions.append(action_type)
        except (AttributeError, KeyError):
            # Skip malformed records
            continue
    
    if not actions:
        return f"{street_upper}_NO_ACTION"
    
    # Simplify: use pattern of last 2-3 actions
    pattern = "_".join(actions[-3:]) if len(actions) >= 3 else "_".join(actions)
    return f"{street_upper}_{pattern}"


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


def compute_infoset_id(
    player_id: str,
    hole_cards: List[str],
    board: List[str],
    street: str,
    action_history: List,
    pot: int,
    player_stack: int,
    big_blind: int = 10,
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
    
    # Combine into infoset ID
    infoset_parts = [
        player_id,
        street.upper(),
        hole_bucket,
        board_bucket,
        betting_bucket,
        pot_bucket,
        stack_bucket,
    ]
    
    return ":".join(infoset_parts)

