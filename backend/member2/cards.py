from __future__ import annotations

RANKS = "23456789TJQKA"
SUITS = "cdhs"  # clubs, diamonds, hearts, spades


def card_to_int(card: str) -> int:
    """Convert a card like 'As' or 'Td' to an int 0-51."""
    if len(card) != 2:
        raise ValueError(f"Invalid card string: {card}")
    rank_ch, suit_ch = card[0], card[1]
    try:
        rank = RANKS.index(rank_ch)
        suit = SUITS.index(suit_ch)
    except ValueError as exc:
        raise ValueError(f"Invalid card string: {card}") from exc
    return suit * 13 + rank


def int_to_card(value: int) -> str:
    if not 0 <= value < 52:
        raise ValueError(f"Invalid card int: {value}")
    rank = value % 13
    suit = value // 13
    return f"{RANKS[rank]}{SUITS[suit]}"


def parse_cards(cards: list[str] | str | None) -> list[int]:
    """Parse cards from list like ['As','Kd'] or string like 'As Kd'."""
    if cards is None:
        return []
    if isinstance(cards, str):
        parts = [p for p in cards.replace(",", " ").split() if p]
    else:
        parts = cards
    values = [card_to_int(p) for p in parts]
    if len(set(values)) != len(values):
        raise ValueError("Duplicate cards found in input")
    return values


def deck_excluding(exclude: list[int]) -> list[int]:
    excluded = set(exclude)
    if len(excluded) != len(exclude):
        raise ValueError("Duplicate cards found in exclusion list")
    return [c for c in range(52) if c not in excluded]
