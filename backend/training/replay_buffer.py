from __future__ import annotations

import json
import random
from collections import deque
from typing import Deque, Dict, Iterable, List, Optional


class ReplayBuffer:
    """Simple experience replay buffer (JSONL-serializable, extensible records)."""

    def __init__(self, capacity: int = 10000, rng: Optional[random.Random] = None) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._capacity = capacity
        self._buffer: Deque[Dict[str, object]] = deque(maxlen=capacity)
        self._rng = rng or random.Random()

    @property
    def capacity(self) -> int:
        return self._capacity

    def __len__(self) -> int:
        return len(self._buffer)

    def add(self, experience: Dict[str, object]) -> None:
        self._buffer.append(dict(experience))

    def sample(self, batch_size: int) -> List[Dict[str, object]]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        items = list(self._buffer)
        if not items:
            return []
        size = min(batch_size, len(items))
        return self._rng.sample(items, size)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            for entry in self._buffer:
                handle.write(json.dumps(entry))
                handle.write("\n")

    @classmethod
    def load(cls, path: str, capacity: Optional[int] = None) -> "ReplayBuffer":
        entries: List[Dict[str, object]] = []
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                entries.append(json.loads(line))

        resolved_capacity = capacity or max(len(entries), 1)
        buffer = cls(capacity=resolved_capacity)
        for entry in entries[-buffer.capacity :]:
            buffer.add(entry)
        return buffer
