from pathlib import Path

from backend.training.replay_buffer import ReplayBuffer


def test_add_and_sample() -> None:
    buffer = ReplayBuffer(capacity=3)
    buffer.add({"timestamp": 1, "street": "preflop", "player_to_act": "p1"})
    buffer.add({"timestamp": 2, "street": "flop", "player_to_act": "p2"})

    batch = buffer.sample(1)
    assert len(batch) == 1
    assert batch[0]["player_to_act"] in {"p1", "p2"}


def test_capacity_eviction() -> None:
    buffer = ReplayBuffer(capacity=2)
    buffer.add({"id": 1})
    buffer.add({"id": 2})
    buffer.add({"id": 3})

    ids = [entry["id"] for entry in buffer.sample(5)]
    assert set(ids) == {2, 3}


def test_save_load_roundtrip(tmp_path: Path) -> None:
    buffer = ReplayBuffer(capacity=3)
    buffer.add({"id": 1, "value": "a"})
    buffer.add({"id": 2, "value": "b"})

    path = tmp_path / "replay.jsonl"
    buffer.save(str(path))

    loaded = ReplayBuffer.load(str(path), capacity=3)
    assert len(loaded) == 2
    entries = loaded.sample(10)
    assert {entry["id"] for entry in entries} == {1, 2}
