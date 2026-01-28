# Poker EHS Simulator

Texas Hold'em Expected Hand Strength (EHS) via Monte Carlo simulation.

## Sampling parameters
- Rollouts: configurable (default 1000; recommended 500-5000 for speed/stability tradeoff).
- Opponents: configurable (default 1).
- Seed: optional for reproducible results.
- Parallelism: optional with `--jobs` using multiprocessing.
- Caching: optional disk cache via `--cache` (shelve DB).

## CLI usage
Compute EHS for a single state:
```
python -m poker.cli ehs --hero "As Kd" --board "2c 7d Jh" --rollouts 2000 --seed 42
```

Export a dataset from JSON/JSONL:
```
python -m poker.cli dataset --input samples.jsonl --output ehs.csv --rollouts 1500 --seed 7
```

Play a CLI Texas Hold'em game:
```
python -m poker.cli play --players 4 --stack 1000 --small-blind 5 --big-blind 10 --ai-style balanced --show-ehs
```

### Input format
`*.json` or `*.jsonl` containing objects like:
```
{"hero": "As Kd", "board": "2c 7d Jh"}
```
or
```
{"hero": ["As", "Kd"], "board": ["2c", "7d", "Jh"]}
```

## Library usage
```
from poker.cards import parse_cards
from poker.ehs import EHSEngine

engine = EHSEngine(rollouts=2000, n_opponents=1, seed=42, n_jobs=2)
hero = parse_cards("As Kd")
board = parse_cards("2c 7d Jh")
result = engine.compute(hero, board)
print(result.ehs)
```

## Notes
- This evaluator computes the best 5-card hand out of 7 cards.
- Results are Monte Carlo estimates; higher rollouts reduce variance.

