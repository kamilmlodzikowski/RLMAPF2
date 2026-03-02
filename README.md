# RLMAPF2

RLMAPF2 is a MAPF research codebase for training and evaluating multi-agent PPO
policies with optional congestion-aware D* Lite guidance.

The core pipeline is config-driven (YAML), reproducible, and built on Ray RLlib.

## Highlights

- Shared-policy PPO training for grid MAPF.
- Optional D* Lite path guidance injected into local observations.
- Structured experiment outputs (resolved config, metadata, metrics, checkpoints).
- Evaluation harness with repeat runs, cross-agent scaling, CSV exports, and optional videos.
- Batch-run launcher for Ray Tune sweeps.

## Requirements

- Python 3.9+
- Optional GPU for larger experiments
- Dependencies pinned in `requirements.txt`

Quick setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Repository Layout

- `train.py`: single-run PPO training entrypoint
- `start_training.py`: batch launcher (one config line per trial)
- `eval.py`: checkpoint evaluation and reporting
- `rlmapf2.py`: multi-agent environment implementation
- `configs/train/`: training configs
- `configs/eval/`: evaluation configs
- `maps/`: JSON map definitions and map generation helpers
- `docs/`: extended workflow docs
- `tests/`: unit tests

## Training

Run a baseline training:

```bash
python3 train.py --config baseline
```

Run with overrides:

```bash
python3 train.py --config baseline \
  --set environment.agents_num=40 \
  --set training.episodes=800
```

Useful flags:

- `--config <name-or-path>`
- `--set key=value` (repeatable, dot-path syntax)
- `--run-name <name>`
- `--train-seed <int>` / `--eval-seed <int>`
- `--no-wandb`

Artifacts are written under the configured experiments root
(default `experiments/train/<run-name>/`), including:

- `config/resolved_config.yaml` and `config/resolved_config.json`
- `metrics.jsonl`
- `checkpoints/` and `checkpoints.jsonl`
- `run_metadata.json`
- `run_summary.json`

## Batch Training

`start_training.py` reads one trial per line from `start_training_config.txt`.

Example line:

```text
baseline --set environment.agents_num=40 --set training.episodes=800
```

Launch:

```bash
python3 start_training.py start_training_config.txt --max_concurrent_trials 2
```

## Evaluation

Run evaluation against a checkpoint:

```bash
python3 eval.py --config baseline --checkpoint 1
```

- Numeric checkpoints resolve as `1 = most recent`, `2 = second most recent`, etc.
- Full details and plotting guidance are in `README_eval.md`.

## Maps

Map files are stored in `maps/*.json`.

Generate a map from an ASCII template:

```bash
python3 maps/json_generator.py --name my_map --agents 4-4 --flip_h --rot90
```

Use `environment.start_goal_on_periphery=true` to spawn on obstacle-free border
cells with mirrored goals.

## Local Validation

Run tests:

```bash
pytest -q tests
```

Run the environment module directly for manual stepping/debug:

```bash
python3 -m rlmapf2
```

## Documentation

- `README_eval.md`: evaluation usage and metric interpretation
- `docs/experiments.md`: config workflow and experiment organization
- `CONTRIBUTING.md`: contribution process
- `SECURITY.md`: vulnerability reporting
- `THIRD_PARTY_NOTICES.md`: external component license notes

## License

This repository is released under the MIT License (`LICENSE`).
