# Experiment Workflow

This repository uses YAML-driven runs so experiments remain reproducible and auditable.

## Single Training Run

```bash
python3 train.py --config baseline
```

Useful options:

- `--config <name-or-path>` from `configs/train/` or explicit YAML
- `--set key=value` for nested overrides (repeatable)
- `--run-name <name>` for deterministic output folder names
- `--train-seed <int>` and `--eval-seed <int>`
- `--no-wandb` to disable Weights & Biases logging

Default output root is `paths.experiments_root` (typically `experiments/train`):

- `config/resolved_config.yaml` and `.json`
- `metrics.jsonl`
- `checkpoints/` and `checkpoints.jsonl`
- `run_metadata.json`
- `run_summary.json`

## Batch Runs (Ray Tune)

`start_training.py` reads one trial per line from `start_training_config.txt`.

Example line:

```text
baseline --set environment.agents_num=40 --set training.episodes=800
```

Run:

```bash
python3 start_training.py start_training_config.txt --max_concurrent_trials 2
```

Use `--use_cnn_observation` to enforce `environment.use_cnn_observation=True` across all lines unless already set.

## Evaluation Workflow

```bash
python3 eval.py --config baseline --checkpoint 1
```

For details on metrics, outputs, and multi-map mode, see `README_eval.md`.

## Creating New Configs

```bash
cp configs/train/baseline.yaml configs/train/my_experiment.yaml
```

Sections to update:

- `run`: naming/logging metadata
- `hardware`: CPU/GPU and worker counts
- `model`: RLlib model stack
- `training`: episode counts, evaluation cadence
- `environment`: runtime env parameters
- `evaluation_environment`: eval-only env overrides
- `paths`: output/map roots

## Reproducibility Tips

- Keep `resolved_config.yaml` and checkpoint references with result artifacts.
- Prefer CLI `--set` overrides for small ablations to avoid config drift.
- Use numeric checkpoint shortcuts (`--checkpoint 1`, `--checkpoint 2`) only when the search roots are stable.
