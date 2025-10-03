# Experiment Workflow

This project now uses YAML-based configurations to describe training runs. Each
config captures the environment, PPO model, logging, and file-system layout so
you can reproduce an experiment without digging through scripts.

## Single Run

```bash
python3 train.py --config baseline
```

Useful flags:

- `--config <name-or-path>`: load a config from `configs/train/` or an explicit
  path. Examples: `baseline`, `cnn`, or `configs/train/custom.yaml`.
- `--set key=value`: override any nested value using dot notation. E.g.
  `--set environment.agents_num=40` or
  `--set training.eval_interval=25`. Repeat `--set` to change multiple fields.
- `--run-name custom-name`: override the generated run directory name.
- `--no-wandb`: disable Weights & Biases even if the config enables it.
- Legacy Ray tune arguments like `--agents_num` are still accepted and converted
  into overrides for backwards compatibility.

Each run creates `experiments/<run-name>/` with:

- `config/` – resolved YAML/JSON config used for the run.
- `run_metadata.json` – git commit (if available), timestamp, and directories.
- `metrics.jsonl` – per-episode metrics logged locally.
- `checkpoints/` – saved policies with an index in `checkpoints.jsonl`.
- `run_summary.json` – final checkpoint plus best metrics summary.

## Batch Runs with Ray Tune

`start_training.py` now reads the same override syntax. Each non-comment line in
`start_training_config.txt` can be a config name optionally followed by
`--set ...` overrides. Ray Tune runs a grid search over the lines.

Example configuration file entry:

```
baseline --set environment.agents_num=40 --set training.episodes=800
```

Launch multiple runs:

```bash
python3 start_training.py start_training_config.txt --max_concurrent_trials 2
```

Use `--use_cnn_observation` if you want every run to enable the CNN observation
flag without editing each line; the script appends
`--set environment.use_cnn_observation=True` when missing.

## Creating New Configs

Copy one of the templates in `configs/train/` and adjust the sections you need:

```bash
cp configs/train/baseline.yaml configs/train/my_experiment.yaml
```

Key sections:

- `run`: naming, wandb project/group, tags.
- `training`: episode counts, evaluation cadence.
- `environment`: values forwarded to `RLMAPF`.
- `evaluation_environment`: overrides for evaluation runs.
- `logging`: metrics to track locally and in wandb.

Commit the new config with your experiment results so the history stays
traceable.
