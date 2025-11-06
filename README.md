# RLMAPF2

RLMAPF2 trains PPO policies for multi-agent path finding (MAPF) problems with
optional D* Lite guidance. The project is built on top of Ray RLlib and ships
with structured experiment configs so you can reproduce runs without editing the
code.

## Requirements

- Python 3.9+
- GPU optional but recommended for larger experiments
- Python packages (see `requirements.txt` for exact versions):
  - `ray[rllib]`
  - `torch`
  - `gymnasium`
  - `numpy`
  - `PyYAML`
  - `matplotlib`
  - `tqdm`
  - `wandb` (optional, disable with `--no-wandb`)

Install the dependencies in a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Quick Start

Train a single run using one of the YAML configs in `configs/train/`:

```bash
python3 train.py --config baseline
```

This command creates `experiments/<run-name>/` containing:

- `config/resolved_config.yaml` and `.json` – exact config used
- `metrics.jsonl` – episode metrics stored locally
- `checkpoints/` plus `checkpoints.jsonl` – saved policies and index
- `run_summary.json` – final checkpoint and best metrics
- `run_metadata.json` – timestamps, git commit, Ray namespace

Use overrides to adjust parameters without editing files:

```bash
python3 train.py --config baseline \
  --set environment.agents_num=40 \
  --set training.episodes=800
```

Useful flags:

- `--config <name-or-path>` – load config by name (`baseline`, `cnn`, etc.) or an
  explicit path
- `--set key=value` – override any nested value using dot notation; repeat as
  needed
- `--run-name my-run` – force a specific run directory name
- `--seed 123` / `--train-seed 123` – set `training.random_seed` (training environment seed). If omitted, a random seed is generated for each run.
- `--eval-seed 42` – set `training.evaluation_seed` (evaluation environment seed). Defaults to 42 when not provided.
- `--no-wandb` – disable Weights & Biases logging

Legacy arguments (`--agents_num`, `--max_steps`, etc.) still work; they are
translated into the corresponding overrides.

## Batch Experiments with Ray Tune

`start_training.py` schedules multiple runs using Ray Tune. Define one line per
trial in `start_training_config.txt` using the same syntax as the single-run
entrypoint:

```shell
baseline --set environment.agents_num=40 --set training.episodes=800
```

Launch the sweep:

```shell
python3 start_training.py start_training_config.txt \
  --max_concurrent_trials 2
```

The `--use_cnn_observation` flag automatically appends
`--set environment.use_cnn_observation=True` to each trial unless already set.

## maps/

The `maps/` directory contains JSON map files used by the environment. Each map specifies the layout, number of agents, and (optionally) start/goal positions.

To generate new map files programmatically, edit `maps/json_generator_map.txt`
with your ASCII layout and run the helper script:

```shell
python3 maps/json_generator.py \
  --name my_custom_map \
  --agents 4-4 \
  --flip_h --rot90
```

Flags control optional augmentations (`--flip_*`, `--rot*`, `--translate`,
`--swap`). Use `--map_txt_file` to point at an alternate ASCII layout. The
script reads the template, produces variants, previews the JSON, and writes
`<name>_<agents>a-<width>x<height>.json` into `maps/`.

Set `environment.start_goal_on_periphery=true` in a train config (or via
`--set`) to override any map-provided spawn points. When enabled, agents are
placed uniformly along the accessible border and their goals are mirrored to
the diametrically opposite border cell (e.g. `(0, 0)` pairs with
`(width-1, height-1)`). Ensure the chosen maps have obstacle-free perimeter
cells; otherwise the run will error during reset.

## Experiment Documentation

`docs/experiments.md` expands on the workflow, including tips for creating new
YAML configs, interpreting run outputs, and tuning evaluation cadence.

## Testing the Environment Without RL

You can step through the `RLMAPF` environment directly to validate map or
reward changes without launching PPO.

```bash
python3 -m rlmapf2
```
