# RLMAPF Evaluation

Evaluate trained PPO agents on multi-agent pathfinding tasks.

## Requirements

```bash
pip install ray[rllib] numpy pandas pyyaml matplotlib
```

## Purpose

Load an eval YAML, restore a trained checkpoint, run the RLMAPF environment across agent counts (and optional multiple maps), collect key metrics (success/deadlock, efficiency, safety), and emit CSVs/plots/videos for comparison.

### Without Video (Fast)

```bash
python eval.py \
    --config baseline \
    --checkpoint 1
```

### With Video Rendering

```bash
python eval.py \
    --config baseline \
    --checkpoint 1 \
    --render-video \
    --video-agents 20
```

## Parameters

### Required
- `--config NAME` - Config file (in `configs/eval/`) or path to YAML
- `--checkpoint N` - Checkpoint number (1 = most recent group, 2 = next group, etc.) or full path

### Optional
- `--checkpoint-group NAME|PATH` - Filter runs by group/name_prefix or by a specific run dir
- `--checkpoint-strategy {best,latest,oldest}` - Which checkpoint inside the selected run (default: best by success rate)
- `--checkpoint-success-tolerance FLOAT` - When strategy=best, accept checkpoints within tolerance of the best success rate and pick the oldest among them (default: 1.0)
- `--sync-train-config` - Override eval env/model settings with those recorded in the training run’s resolved_config.yaml
- `--clamp-agents-to-train` - Force evaluation to the training `agents_num` found in the checkpoint config
- `--success-threshold FLOAT` - Goal completion percent required to mark an episode as success (default 99.9)
- `--render-video` - Enable video generation
- `--video-agents LIST` - Comma-separated numbers and/or ranges (e.g. `12,16-20`) that should get videos
  when `--render-video` is set. Default: all agent counts in the evaluation range.
- `--repeats N` - Number of evaluation repeats (default: from config)
- `--num-threads N` - Parallel evaluation threads (default: 1)
- `--set KEY=VALUE` - Override config values

### Video Rendering Options

Videos now support **smooth motion** with interpolated agent movements between steps:
- `eval_smooth_motion: true/false` - Enable smooth interpolation (default: true)
- `eval_motion_frames: N` - Interpolation frames per step (default: 5, higher = smoother)
- `eval_video_fps: N` - Base FPS (default: 10, actual FPS = fps × motion_frames when smooth)

### Override Examples

```bash
# Test different agent range
--set eval_agents_range=10-30

# Change max steps
--set environment.max_steps=500

# Disable smooth motion for faster rendering
--set eval_smooth_motion=false

# Increase smoothness (more interpolation frames)
--set eval_motion_frames=10

# Multiple overrides
--set eval_agents_range=4-10 --repeats 5

# Use the training agents_num only (ignores eval range)
--clamp-agents-to-train

# Relax success definition to partial completion
--success-threshold 80
```

## Output

Results are written under `paths.experiments_root` from the eval config (default `experiments/eval/`).
If `run.name_prefix` is set (e.g., `cnn`), the tool nests runs like `experiments/eval/cnn/{run-prefix}-{timestamp}/`.
Otherwise it falls back to `experiments/eval/{run-prefix}-{timestamp}/`.
- `summary.csv` - Aggregated results
- `final_results.csv` - Detailed per-episode data
- `evaluation_{N}agents_repeat{R}.mp4` - Videos (if enabled, R is zero-based)
Plots (saved under `plots/`):
- `success_vs_agents.png` (plus `episode_success_vs_agents.png` when available)
- `timeout_or_deadlock_vs_agents.png`
- `throughput_vs_agents.png`
- `collisions_vs_agents.png` (total/agent-agent/agent-obstacle with shaded CI)
- `efficiency_vs_agents.png` (path efficiency; uses D* paths when available, otherwise step-based proxy)
- `wait_fraction_vs_agents.png`
- `makespan_vs_agents.png` (makespan on successes and steps-to-50% completion fallback)
- `dashboard_reliability.png` (success, deadlock, throughput, makespan)
- `dashboard_behavior.png` (collisions, wait fraction, efficiency, efficiency-vs-collisions scatter)
- `goal_completion_heatmap.png` (agents reaching goals per repeat/agent count)
- `tradeoff_efficiency_vs_collisions.png`
- `maps_success_heatmap.png` (multi-map runs only)

## Common Commands

```bash
# Quick test (few agents, few repeats) - use most recent checkpoint
python eval.py --config baseline --checkpoint 1 \
    --set eval_agents_range=4-6 --repeats 2

# Full evaluation (statistics only, no video)
python eval.py --config baseline --checkpoint 1 \
    --set eval_agents_range=4-20 --repeats 10 --num-threads 4

# Generate single demo video
python eval.py --config baseline --checkpoint 1 \
    --render-video --video-agents 20

# Full evaluation + one video
python eval.py --config baseline --checkpoint 1 \
    --render-video --video-agents 20 \
    --set eval_agents_range=4-20 --repeats 10

# Use second most recent checkpoint
python eval.py --config baseline --checkpoint 2 \
    --render-video --video-agents 20

# Pick checkpoints from a specific group (e.g., cnn runs), best checkpoint within that group
python eval.py --config cnn --checkpoint 1 \
    --checkpoint-group cnn --checkpoint-strategy best

# Show grouped checkpoints (printed automatically at start, also in checkpoint_groups.json)
python eval.py --config baseline --checkpoint 1

# Force latest checkpoint in a run regardless of metrics
python eval.py --config baseline --checkpoint 1 \
    --checkpoint-strategy latest

# Multi-map evaluation (iterates over multiple layouts)
python eval.py --config multi_map --checkpoint 1 \
    --set eval_agents_range=4-10 --repeats 5 --num-threads 2
```

## Notes

- **Checkpoint shortcuts**: Use numbers (1, 2, 3...). The script lists grouped runs (by `run.name_prefix` or run-name stem) and writes `checkpoint_groups.json` in the results folder so you can see which group each number maps to.
- Numeric resolution searches `paths.experiments_root`, `paths.train_experiments_root` (default `experiments/train/`), then `experiments/`. Within the selected run, strategy `best` (default) chooses the best success-rate checkpoint, preferring the oldest within tolerance; `latest`/`oldest` are also available.
- Full paths also work: `experiments/train/<run>/checkpoints/<run>_final`
- Checkpoint folder must contain `algorithm_state.pkl`
- If `--render-video` is set and `--repeats` is omitted, repeats drop to 1 to save time.
- Checkpoints saved under older Python (e.g., 3.10) are auto-restored on newer Python (3.11/3.12) by stripping connector state; the script handles this automatically.
- Video rendering is slow (~2-3 min per video with smooth motion)
- Use `--video-agents N` to render only specific agent count
- Results include git commit hash for reproducibility
- **Multi-map runs**: enable via `eval_maps.enabled: true` in the config. Each map gets its own `map_<label>/` folder plus a `cross_map_comparison/` directory with aggregated plots and CSVs. Use a low-cost smoke test (e.g., `--set eval_agents_range=4-4 --repeats 1`) before launching long sweeps. The default multi_map config auto-syncs env/model with the checkpoint (`eval_sync_train_config: true`) so CNN/baseline/D* runs load cleanly.

## Metrics Cheat Sheet (How to Read Plots/CSVs)

- **Episode success rate (%)**: Share of episodes where **all** agents reach goals before `max_steps` (equals 100% goal completion and no timeout).
- **Goal completion rate (%)**: Fraction of agents that reach goals within an episode (per run); highlights partial success when episode success is 0.
- **Timeout/deadlock rate (%)**: Episodes truncated at `max_steps` (interpreted as timeout/deadlock). Lower is better.
- **Throughput (goals/step)**: `goals_completed / episode_steps`; meaningful even for one-shot tasks. Higher is better but interpret alongside collisions/success.
- **Path efficiency (actual/optimal)**: `sum(actual steps) / sum(initial D* path length)` when D* lengths are available; otherwise a step-based proxy. Failed agents are excluded from the ratio (no optimal path observed).
- Path efficiency can be >1.0 (longer than optimal); values near 1 are better.
- **Collisions per 1000 episode steps**: Normalized counts (total, agent–agent, agent–obstacle). Lower is safer.
- **Wait fraction**: `wait_actions / total_actions`; higher often signals congestion or yielding.
- **Uncertainty (shading/bands)**: 95% CI over repeats (and over maps when aggregated).
- **Agent removal**: Agents are removed after reaching goals, so agent-steps and episode steps differ; all normalizations above use episode steps.

Reading the plots:
- `success_vs_agents.png`: Success vs agent count; high success + low deadlock is desired (see `timeout_or_deadlock_vs_agents.png`).
- `efficiency_vs_agents.png`: Path efficiency vs agent count; lower is better. Uses D* when available.
- `collisions_vs_agents.png`: Total, agent-agent, and agent-obstacle collisions per 1000 steps with shaded CI.
- `tradeoff_efficiency_vs_collisions.png`: Scatter of efficiency vs collisions; good models sit top-left (low collisions, efficient paths).
- Multi-map: `maps_success_heatmap.png` shows generalization across layouts.
