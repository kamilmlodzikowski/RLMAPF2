# RLMAPF Evaluation

Evaluate trained PPO agents on multi-agent pathfinding tasks.

## Requirements

```bash
pip install ray[rllib] numpy pandas pyyaml matplotlib
```

## Basic Usage

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
- `--checkpoint N` - Checkpoint number (1 = most recent, 2 = second most recent, etc.) or full path

### Optional
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
```

## Output

Results are written under `paths.experiments_root` from the eval config (default `experiments/eval/`).
If `run.name_prefix` is set (e.g., `cnn`), the tool nests runs like `experiments/eval/cnn/{run-prefix}-{timestamp}/`.
Otherwise it falls back to `experiments/eval/{run-prefix}-{timestamp}/`.
- `summary.csv` - Main results
- `final_results.csv` - Detailed data
- `evaluation_{N}agents_repeat{R}.mp4` - Videos (if enabled, R is zero-based)

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

# Multi-map evaluation (iterates over multiple layouts)
python eval.py --config multi_map --checkpoint 1 \
    --set eval_agents_range=4-10 --repeats 5 --num-threads 2
```

## Notes

- **Checkpoint shortcuts**: Use numbers (1, 2, 3...) where 1 = most recent, 2 = second most recent, etc.
- Numeric checkpoints are resolved by searching `paths.experiments_root`, `paths.train_experiments_root`
  (default `experiments/train/`), and finally `experiments/`, so you can keep training runs under
  `experiments/train/<run>/checkpoints/...` without specifying full paths.
- Full paths also work: `experiments/train/<run>/checkpoints/<run>_final`
- Checkpoint folder must contain `algorithm_state.pkl`
- Video rendering is slow (~2-3 min per video with smooth motion)
- Use `--video-agents N` to render only specific agent count
- Results include git commit hash for reproducibility
- **Multi-map runs**: enable via `eval_maps.enabled: true` in the config. Each map gets its own `map_<label>/` folder plus a `cross_map_comparison/` directory with aggregated plots and CSVs. Use a low-cost smoke test (e.g., `--set eval_agents_range=4-4 --repeats 1`) before launching long sweeps.
