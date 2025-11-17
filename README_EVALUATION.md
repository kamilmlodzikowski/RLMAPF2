# RLMAPF2 Evaluation System

Comprehensive evaluation framework for multi-agent pathfinding with enhanced metrics, visualization, and multi-map support.

## Features Added

### 1. Enhanced Metrics Collection
- **Collision breakdown**: Agent-agent vs agent-obstacle collisions
- **Goal tracking**: Completion rate and average steps to goal
- **Wait actions**: Coordination behavior measurement
- **Path efficiency**: Optimal vs actual path comparison
- **Throughput**: Steps per second performance metric

### 2. Rich Visualizations
- **Summary plots**: All metrics vs agent count
- **Distribution plots**: Box plots showing variance across repeats
- **Collision breakdown**: Stacked bar charts comparing collision types
- **Success heatmaps**: Goal completion rate by agent count and repeat
- **Temporal progression**: Metrics with confidence intervals

### 3. Directory Separation
- **Training**: `experiments/train/` - All training runs
- **Evaluation**: `experiments/eval/` - All evaluation results
- Clean organization with backward compatibility

### 4. Multi-Map Evaluation
- Evaluate single checkpoint across multiple maps
- Automatic per-map result directories
- Cross-map comparison plots (2×3 metric grid)
- Aggregated cross-map summary CSV

### 5. Smooth Motion Rendering
- Interpolated agent movement between grid cells
- Configurable frame interpolation (default: 5 frames)
- Creates fluid, professional-looking videos
- Automatically adjusts video FPS for smooth playback

## Architecture

```
eval.py
├── Argument Parsing & Config Loading
├── Multi-Map Detection
├── Map Iteration Loop
│   ├── Algorithm Build & Checkpoint Restore
│   ├── Agent Range Iteration
│   │   ├── Parallel Evaluation (ThreadPoolExecutor)
│   │   │   └── run_repeat() - Single episode with metrics
│   │   ├── Intermediate Results (per agent count)
│   │   └── CSV/TXT Outputs
│   ├── Final Results Aggregation
│   ├── Summary CSV Generation
│   └── Plot Generation (5 types)
└── Cross-Map Comparison (if multi-map enabled)
    ├── plot_cross_map_comparison()
    └── create_cross_map_summary()
```

## Quick Start

### Simplest Usage (Auto-Discovery)
```bash
python eval.py --config baseline
```

**What happens**: Automatically finds the latest checkpoint for `baseline` in `experiments/train/`.

### Numbered Shortcuts (NEW - Super Fast!)
```bash
# Use latest checkpoint
python eval.py --config baseline --checkpoint 1

# Use second most recent checkpoint
python eval.py --config baseline --checkpoint 2

# Use third most recent checkpoint
python eval.py --config baseline --checkpoint 3
```

**Perfect for**: Quickly comparing different training checkpoints without typing paths!

### Other Checkpoint Selection Methods
```bash
# Explicit checkpoint path
python eval.py --config baseline --checkpoint experiments/train/baseline-xxx/checkpoints/checkpoint-500

# Specific training run (auto-finds best checkpoint within it)
python eval.py --config baseline --checkpoint-run baseline-20250115-143022
```

### Output Structure
`experiments/eval/baseline_eval-{timestamp}/`
- `summary.csv` - Averaged metrics per agent count
- `final_results.csv` - All individual runs
- `plots/` - 5 visualization types
- `evaluation_summary.png` - Quick overview plot

### Multi-Map Evaluation
```bash
# Auto-discover checkpoint
python eval.py --config multi_map

# Or with explicit checkpoint
python eval.py --config multi_map --checkpoint experiments/train/baseline-xxx/checkpoints/checkpoint-500
```

**Output**: `experiments/eval/multi_map_eval-{timestamp}/`
- `map_{label}/` - Per-map results directories
- `cross_map_plots/` - Comparison visualizations
- `cross_map_summary.csv` - Aggregated results

### With Video Rendering
```bash
# Basic video rendering
python eval.py --config baseline --render-video --video-agents 20

# With smooth motion for professional-looking videos
python eval.py --config baseline --render-video --video-agents 20 --smooth-motion

# Custom interpolation frames (higher = smoother but larger file)
python eval.py --config baseline --render-video --video-agents 20 --smooth-motion --motion-frames 10
```

Renders video only for 20-agent evaluation (faster than all). Checkpoint auto-discovered.

## Configuration

Create evaluation configs in `configs/eval/`:

**Single Map** (`baseline.yaml`):
```yaml
paths:
  experiments_root: experiments/eval

eval_agents_range: "4-20"
eval_repeats: 10
eval_num_threads: 4
eval_video_fps: 10

# Video rendering settings
smooth_motion: false  # Enable smooth motion interpolation
motion_frames: 5      # Interpolation frames per step (higher = smoother)

environment:
  maps_names_with_variants:
    big_empty_1-100a-20x20: null
```

**Multi-Map** (`multi_map.yaml`):
```yaml
paths:
  experiments_root: experiments/eval

eval_agents_range: "4-10"
eval_repeats: 10

eval_maps:
  enabled: true
  maps:
    - name: big_empty_1-100a-20x20
      label: "Empty 20x20"
    - name: pprai_1-20a-42x39
      label: "Warehouse"
    - name: corridors_10-10a-32x32
      label: "Corridors"
  aggregate_results: true
  generate_comparison_plots: true
```

## Output Structure

### Single Map
```
experiments/eval/<run-name>/
├── summary.csv                           # Aggregated metrics
├── final_results.csv                     # All runs (13 columns)
├── intermediate_results_*_agents.{csv,txt}
├── plots/
│   ├── evaluation_summary.png           # 2×3 metric grid
│   ├── metric_distributions.png         # Box plots
│   ├── collision_breakdown.png          # Stacked bars
│   ├── success_rate_heatmap.png         # 2D heatmap
│   └── temporal_progression.png         # Confidence intervals
├── eval_metadata.json
├── resolved_config.yaml
└── *.mp4                                 # Optional videos
```

### Multi-Map
```
experiments/eval/<run-name>/
├── map_empty_20x20/                      # Per-map results
│   ├── summary.csv
│   ├── final_results.csv
│   └── plots/
├── map_warehouse/
├── map_corridors/
├── cross_map_plots/
│   └── cross_map_comparison.png         # 2×3 grid (all maps)
├── cross_map_summary.csv                # Aggregated
└── eval_metadata.json
```

## Metrics Reference

### CSV Columns

**Detailed Results** (`final_results.csv`):
- `agents_num`, `repeat` - Test configuration
- `elapsed_time`, `episode_length` - Performance
- `reward` - Total episode reward
- `collisions`, `collision_agent_agent`, `collision_agent_obstacle` - Collision breakdown
- `throughput` - Steps per second
- `path_efficiency` - Optimal/actual path ratio
- `wait_actions` - Total wait actions (action 4)
- `goal_completion_rate` - % agents reaching goal
- `avg_steps_to_goal` - Average for successful agents

**Summary** (`summary.csv`):
- Averaged versions of above (`avg_*`)
- `deadlocks` - Count of deadlocked runs
- `success_rate` - % non-deadlocked runs

### Interpreting Results

**Goal Completion Rate**:
- 100% - All agents successful
- 50-99% - Partial success
- <50% - Major coordination issues

**Collision Breakdown**:
- High agent-agent → Poor coordination
- High agent-obstacle → Path planning issues
- Balanced → Expected in dense environments

**Wait Actions**:
- High → Good coordination (agents yielding)
- Low → Either very efficient or poor coordination

**Path Efficiency**:
- ~1.0 → Near-optimal paths
- <0.7 → Significant detours
- Decreasing with agents → Expected

## Command-Line Options

```bash
python eval.py [OPTIONS]

Required:
  --config NAME               Config file or path

Checkpoint Selection (pick one, or omit for auto-discovery):
  --checkpoint N              Number (1=latest, 2=previous, etc.) OR path to checkpoint
  --checkpoint-run NAME       Training run directory name
  (none)                      Auto-discovers latest checkpoint

Optional:
  --render-video              Enable video rendering
  --video-agents N            Render only for N agents
  --smooth-motion             Enable smooth motion with interpolation
  --motion-frames N           Interpolation frames per step (default: 5)
  --repeats N                 Override eval_repeats
  --num-threads N             Parallel evaluation threads
  --run-name NAME             Custom results directory name
  --set KEY=VALUE             Override config values
```

### Checkpoint Selection Guide

**Numbered shortcuts** (easiest):
- `--checkpoint 1` → Latest run for this config
- `--checkpoint 2` → Second latest run
- `--checkpoint 3` → Third latest run
- etc.

**Auto-discovery** (no argument):
- Omit `--checkpoint` entirely → Same as `--checkpoint 1`

**Explicit path** (full control):
- `--checkpoint experiments/train/baseline-xxx/checkpoints/checkpoint-500`

**What happens behind the scenes**:
1. Searches `experiments/train/{config_name}-*/`
2. Sorts by modification time (newest first)
3. Picks the Nth run based on your number
4. Finds highest checkpoint number in that run
5. Logs the discovered path

## Examples

**Super quick** (latest checkpoint):
```bash
python eval.py --config baseline --checkpoint 1
```

**Compare checkpoints** (test latest vs previous):
```bash
# Evaluate latest
python eval.py --config baseline --checkpoint 1 --run-name eval_latest

# Evaluate previous
python eval.py --config baseline --checkpoint 2 --run-name eval_previous
```

**Full evaluation with numbered checkpoint**:
```bash
python eval.py --config baseline --checkpoint 1 --set eval_agents_range="4-20" --repeats 10 --num-threads 4
```

**Multi-map with specific checkpoint**:
```bash
python eval.py --config multi_map --checkpoint 2
```

**Video rendering** (latest checkpoint):
```bash
# Basic video
python eval.py --config baseline --checkpoint 1 --render-video --video-agents 20

# Smooth motion video (professional quality)
python eval.py --config baseline --checkpoint 1 --render-video --video-agents 20 --smooth-motion

# Ultra-smooth motion (10 interpolation frames)
python eval.py --config baseline --checkpoint 1 --render-video --video-agents 20 --smooth-motion --motion-frames 10
```

**Still works - explicit path**:
```bash
python eval.py --config baseline --checkpoint experiments/train/baseline-xxx/checkpoints/checkpoint-500
```

**Still works - specific run**:
```bash
python eval.py --config baseline --checkpoint-run baseline-20250115-143022
```

## Smooth Motion Rendering

The smooth motion feature creates professional-quality videos by interpolating agent movement between grid cells, eliminating the "jumping" effect of traditional grid-based rendering.

### How It Works

- **Without smooth motion**: Agents teleport from cell to cell each step
- **With smooth motion**: Agents glide smoothly between positions with interpolated frames
- **Frame calculation**: Total frames = base_fps × motion_frames

### Configuration Methods

**1. Via Command Line** (overrides config file):
```bash
python eval.py --config baseline --render-video --smooth-motion --motion-frames 5
```

**2. Via Config File** (persistent setting):
```yaml
# In configs/eval/baseline.yaml
smooth_motion: true
motion_frames: 5
```

### Motion Frames Guide

- `motion_frames: 3` - Basic smoothing (fast rendering, smaller files)
- `motion_frames: 5` - Recommended default (good balance)
- `motion_frames: 10` - Very smooth (slower rendering, larger files)
- `motion_frames: 15` - Ultra-smooth (production quality)

### Performance Impact

**File size**: Increases proportionally with `motion_frames` (5× frames = ~5× larger)

**Rendering time**: Linear increase with frame count

**Effective FPS**: `video_fps × motion_frames`
- Base FPS: 10 (default)
- With 5 frames: 50 FPS effective
- With 10 frames: 100 FPS effective

### Example Comparison

```bash
# Standard video (10 FPS, choppy movement)
python eval.py --config baseline --render-video --video-agents 10

# Smooth video (50 FPS effective, fluid movement)
python eval.py --config baseline --render-video --video-agents 10 --smooth-motion

# Ultra-smooth video (100 FPS effective, cinematic quality)
python eval.py --config baseline --render-video --video-agents 10 --smooth-motion --motion-frames 10
```

### Use Cases

**Research presentations**: Use `--motion-frames 10` for clear agent trajectories

**Quick tests**: Omit `--smooth-motion` to save time and disk space

**Publications**: Use `--motion-frames 15` for publication-quality visualizations

**Debugging**: Standard rendering is faster for quick debugging

## Performance Tips

1. **Faster evaluation**: Use `--num-threads 4` for parallelization
2. **Better statistics**: Increase repeats to 20-50
3. **Skip videos**: Don't use `--render-video` for speed
4. **Quick tests**: Reduce agent range and repeats
5. **Video rendering**: Use `--smooth-motion` only when needed (increases file size and render time)
6. **Optimize smoothness**: Balance quality vs performance with `--motion-frames` (3=fast, 5=balanced, 10+=quality)

## Requirements

```bash
pip install ray[rllib] numpy pandas pyyaml matplotlib seaborn
```

## Backward Compatibility

- Existing configs continue to work
- Old CSV formats preserved (with new columns added)
- Previous plots still generated
- No breaking changes

## Troubleshooting

**"Could not generate plots"**
- Ensure at least 2 repeats per agent count
- Check CSV has all expected columns

**"goal_completion_rate not found"**
- Update to latest eval.py
- Old environment may not track completion

**Empty plots**
- Verify `final_results.csv` contains data
- Check for NaN values indicating collection issues
- Ensure agent range has multiple points

**Smooth motion not working**
- Ensure `--render-video` is enabled (smooth motion requires video rendering)
- Check both command-line flag (`--smooth-motion`) and config file setting
- Verify `render_mode: "human"` in environment config when rendering

**Video file too large**
- Reduce `--motion-frames` value (try 3 instead of 10)
- Use lower `video_dpi` in config (e.g., 150 instead of 300)
- Render only specific agent counts with `--video-agents N`

## Documentation

- This file: High-level overview and usage
- `TECHNICAL_IMPLEMENTATION.md`: Detailed code changes and implementation
- `eval.py --help`: Command-line reference
