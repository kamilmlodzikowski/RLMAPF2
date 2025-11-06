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
    --checkpoint experiments/baseline-TIMESTAMP/checkpoints/baseline-TIMESTAMP_final
```

### With Video Rendering

```bash
python eval.py \
    --config baseline \
    --checkpoint experiments/baseline-TIMESTAMP/checkpoints/baseline-TIMESTAMP_final \
    --render-video \
    --video-agents 20
```

## Parameters

### Required
- `--config NAME` - Config file (in `configs/eval/`) or path to YAML
- `--checkpoint PATH` - Path to trained model checkpoint

### Optional
- `--render-video` - Enable video generation
- `--video-agents N` - Generate video only for N agents (default: all). Requires `--render-video` to be set.
- `--repeats N` - Number of evaluation repeats (default: from config)
- `--num-threads N` - Parallel evaluation threads (default: 1)
- `--set KEY=VALUE` - Override config values

### Override Examples

```bash
# Test different agent range
--set eval_agents_range=10-30

# Change max steps
--set environment.max_steps=500

# Multiple overrides
--set eval_agents_range=4-10 --repeats 5
```

## Output

Results saved in `experiments/eval-{config}-{timestamp}/`:
- `summary.csv` - Main results
- `final_results.csv` - Detailed data
- `evaluation_Nagents_repeatR.mp4` - Videos (if enabled)

## Common Commands

```bash
# Quick test (few agents, few repeats)
python eval.py --config baseline --checkpoint path \
    --set eval_agents_range=4-6 --repeats 2

# Full evaluation (statistics only, no video)
python eval.py --config baseline --checkpoint path \
    --set eval_agents_range=4-20 --repeats 10 --num-threads 4

# Generate single demo video
python eval.py --config baseline --checkpoint path \
    --render-video --video-agents 20

# Full evaluation + one video
python eval.py --config baseline --checkpoint path \
    --render-video --video-agents 20 \
    --set eval_agents_range=4-20 --repeats 10
```

## Notes

- Checkpoint path must point to the checkpoint folder (contains `algorithm_state.pkl`)
- Video rendering is slow (~2-3 min per video)
- Use `--video-agents N` to render only specific agent count
- Results include git commit hash for reproducibility