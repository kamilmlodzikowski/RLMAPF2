#!/usr/bin/env python3
"""Evaluate trained PPO agents for RLMAPF with structured experiment configs."""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Union

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ray
try:  # Ray versions prior to 2.7 don't expose this symbol
    from ray._private.utils import RayDeprecationWarning  # type: ignore
except ImportError:  # pragma: no cover - fallback for older Ray releases
    class RayDeprecationWarning(DeprecationWarning):
        pass
import yaml
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.ppo import PPOConfig

from rlmapf2 import RLMAPF

# Ignore deprecation warnings
os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning"
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RayDeprecationWarning)

ACTION_DELTAS = {
    0: (0, -1),  # Up
    1: (0, 1),   # Down
    2: (-1, 0),  # Left
    3: (1, 0),   # Right
    4: (0, 0),   # Wait
}

LOG_LEVEL = logging.INFO
logger = logging.getLogger("eval")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate trained PPO agents for RLMAPF.")
    parser.add_argument(
        "--config",
        required=True,
        help="Config name (relative to configs/eval) or explicit path to YAML file.",
    )
    parser.add_argument(
        "--config-dir",
        default=None,
        help="Base directory for config lookup when --config is a name (default: configs/eval).",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Checkpoint number (1=most recent, 2=second most recent, etc.) or full path to checkpoint directory.",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override config values, e.g. --set eval_agents_range=10-15.",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=None,
        help="Number of threads for parallel evaluation (default: from config or 1).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=None,
        help="Number of repeats for each agent count (default: from config or 10).",
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Custom directory to save results (default: auto-generated with timestamp).",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Custom run name for results folder.",
    )
    parser.add_argument(
        "--render-video",
        action="store_true",
        help="Enable video rendering for evaluation episodes.",
    )
    parser.add_argument(
        "--video-agents",
        type=str,
        default=None,
        help="Comma-separated list or ranges of agent counts to render (e.g. '8,16,32-36').",
    )
    return parser.parse_args(argv)


def resolve_config_path(name: str, config_dir: Optional[str], repo_root: Path) -> Path:
    """Resolve config file path from name or explicit path."""
    candidate = Path(name)
    if candidate.is_file():
        return candidate.resolve()

    search_dirs = []
    if config_dir:
        search_dirs.append(Path(config_dir))
    search_dirs.append(repo_root / "configs" / "eval")

    for base in search_dirs:
        with_suffix = name if name.endswith(".yaml") else f"{name}.yaml"
        path = (base / with_suffix).resolve()
        if path.exists():
            return path
    raise FileNotFoundError(f"Unable to locate config '{name}'. Looked in: {search_dirs}")


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def apply_overrides(config: Dict[str, Any], overrides: List[str]) -> Dict[str, Any]:
    """Apply command-line overrides to config."""
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid override format: {override}. Expected KEY=VALUE")
        
        key_path, value = override.split("=", 1)
        keys = key_path.split(".")
        
        # Navigate to the nested dict
        current = config
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # Set the value (try to parse as int/float/bool)
        final_key = keys[-1]
        try:
            if value.lower() in ("true", "false"):
                current[final_key] = value.lower() == "true"
            elif "." in value:
                current[final_key] = float(value)
            else:
                current[final_key] = int(value)
        except ValueError:
            current[final_key] = value
    
    return config


def parse_agent_range(range_str: str) -> range:
    """Parse agent range string like '10-20' to range object."""
    if isinstance(range_str, range):
        return range_str
    if isinstance(range_str, str):
        start, end = map(int, range_str.split('-'))
        return range(start, end + 1)
    raise ValueError(f"Invalid agent range format: {range_str}")


def parse_video_agent_selection(selection: Optional[str]) -> Optional[Set[int]]:
    """Parse comma-separated agent counts/ranges for video rendering."""
    if selection is None:
        return None

    selected: Set[int] = set()
    for part in selection.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            try:
                start_str, end_str = part.split('-', 1)
                start = int(start_str)
                end = int(end_str)
            except ValueError as exc:
                raise ValueError(f"Invalid video agent range: '{part}'") from exc
            if start > end:
                raise ValueError(f"Video agent range start > end in '{part}'")
            selected.update(range(start, end + 1))
        else:
            try:
                selected.add(int(part))
            except ValueError as exc:
                raise ValueError(f"Invalid agent count '{part}' in --video-agents") from exc

    if not selected:
        raise ValueError("--video-agents did not include any valid agent counts")
    return selected


def resolve_checkpoint_path(checkpoint_arg: str,
                            experiments_roots: Union[str, Path, Iterable[Union[str, Path]]] = "experiments") -> str:
    """
    Resolve checkpoint argument to full path.

    If checkpoint_arg is a digit (1, 2, 3...), finds the Nth most recent checkpoint
    where 1 = most recent, 2 = second most recent, etc.
    Otherwise, returns the path as-is.

    Args:
        checkpoint_arg: Either a number or a full checkpoint path
        experiments_roots: One or more root directories to search for checkpoints

    Returns:
        Full path to checkpoint directory
    """
    # If it's a digit, find the Nth most recent checkpoint
    if checkpoint_arg.isdigit():
        n = int(checkpoint_arg)

        if isinstance(experiments_roots, (str, Path)):
            roots: List[Union[str, Path]] = [experiments_roots]
        else:
            roots = list(experiments_roots)
        if not roots:
            roots = ["experiments"]

        checkpoint_dirs: List[Path] = []
        for experiments_root in roots:
            experiments_path = Path(experiments_root)
            if not experiments_path.exists():
                continue
            for ckpt_state in experiments_path.rglob("algorithm_state.pkl"):
                checkpoint_dirs.append(ckpt_state.parent)

        if not checkpoint_dirs:
            raise FileNotFoundError(
                "No checkpoints found in any search roots: " + ", ".join(str(r) for r in roots)
            )

        # Remove duplicates and sort by modification time (most recent first)
        unique_dirs = []
        seen = set()
        for path in checkpoint_dirs:
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                unique_dirs.append(path)
        unique_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)

        if n > len(unique_dirs):
            raise ValueError(f"Requested checkpoint #{n} but only {len(unique_dirs)} checkpoints found")

        selected_checkpoint = unique_dirs[n - 1]
        logger.info(f"Checkpoint #{n} resolved to: {selected_checkpoint}")
        return str(selected_checkpoint)

    # Otherwise, return the path as-is
    return checkpoint_arg


def collect_git_info(repo_root: Path) -> Dict[str, Any]:
    """Collect git repository information."""
    git_dir = repo_root / ".git"
    if not git_dir.exists():
        return {}
    try:
        commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root)
            .decode()
            .strip()
        )
        status = (
            subprocess.check_output(["git", "status", "--short"], cwd=repo_root)
            .decode()
            .strip()
        )
        branch = (
            subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_root)
            .decode()
            .strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return {}
    return {"commit": commit, "branch": branch, "status": status}


def build_run_name(config: Dict[str, Any], explicit_name: Optional[str]) -> str:
    """Build run name for the evaluation."""
    if explicit_name:
        return explicit_name
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    name_prefix = config.get("run", {}).get("name_prefix", "eval")
    return f"{name_prefix}-{timestamp}"


def configure_env(config: Dict[str, Any], agents_num: Optional[int] = None, 
                 render_mode: Optional[str] = None, seed: int = 42,
                 render_video: bool = False, video_path: Optional[str] = None) -> Dict[str, Any]:
    """Configure the environment with parameters from config."""
    env_config = config['environment']
    paths_config = config['paths']
    
    # Use provided values or defaults from config
    agents_num = agents_num if agents_num is not None else env_config['agents_num']
    
    # Override render_mode when rendering videos
    # Video rendering requires "human" mode, while config specifies "rgb_array" 
    # for non-video evaluation runs
    if render_video:
        render_mode = "human"
    else:
        render_mode = render_mode if render_mode is not None else env_config.get('render_mode', 'none')
    
    # Get map root path
    map_root = paths_config.get('map_root', 'maps')
    # Convert to absolute path if not already
    if not os.path.isabs(map_root):
        map_root = os.path.join(os.getcwd(), map_root)
    
    # Setup video path
    if video_path is None:
        video_path = f"evaluation_video_{agents_num}agents.mp4"
    
    return {
        "agents_num": agents_num,
        "render_mode": render_mode,
        "render_delay": env_config.get('render_delay', 0.01),
        "seed": seed,
        "map_path": map_root,
        "maps_names_with_variants": env_config.get('maps_names_with_variants', {}),
        "max_steps": env_config.get('max_steps', 250),
        "collision_penalty": env_config.get('collision_penalty', 0.1),
        "step_cost": env_config.get('step_cost', 0.02),
        "wait_cost_multiplier": env_config.get('wait_cost_multiplier', 2),
        "goal_reward": env_config.get('goal_reward', 10),
        "observation_type": env_config.get('observation_type', 'array'),
        "penalize_collision": env_config.get('penalize_collision', True),
        "penalize_waiting": env_config.get('penalize_waiting', True),
        "penalize_steps": env_config.get('penalize_steps', True),
        "reward_closer_to_goal_each_step": env_config.get('reward_closer_to_goal_each_step', False),
        "reward_closer_to_goal_final": env_config.get('reward_closer_to_goal_final', True),
        "reward_final_d_star": env_config.get('reward_final_d_star', False),
        "reward_low_density": env_config.get('reward_low_density', False),
        "use_d_star_lite": env_config.get('use_d_star_lite', False),
        "use_cnn_observation": env_config.get('use_cnn_observation', False),
        "penalize_left_side_bottom_passing": env_config.get('penalize_left_side_bottom_passing', False),
        "start_goal_on_periphery": env_config.get('start_goal_on_periphery', False),
        "render_config": {
            "show_render": False,
            "save_video": render_video,
            "include_legend": True,
            "legend_position": (0, 0),
            "video_path": video_path,
            "video_fps": config.get('eval_video_fps', 10),
            "video_dpi": 300,
            "render_delay": 0.2,
            "title": f"RLMAPF Evaluation - {agents_num} agents" + (" (D*)" if env_config.get('use_d_star_lite', False) else ""),
            "save_frames": False,
            "frames_path": "frames/",
            "smooth_motion": config.get('eval_smooth_motion', False),
            "motion_frames": config.get('eval_motion_frames', 5),
        },
    }


def save_metadata(
    experiment_dir: Path,
    config: Dict[str, Any],
    config_path: Path,
    checkpoint_path: str,
    run_name: str,
    git_info: Dict[str, Any],
) -> None:
    """Save evaluation metadata."""
    metadata = {
        "run_name": run_name,
        "config_path": str(config_path),
        "checkpoint_path": checkpoint_path,
        "start_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git": git_info,
    }
    
    metadata_file = experiment_dir / "eval_metadata.json"
    metadata_file.write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    
    # Save resolved config
    config_file = experiment_dir / "resolved_config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def plot_success_and_deadlocks(detailed_results_path: Path, plots_dir: Path) -> None:
    """Line plots for success and deadlock rates vs agent count."""
    try:
        df = pd.read_csv(detailed_results_path)
        if 'success' not in df.columns:
            logger.warning("Success column not found; skipping success plot")
            return
        summary = df.groupby('agents_num').agg(
            success_rate_percent=('success', lambda s: s.mean() * 100.0),
            deadlock_rate_percent=('deadlock', lambda s: s.mean() * 100.0 if 'deadlock' in df.columns else np.nan),
        ).reset_index()
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(summary['agents_num'], summary['success_rate_percent'], marker='o', label='Success rate (%)')
        if 'deadlock_rate_percent' in summary:
            ax.plot(summary['agents_num'], summary['deadlock_rate_percent'], marker='s', label='Deadlock rate (%)')
        ax.set_xlabel('Number of agents')
        ax.set_ylabel('Rate (%)')
        ax.set_title('Success and deadlock rates')
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / 'success_deadlocks.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Success/deadlock plot saved")
    except Exception as exc:
        logger.warning("Could not generate success/deadlock plot: %s", exc)


def plot_efficiency(detailed_results_path: Path, plots_dir: Path) -> None:
    """Efficiency plots: episode length (successful), throughput, path efficiency."""
    try:
        df = pd.read_csv(detailed_results_path)
        metrics = [
            ('success_episode_length_steps', 'Episode length (successful only)'),
            ('throughput_steps_per_sec', 'Throughput (steps/s)'),
            ('path_efficiency', 'Path efficiency (actual/optimal)'),
        ]
        available = [(m, title) for m, title in metrics if m in df.columns and not df[m].isna().all()]
        if not available:
            logger.warning("No efficiency metrics available for plotting")
            return
        fig, axes = plt.subplots(1, len(available), figsize=(6 * len(available), 5))
        if len(available) == 1:
            axes = [axes]
        for ax, (metric, title) in zip(axes, available):
            summary = df.groupby('agents_num')[metric].agg(['mean', 'std']).reset_index()
            ax.plot(summary['agents_num'], summary['mean'], marker='o', label='mean')
            ax.fill_between(summary['agents_num'], summary['mean'] - summary['std'], summary['mean'] + summary['std'],
                            alpha=0.25, label='±1 std')
            ax.set_xlabel('Number of agents')
            ax.set_ylabel(title)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            ax.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / 'efficiency.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Efficiency plots saved")
    except Exception as exc:
        logger.warning("Could not generate efficiency plot: %s", exc)


def plot_collisions(detailed_results_path: Path, plots_dir: Path) -> None:
    """Stacked bars for collisions per agent-step split by type."""
    try:
        df = pd.read_csv(detailed_results_path)
        required = {'collision_agent_agent_per_agent_step', 'collision_agent_obstacle_per_agent_step'}
        if not required.issubset(df.columns):
            logger.warning("Collision rate columns not found; skipping collision plot")
            return
        summary = df.groupby('agents_num').agg({
            'collision_agent_agent_per_agent_step': 'mean',
            'collision_agent_obstacle_per_agent_step': 'mean',
        }).reset_index()
        fig, ax = plt.subplots(figsize=(8, 5))
        x = summary['agents_num']
        ao = summary['collision_agent_obstacle_per_agent_step']
        aa = summary['collision_agent_agent_per_agent_step']
        ax.bar(x, ao, label='Agent–Obstacle', color='steelblue', alpha=0.85)
        ax.bar(x, aa, bottom=ao, label='Agent–Agent', color='coral', alpha=0.85)
        ax.set_xlabel('Number of agents')
        ax.set_ylabel('Collisions per agent-step')
        ax.set_title('Safety: collisions by type')
        ax.grid(True, axis='y', alpha=0.3)
        ax.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / 'collisions.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Collision plot saved")
    except Exception as exc:
        logger.warning("Could not generate collision plot: %s", exc)


def plot_cross_map_comparison(map_summaries: Dict[str, Path], plots_dir: Path):
    """Generate 2×3 grid comparing metrics across maps."""
    try:
        dfs = {}
        for label, summary_path in map_summaries.items():
            if summary_path.exists():
                dfs[label] = pd.read_csv(summary_path)
        if len(dfs) < 2:
            logger.warning("Need at least 2 maps for cross-map comparison")
            return
        metrics = [
            ('avg_success_rate_percent', 'Success rate (%)'),
            ('avg_deadlock_rate_percent', 'Deadlock rate (%)'),
            ('avg_success_length_steps', 'Episode length (success)'),
            ('avg_throughput_steps_per_sec', 'Throughput (steps/s)'),
            ('avg_collision_agent_agent_per_agent_step', 'Agent–Agent collisions/agent-step'),
            ('avg_collision_agent_obstacle_per_agent_step', 'Agent–Obstacle collisions/agent-step'),
        ]
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        for idx, (metric, title) in enumerate(metrics):
            for map_label, df in dfs.items():
                if metric in df.columns:
                    axes[idx].plot(df['agents_num'], df[metric], marker='o', linewidth=2, label=map_label)
            axes[idx].set_xlabel('Number of agents')
            axes[idx].set_ylabel(title)
            axes[idx].set_title(title)
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / 'cross_map_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Cross-map comparison plot saved")
    except Exception as e:
        logger.warning(f"Could not generate cross-map comparison plot: {e}")


def create_cross_map_summary(map_summaries: Dict[str, Path], output_path: Path):
    """Aggregate all map results into single CSV."""
    try:
        all_data = []
        for map_label, summary_path in map_summaries.items():
            if summary_path.exists():
                df = pd.read_csv(summary_path)
                df['map'] = map_label
                all_data.append(df)
        if not all_data:
            logger.warning("No map summaries found for aggregation")
            return
        combined_df = pd.concat(all_data, ignore_index=True)
        cols = ['map'] + [col for col in combined_df.columns if col != 'map']
        combined_df = combined_df[cols]
        combined_df.to_csv(output_path, index=False)
        logger.info(f"Cross-map summary saved to {output_path}")
    except Exception as e:
        logger.warning(f"Could not create cross-map summary: {e}")


def main(argv: Optional[List[str]] = None) -> int:
    """Main evaluation function."""
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
    )
    
    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parent
    try:
        video_agent_selection = parse_video_agent_selection(args.video_agents)
    except ValueError as exc:
        logger.error(str(exc))
        return 1
    
    # Load and resolve config
    config_path = resolve_config_path(args.config, args.config_dir, repo_root)
    logger.info("Loading config from %s", config_path)
    config = load_config(config_path)
    
    # Apply overrides
    if args.overrides:
        logger.info("Applying %d override(s)", len(args.overrides))
        config = apply_overrides(config, args.overrides)
    
    # Extract configuration values
    hardware_config = config['hardware']
    model_config = config['model']
    env_config = config['environment']
    
    NUM_CPUS = hardware_config.get('num_cpus', 2)
    NUM_GPUS = hardware_config.get('num_gpus', 0)

    # Resolve and convert checkpoint path to absolute path
    paths_config = config.get('paths', {})
    experiments_root = paths_config.get('experiments_root')
    train_experiments_root = paths_config.get('train_experiments_root', 'experiments/train')

    checkpoint_search_roots: List[Union[str, Path]] = []
    if experiments_root:
        checkpoint_search_roots.append(experiments_root)
    if train_experiments_root and all(Path(train_experiments_root) != Path(existing)
                                     for existing in checkpoint_search_roots):
        checkpoint_search_roots.append(train_experiments_root)
    # Always include generic fallback to maintain backwards compatibility
    if all(Path('experiments') != Path(existing) for existing in checkpoint_search_roots):
        checkpoint_search_roots.append('experiments')

    resolved_checkpoint = resolve_checkpoint_path(args.checkpoint, checkpoint_search_roots)
    CHECKPOINT_PATH = os.path.abspath(resolved_checkpoint)
    
    # Parse evaluation parameters
    agent_range_str = config.get('eval_agents_range', '4-20')
    AGENTS_RANGE = parse_agent_range(agent_range_str)
    
    NUM_THREADS = args.num_threads if args.num_threads is not None else config.get('eval_num_threads', 1)
    REPEATS = args.repeats if args.repeats is not None else config.get('eval_repeats', 10)
    
    # Video rendering settings
    RENDER_VIDEO = args.render_video
    VIDEO_AGENTS = video_agent_selection  # If set, only render for specific agent counts
    
    # If rendering video, only do 1 repeat (to save time) unless specified otherwise
    if RENDER_VIDEO and args.repeats is None:
        REPEATS = 1
        logger.info("Video rendering enabled - setting repeats to 1 (override with --repeats if needed)")
    
    # Build run name and create directories
    run_name = build_run_name(config, args.run_name)
    
    if args.results_dir:
        results_dir = Path(args.results_dir).resolve()
    else:
        results_base = Path(config.get('paths', {}).get('experiments_root', 'experiments'))
        name_prefix = config.get('run', {}).get('name_prefix')
        if name_prefix:
            results_dir = (results_base / name_prefix / run_name).resolve()
        else:
            results_dir = (results_base / run_name).resolve()
    
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect git info
    git_info = collect_git_info(repo_root) if config.get('logging', {}).get('log_git_info', True) else {}
    
    # Save metadata
    save_metadata(results_dir, config, config_path, CHECKPOINT_PATH, run_name, git_info)
    
    logger.info("=" * 80)
    logger.info("Evaluation Configuration")
    logger.info("=" * 80)
    logger.info("Run name: %s", run_name)
    logger.info("Results directory: %s", results_dir)
    logger.info("Checkpoint: %s", CHECKPOINT_PATH)
    logger.info("Agent range: %s", list(AGENTS_RANGE))
    logger.info("Threads: %d, Repeats: %d", NUM_THREADS, REPEATS)
    logger.info("CPUs: %d, GPUs: %d", NUM_CPUS, NUM_GPUS)
    if RENDER_VIDEO:
        if VIDEO_AGENTS:
            selection_str = ", ".join(str(num) for num in sorted(VIDEO_AGENTS))
            logger.info("Video rendering: ENABLED (only for agents: %s)", selection_str)
        else:
            logger.info("Video rendering: ENABLED (for all agent counts)")
    else:
        logger.info("Video rendering: DISABLED")
    logger.info("=" * 80)
    
    return_code = 0
    
    try:
        # Initialize Ray
        ray_context = ray.init(
            num_cpus=NUM_CPUS,
            num_gpus=NUM_GPUS,
        )
        logger.info("Ray initialized (dashboard: %s)", ray_context.dashboard_url)
        
        # Create the algorithm configuration
        algorithm_config_builder = (
            PPOConfig()
            .api_stack(
                enable_rl_module_and_learner=model_config['api_stack'].get('enable_rl_module_and_learner', False),
                enable_env_runner_and_connector_v2=model_config['api_stack'].get('enable_env_runner_and_connector_v2', False)
            )
            .framework(model_config.get('framework', 'torch'))
            .training(model=model_config.get('model', {}))
            .environment(RLMAPF, env_config=configure_env(config))
            .evaluation(
                evaluation_interval=1,
                evaluation_duration=1,
                evaluation_config={"env_config": configure_env(config, render_mode="none")}
            )
            .resources(num_gpus=NUM_GPUS)
        )
        
        # Check for multi-map evaluation
        eval_maps_config = config.get('eval_maps', {})
        multi_map_enabled = eval_maps_config.get('enabled', False)
        
        if multi_map_enabled:
            maps_to_evaluate = eval_maps_config.get('maps', [])
            logger.info("=" * 80)
            logger.info("MULTI-MAP EVALUATION ENABLED: %d maps", len(maps_to_evaluate))
            logger.info("=" * 80)
        else:
            current_map = list(env_config.get('maps_names_with_variants', {}).keys())[0] if env_config.get('maps_names_with_variants') else 'default'
            maps_to_evaluate = [{
                'name': current_map,
                'variants': env_config.get('maps_names_with_variants', {}).get(current_map),
                'label': current_map
            }]
        
        map_summary_paths = {}
        
        for map_idx, map_config_item in enumerate(maps_to_evaluate):
            map_name = map_config_item['name']
            map_label = map_config_item.get('label', map_name)
            map_variants = map_config_item.get('variants')
        
            if multi_map_enabled:
                logger.info("")
                logger.info("=" * 80)
                logger.info(f"EVALUATING MAP {map_idx + 1}/{len(maps_to_evaluate)}: {map_label}")
                logger.info(f"Map: {map_name}")
                logger.info("=" * 80)
        
            if multi_map_enabled:
                current_results_dir = results_dir / f"map_{map_label.replace(' ', '_').lower()}"
                current_results_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"Map results directory: {current_results_dir}")
            else:
                current_results_dir = results_dir
        
            # Setup results tracking
            time_results = {}
            lengths_results = {}
            rewards_results = {}
            collisions_results = {}
            collision_agent_agent_results = {}
            collision_agent_obstacle_results = {}
            throughput_results = {}
            wait_actions_results = {}
            goal_completion_rate_results = {}
            avg_steps_to_goal_results = {}
            completion_deviation_results = {}
            success_flags = {}
            success_length_results = {}
            path_efficiency_results = {}
            collision_agent_agent_rate_results = {}
            collision_agent_obstacle_rate_results = {}
            collision_total_rate_results = {}
            deadlock_flags = {}
            deadlocks = {}
        
            def run_repeat(agents_num: int, repeat: int) -> tuple:
                """Run a single evaluation repeat."""
                logger.info("Evaluating with %d agents, repeat %d/%d", agents_num, repeat + 1, REPEATS)
            
                # Check if we should render video for this agent count
                should_render = RENDER_VIDEO and (VIDEO_AGENTS is None or agents_num in VIDEO_AGENTS)
            
                # Setup video path if rendering
                video_path = None
                if should_render:
                    video_filename = f"evaluation_{agents_num}agents_repeat{repeat}.mp4"
                    video_path = str(results_dir / video_filename)
                    logger.info("Rendering video to: %s", video_path)
            
                # For video rendering, manually step through environment
                if should_render:
                    # Create environment directly with video rendering enabled
                    env_config_dict = configure_env(
                        config,
                        agents_num=agents_num,
                        render_mode=None,  # Will be set by configure_env
                        seed=42 + repeat,
                        render_video=True,
                        video_path=video_path
                    )
                
                    # RLMAPF is already imported at the top of the file
                    env = RLMAPF(env_config_dict)
                
                    # Load policy from checkpoint
                    temp_algorithm = algorithm_config_builder.build()
                    temp_algorithm.restore(CHECKPOINT_PATH)
                    policy = temp_algorithm.get_policy()

                    try:
                        # Run episode manually
                        start_time = time.time()
                        obs, info = env.reset()
                        done = False
                        step_count = 0
                        max_steps = env_config_dict['max_steps']

                        while not done and step_count < max_steps:
                            # Get actions from policy
                            actions = {}
                            for agent_id, agent_obs in obs.items():
                                action = policy.compute_single_action(agent_obs)[0]
                                actions[agent_id] = action
                        
                            # Step environment (render() called automatically inside step())
                            obs, rewards, dones, truncated, info = env.step(actions)
                            step_count += 1
                            done = all(dones.values()) or all(truncated.values())

                        # CRITICAL: Finalize video to save it
                        env.finalize_video()
                        env.close()

                        elapsed_time = time.time() - start_time
                        episode_len = step_count

                        logger.info("Video saved to: %s", video_path)
                    finally:
                        # Cleanup temporary algorithm
                        temp_algorithm.stop()
                        del temp_algorithm
                else:
                    # Normal evaluation without video using RLlib's evaluate()
                    eval_config = algorithm_config_builder.environment(
                        RLMAPF,
                        env_config=configure_env(
                            config,
                            agents_num=agents_num,
                            render_mode="none",
                            seed=42 + repeat,
                            render_video=False
                        )
                    )
                
                    eval_algorithm = eval_config.build()
                    eval_algorithm.restore(CHECKPOINT_PATH)
                
                    start_time = time.time()
                    try:
                        results = eval_algorithm.evaluate()
                        elapsed_time = time.time() - start_time
                        episode_len = results['env_runners']['episode_len_mean']
                    except Exception as e:
                        logger.error("Error during evaluation: %s", e)
                        return (
                            agents_num,
                            repeat,
                            None,
                            None,
                            None,
                            None,
                            None,
                            None,
                            None,
                            None,
                            None,
                            None,
                            None,
                            False,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            1,
                        )
                    finally:
                        eval_algorithm.stop()
            
                # Check for deadlock
                max_steps = env_config.get('max_steps', 250)
                deadlock = 0 if episode_len < max_steps else 1
            
                logger.debug("Completed in %.5fs, steps: %.2f", elapsed_time, episode_len)
            

                # Always run episodes manually to collect detailed metrics
                env_config_dict = configure_env(
                    config,
                    agents_num=agents_num,
                    render_mode=None,
                    seed=42 + repeat,
                    render_video=should_render,
                    video_path=video_path
                )

                env = RLMAPF(env_config_dict)
                temp_algorithm = algorithm_config_builder.build()
                temp_algorithm.restore(CHECKPOINT_PATH)
                policy = temp_algorithm.get_policy()

                try:
                    start_time = time.time()
                    obs, info = env.reset()
                    done = False
                    step_count = 0
                    max_steps = env_config_dict['max_steps']

                    actions_history = []
                    agents_reached_goal = set()
                    agent_completion_steps = {}
                    total_reward = 0.0
                    agent_agent_collision_total = 0
                    agent_obstacle_collision_total = 0
                    prev_positions = {}
                    total_agents = len(env.agents) if hasattr(env, 'agents') else agents_num
                    collision_counters = {
                        agent_id: info.get(agent_id, {}).get("number_of_collisions", 0)
                        for agent_id in obs.keys()
                    }

                    while not done and step_count < max_steps:
                        if hasattr(env, 'agent_positions'):
                            prev_positions = {
                                agent_id: env.agent_positions[agent_id].copy()
                                if hasattr(env.agent_positions[agent_id], 'copy')
                                else tuple(env.agent_positions[agent_id])
                                for agent_id in obs.keys()
                            }

                        actions = {}
                        intended_positions = {}
                        for agent_id, agent_obs in obs.items():
                            action = policy.compute_single_action(agent_obs)[0]
                            actions[agent_id] = action
                            delta = ACTION_DELTAS.get(action, (0, 0))
                            prev = prev_positions.get(agent_id, (0, 0))
                            intended_positions[agent_id] = (prev[0] + delta[0], prev[1] + delta[1])

                        actions_history.append(actions.copy())
                        obs, rewards, dones, truncated, info = env.step(actions)
                        total_reward += sum(rewards.values())

                        for agent_id, is_done in dones.items():
                            if is_done and agent_id not in agents_reached_goal:
                                agents_reached_goal.add(agent_id)
                                agent_completion_steps[agent_id] = step_count

                        collision_deltas = {}
                        for agent_id in collision_counters.keys():
                            new_count = info.get(agent_id, {}).get("number_of_collisions", collision_counters[agent_id])
                            delta = new_count - collision_counters[agent_id]
                            if delta > 0:
                                collision_deltas[agent_id] = delta
                            collision_counters[agent_id] = new_count

                        if collision_deltas:
                            for agent_id, delta in collision_deltas.items():
                                is_agent_collision = False
                                for other_id in collision_deltas.keys():
                                    if other_id == agent_id:
                                        continue
                                    if intended_positions.get(agent_id) == intended_positions.get(other_id):
                                        is_agent_collision = True
                                        break
                                    if (
                                        intended_positions.get(agent_id) == prev_positions.get(other_id)
                                        and intended_positions.get(other_id) == prev_positions.get(agent_id)
                                    ):
                                        is_agent_collision = True
                                        break
                                    if (
                                        intended_positions.get(agent_id) == prev_positions.get(other_id)
                                        and actions.get(other_id, 4) == 4
                                    ):
                                        is_agent_collision = True
                                        break
                                if is_agent_collision:
                                    agent_agent_collision_total += delta
                                else:
                                    agent_obstacle_collision_total += delta

                        step_count += 1
                        done = all(dones.values()) or all(truncated.values())

                    if should_render:
                        env.finalize_video()
                        logger.info("Video saved to: %s", video_path)

                    env.close()
                    elapsed_time = time.time() - start_time
                finally:
                    temp_algorithm.stop()
                    del temp_algorithm

                episode_len = step_count
                wait_actions = sum(1 for action_set in actions_history for action in action_set.values() if action == 4)
                goal_completion_rate = (len(agents_reached_goal) / total_agents * 100) if total_agents > 0 else 0
                success = (
                    episode_len < max_steps
                    and total_agents > 0
                    and goal_completion_rate >= 99.9
                )
                avg_steps_to_goal = (
                    sum(agent_completion_steps.values()) / len(agent_completion_steps)
                    if agent_completion_steps
                    else np.nan
                )
                total_collisions = agent_agent_collision_total + agent_obstacle_collision_total
                collision_agent_agent = agent_agent_collision_total
                collision_agent_obstacle = agent_obstacle_collision_total
                throughput = episode_len / elapsed_time if elapsed_time > 0 else 0
                if agent_completion_steps:
                    mean_completion = float(np.mean(list(agent_completion_steps.values())))
                    completion_deviation = float(np.mean([abs(v - mean_completion) for v in agent_completion_steps.values()]))
                else:
                    completion_deviation = np.nan

                denom = max(episode_len * total_agents, 1)
                collision_agent_agent_rate = collision_agent_agent / denom
                collision_agent_obstacle_rate = collision_agent_obstacle / denom
                collision_total_rate = total_collisions / denom

                path_efficiency = np.nan
                if getattr(env, "start_d_star_paths", None) and agent_completion_steps:
                    efficiencies = []
                    for agent_id, steps_to_goal in agent_completion_steps.items():
                        start_path = env.start_d_star_paths.get(agent_id)
                        if start_path:
                            optimal_len = max(len(start_path), 1)
                            efficiencies.append(steps_to_goal / optimal_len)
                    if efficiencies:
                        path_efficiency = float(np.mean(efficiencies))

                success_episode_length = episode_len if success else np.nan
                deadlock = 0 if episode_len < max_steps else 1

                logger.info(
                    "Completed in %.4f s | steps=%d | reward=%.2f | collisions=%d "
                    "(agent-agent=%d, agent-obstacle=%d) | wait actions=%d | "
                    "goal_rate=%.1f%% | throughput=%.2f steps/s | success=%s",
                    elapsed_time,
                    episode_len,
                    total_reward,
                    total_collisions,
                    collision_agent_agent,
                    collision_agent_obstacle,
                    wait_actions,
                    goal_completion_rate,
                    throughput,
                    success,
                )

                return (
                    agents_num,
                    repeat,
                    elapsed_time,
                    episode_len,
                    total_reward,
                    total_collisions,
                    collision_agent_agent,
                    collision_agent_obstacle,
                    throughput,
                    wait_actions,
                    goal_completion_rate,
                    avg_steps_to_goal,
                    completion_deviation,
                    success,
                    path_efficiency,
                    collision_agent_agent_rate,
                    collision_agent_obstacle_rate,
                    collision_total_rate,
                    success_episode_length,
                    deadlock,
                )
        
            # Run evaluations for each agent count
            for agents_num in AGENTS_RANGE:
                logger.info("")
                logger.info("=" * 80)
                logger.info("Evaluating with %d agents", agents_num)
                logger.info("=" * 80)
            
                deadlocks[agents_num] = 0
            
                with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
                    futures = [executor.submit(run_repeat, agents_num, repeat) for repeat in range(REPEATS)]
                
                    for future in as_completed(futures):
                        (
                            agents_num_r,
                            repeat_r,
                            elapsed_time,
                            episode_len,
                            reward,
                            collisions,
                            collision_aa,
                            collision_ao,
                            throughput,
                            wait_acts,
                            goal_rate,
                            avg_steps,
                            completion_dev,
                            success,
                            path_eff,
                            collision_aa_rate,
                            collision_ao_rate,
                            collision_total_rate,
                            success_episode_len,
                            deadlock_flag,
                        ) = future.result()

                        # Save all results
                        time_results[(agents_num_r, repeat_r)] = elapsed_time
                        lengths_results[(agents_num_r, repeat_r)] = episode_len
                        rewards_results[(agents_num_r, repeat_r)] = reward
                        collisions_results[(agents_num_r, repeat_r)] = collisions
                        collision_agent_agent_results[(agents_num_r, repeat_r)] = collision_aa
                        collision_agent_obstacle_results[(agents_num_r, repeat_r)] = collision_ao
                        throughput_results[(agents_num_r, repeat_r)] = throughput
                        wait_actions_results[(agents_num_r, repeat_r)] = wait_acts
                        goal_completion_rate_results[(agents_num_r, repeat_r)] = goal_rate
                        avg_steps_to_goal_results[(agents_num_r, repeat_r)] = avg_steps
                        completion_deviation_results[(agents_num_r, repeat_r)] = completion_dev
                        success_flags[(agents_num_r, repeat_r)] = success
                        success_length_results[(agents_num_r, repeat_r)] = success_episode_len
                        path_efficiency_results[(agents_num_r, repeat_r)] = path_eff
                        collision_agent_agent_rate_results[(agents_num_r, repeat_r)] = collision_aa_rate
                        collision_agent_obstacle_rate_results[(agents_num_r, repeat_r)] = collision_ao_rate
                        collision_total_rate_results[(agents_num_r, repeat_r)] = collision_total_rate
                        deadlock_flags[(agents_num_r, repeat_r)] = deadlock_flag
                        deadlocks[agents_num_r] += deadlock_flag
            
                # Calculate and save intermediate results
                agent_time_results = [(k, v) for k, v in time_results.items() if k[0] == agents_num]
                agent_length_results = [(k, v) for k, v in lengths_results.items() if k[0] == agents_num]
                agent_success_lengths = [(k, v) for k, v in success_length_results.items() if k[0] == agents_num and not np.isnan(v)]
                agent_success_flags = [(k, v) for k, v in success_flags.items() if k[0] == agents_num]
            
                if agent_time_results:
                    avg_time = sum(v for _, v in agent_time_results) / len(agent_time_results)
                    avg_length = sum(v for _, v in agent_length_results) / len(agent_length_results)
                    avg_success_length = (
                        sum(v for _, v in agent_success_lengths) / len(agent_success_lengths)
                        if agent_success_lengths else np.nan
                    )
                    success_rate = (
                        sum(1 for _, flag in agent_success_flags if flag) / len(agent_success_flags) * 100
                        if agent_success_flags else 0
                    )
                
                    logger.info("")
                    logger.info("Results for %d agents:", agents_num)
                    logger.info("  Average episode time: %.5f s", avg_time)
                    logger.info("  Average episode length: %.2f steps", avg_length)
                    logger.info("  Success rate: %.1f%%", success_rate)
                    logger.info("  Deadlocks: %d/%d", deadlocks[agents_num], REPEATS)
                
                    # Save intermediate results
                    intermediate_name = results_dir / f'intermediate_results_{agents_num}_agents'
                
                    with open(str(intermediate_name) + '.txt', 'w') as f:
                        f.write(f"Results for {agents_num} agents:\n")
                        f.write(f"Average episode time: {avg_time:.5f} seconds\n")
                        f.write(f"Average episode length: {avg_length:.2f} steps\n")
                        f.write(f"Average success episode length: {avg_success_length:.2f} steps\n")
                        f.write(f"Success rate: {success_rate:.1f}%\n")
                        f.write(f"Deadlocks: {deadlocks[agents_num]}/{REPEATS}\n")
                        f.write("\nDetailed results:\n")
                        for (agents_num_r, repeat), elapsed_time in sorted(agent_time_results):
                            episode_len = lengths_results[(agents_num_r, repeat)]
                            success_len = success_length_results.get((agents_num_r, repeat))
                            f.write(
                                f"Repeat {repeat}: Time={elapsed_time:.5f}s, Length={episode_len:.2f} steps, "
                                f"Success_len={success_len if success_len is not None else 'nan'}\n"
                            )
                
                    # Save intermediate CSV
                    df_data = []
                    for (agents_num_r, repeat), elapsed_time in agent_time_results:
                        df_data.append({
                            'agents_num': agents_num_r,
                            'repeat': repeat,
                            'episode_runtime_seconds': elapsed_time,
                            'episode_length_steps': lengths_results[(agents_num_r, repeat)],
                            'total_reward': rewards_results[(agents_num_r, repeat)],
                            'total_collisions': collisions_results[(agents_num_r, repeat)],
                            'collision_agent_agent': collision_agent_agent_results[(agents_num_r, repeat)],
                            'collision_agent_obstacle': collision_agent_obstacle_results[(agents_num_r, repeat)],
                            'throughput_steps_per_sec': throughput_results[(agents_num_r, repeat)],
                            'wait_actions': wait_actions_results[(agents_num_r, repeat)],
                            'goal_completion_rate_percent': goal_completion_rate_results[(agents_num_r, repeat)],
                            'average_steps_to_goal': avg_steps_to_goal_results[(agents_num_r, repeat)],
                            'completion_step_deviation': completion_deviation_results[(agents_num_r, repeat)],
                            'success': success_flags.get((agents_num_r, repeat), False),
                            'success_episode_length_steps': success_length_results.get((agents_num_r, repeat)),
                            'path_efficiency': path_efficiency_results.get((agents_num_r, repeat)),
                            'collision_agent_agent_per_agent_step': collision_agent_agent_rate_results.get((agents_num_r, repeat)),
                            'collision_agent_obstacle_per_agent_step': collision_agent_obstacle_rate_results.get((agents_num_r, repeat)),
                            'collision_total_per_agent_step': collision_total_rate_results.get((agents_num_r, repeat)),
                            'deadlock': deadlock_flags.get((agents_num_r, repeat), np.nan),
                        })

                    df = pd.DataFrame(df_data)
                    df.to_csv(str(intermediate_name) + '.csv', index=False)
        
            # Calculate and save final results
            logger.info("")
            logger.info("=" * 80)
            logger.info("FINAL RESULTS")
            logger.info("=" * 80)
        
            summary_data = []
        
            for agents_num in AGENTS_RANGE:
                agent_results = [(k, v) for k, v in time_results.items() if k[0] == agents_num]
                if agent_results:
                    avg_time = np.nanmean([time_results[(agents_num, r)] for r in range(REPEATS)])
                    avg_length = np.nanmean([lengths_results[(agents_num, r)] for r in range(REPEATS)])
                    avg_reward = np.nanmean([rewards_results[(agents_num, r)] for r in range(REPEATS)])
                    avg_collisions = np.nanmean([collisions_results[(agents_num, r)] for r in range(REPEATS)])
                    avg_collision_aa = np.nanmean([collision_agent_agent_results[(agents_num, r)] for r in range(REPEATS)])
                    avg_collision_ao = np.nanmean([collision_agent_obstacle_results[(agents_num, r)] for r in range(REPEATS)])
                    avg_throughput = np.nanmean([throughput_results[(agents_num, r)] for r in range(REPEATS)])
                    avg_wait_actions = np.nanmean([wait_actions_results[(agents_num, r)] for r in range(REPEATS)])
                    avg_goal_rate = np.nanmean([goal_completion_rate_results[(agents_num, r)] for r in range(REPEATS)])
                    avg_completion_deviation = np.nanmean([
                        completion_deviation_results[(agents_num, r)] for r in range(REPEATS)
                    ])
                    avg_success_length = np.nanmean([success_length_results.get((agents_num, r), np.nan) for r in range(REPEATS)])
                    avg_collision_aa_rate = np.nanmean([collision_agent_agent_rate_results.get((agents_num, r), np.nan) for r in range(REPEATS)])
                    avg_collision_ao_rate = np.nanmean([collision_agent_obstacle_rate_results.get((agents_num, r), np.nan) for r in range(REPEATS)])
                    avg_collision_total_rate = np.nanmean([collision_total_rate_results.get((agents_num, r), np.nan) for r in range(REPEATS)])
                    avg_path_efficiency = np.nanmean([path_efficiency_results.get((agents_num, r), np.nan) for r in range(REPEATS)])
                    success_rate_percent = (
                        sum(1 for r in range(REPEATS) if success_flags.get((agents_num, r)))
                        / REPEATS * 100
                    )
                    deadlock_rate_percent = deadlocks[agents_num] / REPEATS * 100

                    summary_data.append({
                        'agents_num': agents_num,
                        'avg_time_seconds': avg_time,
                        'avg_length_steps': avg_length,
                        'avg_success_length_steps': avg_success_length,
                        'avg_reward': avg_reward,
                        'avg_total_collisions': avg_collisions,
                        'avg_collision_agent_agent': avg_collision_aa,
                        'avg_collision_agent_obstacle': avg_collision_ao,
                        'avg_collision_agent_agent_per_agent_step': avg_collision_aa_rate,
                        'avg_collision_agent_obstacle_per_agent_step': avg_collision_ao_rate,
                        'avg_collision_total_per_agent_step': avg_collision_total_rate,
                        'avg_throughput_steps_per_sec': avg_throughput,
                        'avg_wait_actions': avg_wait_actions,
                        'avg_goal_completion_rate_percent': avg_goal_rate,
                        'avg_completion_step_deviation': avg_completion_deviation,
                        'avg_path_efficiency': avg_path_efficiency,
                        'deadlocks': deadlocks[agents_num],
                        'total_runs': REPEATS,
                        'success_rate_percent': success_rate_percent,
                        'deadlock_rate_percent': deadlock_rate_percent,
                    })
                
                    logger.info(
                        "Agents: %2d | Time: %7.5fs | Length: %6.2f steps | Success: %.1f%% | Deadlocks: %d/%d",
                        agents_num,
                        avg_time,
                        avg_length,
                        success_rate_percent,
                        deadlocks[agents_num],
                        REPEATS,
                    )
        
            # Save final summary
            final_name = results_dir / 'final_results'
        
            with open(str(final_name) + '.txt', 'w') as f:
                f.write("=" * 80 + "\n")
                f.write("Final Evaluation Results\n")
                f.write("=" * 80 + "\n\n")
                for item in summary_data:
                    f.write(f"Agents: {item['agents_num']:2d} | ")
                    f.write(f"Time: {item['avg_time_seconds']:7.5f}s | ")
                    f.write(f"Length: {item['avg_length_steps']:6.2f} steps | ")
                    f.write(f"Success: {item['success_rate_percent']:.1f}% | ")
                    f.write(f"Deadlocks: {item['deadlocks']}/{item['total_runs']}")
                    f.write(f" ({item['deadlock_rate_percent']:.1f}% deadlock)\n")
            
                f.write("\n" + "=" * 80 + "\n")
                f.write("Detailed Results\n")
                f.write("=" * 80 + "\n\n")
                for (agents_num, repeat), elapsed_time in sorted(time_results.items()):
                    episode_len = lengths_results[(agents_num, repeat)]
                    f.write(f"Agents: {agents_num}, Repeat: {repeat}, ")
                    f.write(f"Time: {elapsed_time:.5f}s, Length: {episode_len:.2f} steps\n")
        
            # Save final CSV with all data
            df_data = []
            for (agents_num, repeat), elapsed_time in sorted(time_results.items()):
                df_data.append({
                    'agents_num': agents_num,
                    'repeat': repeat,
                    'episode_runtime_seconds': elapsed_time,
                    'episode_length_steps': lengths_results[(agents_num, repeat)],
                    'total_reward': rewards_results[(agents_num, repeat)],
                    'total_collisions': collisions_results[(agents_num, repeat)],
                    'collision_agent_agent': collision_agent_agent_results[(agents_num, repeat)],
                    'collision_agent_obstacle': collision_agent_obstacle_results[(agents_num, repeat)],
                    'throughput_steps_per_sec': throughput_results[(agents_num, repeat)],
                    'wait_actions': wait_actions_results[(agents_num, repeat)],
                    'goal_completion_rate_percent': goal_completion_rate_results[(agents_num, repeat)],
                    'average_steps_to_goal': avg_steps_to_goal_results[(agents_num, repeat)],
                    'completion_step_deviation': completion_deviation_results[(agents_num, repeat)],
                    'success': success_flags.get((agents_num, repeat), False),
                    'success_episode_length_steps': success_length_results.get((agents_num, repeat)),
                    'path_efficiency': path_efficiency_results.get((agents_num, repeat)),
                    'collision_agent_agent_per_agent_step': collision_agent_agent_rate_results.get((agents_num, repeat)),
                    'collision_agent_obstacle_per_agent_step': collision_agent_obstacle_rate_results.get((agents_num, repeat)),
                    'collision_total_per_agent_step': collision_total_rate_results.get((agents_num, repeat)),
                    'deadlock': deadlock_flags.get((agents_num, repeat), np.nan),
                })

            df = pd.DataFrame(df_data)
            df.to_csv(str(final_name) + '.csv', index=False)
        
            # Save summary CSV
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(str(current_results_dir / 'summary.csv'), index=False)

            # Save a compact at-a-glance CSV with headline metrics
            headline_cols = [
                'agents_num',
                'success_rate_percent',
                'deadlock_rate_percent',
                'avg_success_length_steps',
                'avg_throughput_steps_per_sec',
                'avg_collision_agent_agent_per_agent_step',
                'avg_collision_agent_obstacle_per_agent_step',
                'avg_collision_total_per_agent_step',
                'avg_path_efficiency',
            ]
            existing_cols = [c for c in headline_cols if c in summary_df.columns]
            if existing_cols:
                summary_df[existing_cols].to_csv(str(current_results_dir / 'at_a_glance.csv'), index=False)

            # Store summary path for cross-map comparison
            if multi_map_enabled:
                map_summary_paths[map_label] = current_results_dir / 'summary.csv'

            # Generate plots
            logger.info("")
            logger.info("=" * 80)
            logger.info("GENERATING PLOTS")
            logger.info("=" * 80)

            plots_dir = results_dir / "plots"
            plots_dir.mkdir(exist_ok=True)

            detailed_results_csv = Path(str(final_name) + '.csv')
            if detailed_results_csv.exists():
                plot_success_and_deadlocks(detailed_results_csv, plots_dir)
                plot_efficiency(detailed_results_csv, plots_dir)
                plot_collisions(detailed_results_csv, plots_dir)
                logger.info("Plots generated in: %s", plots_dir)
            else:
                logger.warning("Detailed results CSV not found, skipping plotting")

            logger.info("")
            logger.info("=" * 80)
            logger.info("Evaluation completed successfully!")
            logger.info("Results saved to: %s", results_dir)
            if RENDER_VIDEO:
                logger.info("Videos saved to: %s/*.mp4", results_dir)
            logger.info("=" * 80)
        
            return_code = 0

        # Cross-map aggregation
        if multi_map_enabled and len(map_summary_paths) > 1:
            logger.info("")
            logger.info("=" * 80)
            logger.info("CROSS-MAP COMPARISON")
            logger.info("=" * 80)

            cross_map_dir = results_dir / "cross_map_comparison"
            cross_map_dir.mkdir(exist_ok=True)

            # Generate cross-map comparison plots
            if eval_maps_config.get('generate_comparison_plots', True):
                plot_cross_map_comparison(map_summary_paths, cross_map_dir)

            # Create aggregated summary
            if eval_maps_config.get('aggregate_results', True):
                create_cross_map_summary(map_summary_paths, cross_map_dir / 'all_maps_summary.csv')

            logger.info("Cross-map comparison saved to: %s", cross_map_dir)
            logger.info("=" * 80)

    except KeyboardInterrupt:
        logger.warning("Interrupted by user.")
        return_code = 130
    except Exception as e:
        logger.exception("Evaluation failed due to an unexpected error: %s", e)
        return_code = 1
    finally:
        ray.shutdown()

    
    return return_code


if __name__ == "__main__":
    sys.exit(main())
