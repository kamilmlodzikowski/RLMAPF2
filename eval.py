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
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import ray
import yaml
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.ppo import PPOConfig

from rlmapf2 import RLMAPF

# Ignore deprecation warnings
os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning"

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
        help="Path to model checkpoint directory.",
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
        type=int,
        default=None,
        help="Specific agent count to render video for (renders only this count).",
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


def is_valid_checkpoint_dir(path: Path) -> bool:
    """Return True if the directory looks like an RLlib checkpoint."""
    if not path.is_dir():
        return False
    for filename in ("algorithm_state.pkl", "algorithm_state.msgpck"):
        if (path / filename).is_file():
            return True
    return any(child.is_dir() and child.name.startswith("checkpoint-") for child in path.iterdir())


def resolve_checkpoint_path(path: Path) -> Path:
    """
    Resolve a user-provided checkpoint path to a valid RLlib checkpoint directory.

    Handles the common cases where the user points to a parent 'checkpoints' folder
    instead of the checkpoint directory itself.
    """
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {path}")

    if path.is_file() or is_valid_checkpoint_dir(path):
        return path

    candidates = [
        child for child in path.iterdir()
        if is_valid_checkpoint_dir(child)
    ]
    if candidates:
        # Pick the most recently modified checkpoint directory.
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        resolved = candidates[0]
        logger.info("Using checkpoint directory '%s' inside '%s'", resolved, path)
        return resolved

    raise ValueError(f"Given checkpoint path does not contain a valid checkpoint: {path}")


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
    
    # CRITICAL: For video rendering, use "human" mode (not "rgb_array")!
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

def plot_evaluation_summary(summary_path: Path, save_path: Optional[Path] = None) -> None:
    """
    Create a multi-subplot figure with all numeric metrics from summary.csv.
    Each metric gets its own subplot automatically.
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_csv(summary_path)
    print(f"[Plot] Loaded {len(df)} rows from {summary_path}")
    df = df.sort_values("agents_num")

    # Select numeric columns except 'agents_num'
    numeric_cols = [c for c in df.select_dtypes(include=['number']).columns if c != "agents_num"]
    if not numeric_cols:
        print("[Plot] No numeric metrics found.")
        return

    n_metrics = len(numeric_cols)
    ncols = 2
    nrows = (n_metrics + 1) // ncols

    plt.style.use("seaborn-v0_8-muted")
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4 * nrows))
    axes = axes.flatten()

    for i, metric in enumerate(numeric_cols):
        ax = axes[i]
        ax.plot(df["agents_num"], df[metric], marker="o", linewidth=2)
        ax.set_title(metric.replace("_", " ").title())
        ax.set_xlabel("Number of Agents")
        ax.set_ylabel(metric)
        ax.grid(True)

    # Hide unused subplots if any
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("RLMAPF Evaluation Metrics Summary", fontsize=16, weight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"[Plot] ✅ Saved summary plot to {save_path}")
    else:
        plt.show()



def main(argv: Optional[List[str]] = None) -> int:
    """Main evaluation function."""
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
    )
    
    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parent
    
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
    
    # Resolve checkpoint path, allowing users to point at parent directories
    raw_checkpoint_path = Path(args.checkpoint).expanduser()
    resolved_checkpoint_path = resolve_checkpoint_path(raw_checkpoint_path.resolve())
    CHECKPOINT_PATH = str(resolved_checkpoint_path)
    
    # Parse evaluation parameters
    agent_range_str = config.get('eval_agents_range', '4-20')
    AGENTS_RANGE = parse_agent_range(agent_range_str)
    
    NUM_THREADS = args.num_threads if args.num_threads is not None else config.get('eval_num_threads', 1)
    REPEATS = args.repeats if args.repeats is not None else config.get('eval_repeats', 10)
    
    # Video rendering settings
    RENDER_VIDEO = args.render_video
    VIDEO_AGENTS = args.video_agents  # If set, only render video for this agent count
    
    # If rendering video, only do 1 repeat (to save time) unless specified otherwise
    if RENDER_VIDEO and args.repeats is None:
        REPEATS = 1
        logger.info("Video rendering enabled - setting repeats to 1 (override with --repeats if needed)")
    
    # Build run name and create directories
    run_name = build_run_name(config, args.run_name)
    
    if args.results_dir:
        results_dir = Path(args.results_dir).resolve()
    else:
        results_base = config.get('paths', {}).get('experiments_root', 'experiments')
        results_dir = (Path(results_base) / run_name).resolve()
    
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
            logger.info("Video rendering: ENABLED (only for %d agents)", VIDEO_AGENTS)
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
        
        # Build and restore the algorithm
        logger.info("Building algorithm and loading checkpoint...")
        algorithm = algorithm_config_builder.build()
        algorithm.restore(CHECKPOINT_PATH)
        logger.info("Checkpoint loaded successfully!")
        
        # Setup results tracking
        time_results = {}
        lengths_results = {}
        deadlocks = {}
        rewards_results = {}
        collisions_results = {}
        throughput_results = {}
        efficiency_results = {}

        
        def run_repeat(agents_num: int, repeat: int) -> tuple:
            """Run a single evaluation repeat."""
            logger.info("Evaluating with %d agents, repeat %d/%d", agents_num, repeat + 1, REPEATS)
            
            info = {}  # ✅ ensure defined early
            
            # Check if we should render video for this agent count
            should_render = RENDER_VIDEO and (VIDEO_AGENTS is None or VIDEO_AGENTS == agents_num)
            
            # Setup video path if rendering
            video_path = None
            if should_render:
                video_filename = f"evaluation_{agents_num}agents_repeat{repeat}.mp4"
                video_path = str(results_dir / video_filename)
                logger.info("Rendering video to: %s", video_path)
            
            # Run episode manually for rendering
            if should_render:
                env_config_dict = configure_env(
                    config,
                    agents_num=agents_num,
                    render_mode=None,
                    seed=42 + repeat,
                    render_video=True,
                    video_path=video_path
                )
                
                env = RLMAPF(env_config_dict)
                temp_algorithm = algorithm_config_builder.build()
                temp_algorithm.restore(CHECKPOINT_PATH)
                policy = temp_algorithm.get_policy()
                
                start_time = time.time()
                obs, info = env.reset()
                done = False
                step_count = 0
                max_steps = env_config_dict['max_steps']
                
                while not done and step_count < max_steps:
                    actions = {agent_id: policy.compute_single_action(agent_obs)[0] for agent_id, agent_obs in obs.items()}
                    obs, rewards, dones, truncated, info = env.step(actions)
                    step_count += 1
                    done = all(dones.values()) or all(truncated.values())
                
                env.finalize_video()
                env.close()
                elapsed_time = time.time() - start_time
                episode_len = step_count
                
                del temp_algorithm
                logger.info("Video saved to: %s", video_path)
            
            else:
                # RLlib internal evaluation
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
                    # ✅ Create fake info if RLlib doesn’t provide one
                    info = {"episode_reward": results.get('env_runners', {}).get('episode_reward_mean', np.nan)}
                except Exception as e:
                    logger.error("Error during evaluation: %s", e)
                    return (agents_num, repeat, None, None, 1, np.nan, 0, 0, np.nan)
            
            # ✅ Safe metrics extraction
            collisions = info.get("total_collisions", 0)
            reward_sum = info.get("episode_reward", np.nan)
            optimal_len = info.get("optimal_length", np.nan)
            throughput = episode_len / max(elapsed_time, 1e-9)
            efficiency = (optimal_len / episode_len) if (optimal_len and episode_len) else np.nan
            deadlock = 0 if episode_len < env_config.get('max_steps', 250) else 1
            
            logger.debug(
                "Completed in %.4fs | steps=%d | reward=%.3f | collisions=%d | throughput=%.2f/s | eff=%.3f",
                elapsed_time, episode_len, reward_sum, collisions, throughput, efficiency
            )
            
            return (
                agents_num, repeat, elapsed_time, episode_len,
                deadlock, reward_sum, collisions, throughput, efficiency
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
                    (agents_num_r, repeat_r, elapsed_time, episode_len,
                     deadlock, reward_sum, collisions, throughput, efficiency) = future.result()

                    
                    # Save all results, even if deadlocked
                    if elapsed_time is not None:
                        time_results[(agents_num_r, repeat_r)] = elapsed_time
                    if episode_len is not None:
                        lengths_results[(agents_num_r, repeat_r)] = episode_len
                    rewards_results[(agents_num_r, repeat_r)] = reward_sum
                    collisions_results[(agents_num_r, repeat_r)] = collisions
                    throughput_results[(agents_num_r, repeat_r)] = throughput
                    efficiency_results[(agents_num_r, repeat_r)] = efficiency
                    deadlocks[agents_num_r] += deadlock

            
            # Calculate and save intermediate results
            agent_time_results = [(k, v) for k, v in time_results.items() if k[0] == agents_num]
            agent_length_results = [(k, v) for k, v in lengths_results.items() if k[0] == agents_num]
            
            if agent_time_results:
                avg_time = sum(v for _, v in agent_time_results) / len(agent_time_results)
                avg_length = sum(v for _, v in agent_length_results) / len(agent_length_results)
                
                logger.info("")
                logger.info("Results for %d agents:", agents_num)
                logger.info("  Average time: %.5f seconds", avg_time)
                logger.info("  Average length: %.2f steps", avg_length)
                logger.info("  Deadlocks: %d/%d", deadlocks[agents_num], REPEATS)
                
                # Save intermediate results
                intermediate_name = results_dir / f'intermediate_results_{agents_num}_agents'
                
                with open(str(intermediate_name) + '.txt', 'w') as f:
                    avg_reward = np.nanmean([rewards_results[(agents_num, r)] for r in range(REPEATS)])
                    avg_collisions = np.nanmean([collisions_results[(agents_num, r)] for r in range(REPEATS)])
                    avg_throughput = np.nanmean([throughput_results[(agents_num, r)] for r in range(REPEATS)])
                    avg_efficiency = np.nanmean([efficiency_results[(agents_num, r)] for r in range(REPEATS)])

                    f.write(f"Average reward: {avg_reward:.3f}\n")
                    f.write(f"Average collisions: {avg_collisions:.2f}\n")
                    f.write(f"Average throughput: {avg_throughput:.2f} steps/s\n")
                    f.write(f"Average path efficiency: {avg_efficiency:.3f}\n")
                    f.write(f"Results for {agents_num} agents:\n")
                    f.write(f"Average time: {avg_time:.5f} seconds\n")
                    f.write(f"Average length: {avg_length:.2f} steps\n")
                    f.write(f"Deadlocks: {deadlocks[agents_num]}/{REPEATS}\n")
                    f.write("\nDetailed results:\n")
                    for (agents_num_r, repeat), elapsed_time in sorted(agent_time_results):
                        episode_len = lengths_results[(agents_num_r, repeat)]
                        f.write(f"Repeat {repeat}: Time={elapsed_time:.5f}s, Length={episode_len:.2f}\n")
                
                # Save intermediate CSV
                df_data = []
                for (agents_num_r, repeat), elapsed_time in agent_time_results:
                    df_data.append({
                        'agents_num': agents_num_r,
                        'repeat': repeat,
                        'elapsed_time': elapsed_time,
                        'episode_length': lengths_results[(agents_num_r, repeat)],
                        'reward': rewards_results[(agents_num_r, repeat)],
                        'collisions': collisions_results[(agents_num_r, repeat)],
                        'throughput': throughput_results[(agents_num_r, repeat)],
                        'path_efficiency': efficiency_results[(agents_num_r, repeat)]
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
                avg_time = sum(v for _, v in agent_results) / len(agent_results)
                agent_lengths = [(k, v) for k, v in lengths_results.items() if k[0] == agents_num]
                avg_length = sum(v for _, v in agent_lengths) / len(agent_lengths)
                avg_reward = np.nanmean([rewards_results[(agents_num, r)] for r in range(REPEATS)])
                avg_collisions = np.nanmean([collisions_results[(agents_num, r)] for r in range(REPEATS)])
                avg_throughput = np.nanmean([throughput_results[(agents_num, r)] for r in range(REPEATS)])
                avg_efficiency = np.nanmean([efficiency_results[(agents_num, r)] for r in range(REPEATS)])

                
                summary_data.append({
                    'agents_num': agents_num,
                    'avg_time': avg_time,
                    'avg_length': avg_length,
                    'avg_reward': avg_reward,
                    'avg_collisions': avg_collisions,
                    'avg_throughput': avg_throughput,
                    'avg_efficiency': avg_efficiency,
                    'deadlocks': deadlocks[agents_num],
                    'total_runs': REPEATS,
                    'success_rate': (REPEATS - deadlocks[agents_num]) / REPEATS * 100
                })

                
                logger.info(
                    "Agents: %2d | Time: %6.3fs | Len: %5.1f | Rwd: %7.3f | Coll: %5.2f | Thr: %5.2f/s | Eff: %.3f | Dead: %d/%d (%.1f%%)",
                    agents_num, avg_time, avg_length, avg_reward, avg_collisions,
                    avg_throughput, avg_efficiency, deadlocks[agents_num],
                    REPEATS, (REPEATS - deadlocks[agents_num]) / REPEATS * 100
                )

        
        # Save final summary
        final_name = results_dir / 'final_results'
        
        with open(str(final_name) + '.txt', 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("Final Evaluation Results\n")
            f.write("=" * 80 + "\n\n")
            for item in summary_data:
                f.write(f"Agents: {item['agents_num']:2d} | ")
                f.write(f"Time: {item['avg_time']:7.5f}s | ")
                f.write(f"Length: {item['avg_length']:6.2f} | ")
                f.write(f"Deadlocks: {item['deadlocks']}/{item['total_runs']} ")
                f.write(f"({item['success_rate']:.1f}% success)\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("Detailed Results\n")
            f.write("=" * 80 + "\n\n")
            for (agents_num, repeat), elapsed_time in sorted(time_results.items()):
                episode_len = lengths_results[(agents_num, repeat)]
                f.write(f"Agents: {agents_num}, Repeat: {repeat}, ")
                f.write(f"Time: {elapsed_time:.5f}s, Length: {episode_len:.2f}\n")
        
        # Save final CSV with all data
        df_data = []
        for (agents_num, repeat), elapsed_time in sorted(time_results.items()):
            df_data.append({
                'agents_num': agents_num,
                'repeat': repeat,
                'elapsed_time': elapsed_time,
                'episode_length': lengths_results[(agents_num, repeat)],
                'reward': rewards_results[(agents_num, repeat)],
                'collisions': collisions_results[(agents_num, repeat)],
                'throughput': throughput_results[(agents_num, repeat)],
                'path_efficiency': efficiency_results[(agents_num, repeat)]
            })

        
        df = pd.DataFrame(df_data)
        df.to_csv(str(final_name) + '.csv', index=False)
        
        # Save summary CSV
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(str(results_dir / 'summary.csv'), index=False)

        # === Plot summary automatically ===
        summary_csv = results_dir / "summary.csv"
        if summary_csv.exists():
            plot_evaluation_summary(summary_csv, results_dir / "evaluation_summary.png")
        else:
            logger.warning("Summary CSV not found, skipping plot generation.")

        logger.info("")
        logger.info("=" * 80)
        logger.info("Evaluation completed successfully!")
        logger.info("Results saved to: %s", results_dir)
        if RENDER_VIDEO:
            logger.info("Videos saved to: %s/*.mp4", results_dir)
        logger.info("=" * 80)
        
        return_code = 0
        
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
