#!/usr/bin/env python3
"""Train PPO agents for RLMAPF using structured experiment configs."""
from __future__ import annotations

import argparse
import collections
import json
import logging
import math
import numbers
import os
import secrets
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from random import randint
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env.env_runner_group import EnvRunnerGroup
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.rllib.utils.spaces.space_utils import flatten_to_single_ndarray

import matplotlib.pyplot as plt

from rlmapf2 import RLMAPF
from rlmapf_config import (
    TrainConfig,
    apply_overrides,
    dump_config_to_file,
    load_train_config,
    serialise_config,
)

try:
    import wandb  # type: ignore
except ImportError:  # pragma: no cover
    wandb = None  # type: ignore


LOG_LEVEL = logging.INFO
logger = logging.getLogger("train")


class SuccessCallbacks(DefaultCallbacks):
    """Track percentage of agents reaching their goal each episode."""

    def on_episode_end(self, worker, base_env, policies, episode, **kwargs):  # type: ignore[override]
        try:
            # Single environment instance per rollout worker expected here.
            sub_envs = base_env.get_sub_environments()  # type: ignore[attr-defined]
            if not sub_envs:
                return
            env = sub_envs[0]
        except Exception:
            return

        successes = getattr(env, "successful_agents", None)
        initial = getattr(env, "initial_agents_num", None)
        if successes is None or initial is None or initial <= 0:
            return
        success_rate = float(successes) / float(initial)
        # Store as percentage for easier reading in dashboards.
        episode.custom_metrics["success_rate_pct"] = success_rate * 100.0


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO with configurable environment.")
    parser.add_argument(
        "--config",
        default="baseline",
        help="Config name (relative to configs/train) or explicit path to YAML file.",
    )
    parser.add_argument(
        "--config-dir",
        default=None,
        help="Base directory for config lookup when --config is a name (default: configs/train).",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override config values, e.g. --set environment.agents_num=40.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Explicit run name (otherwise derived from config.run.name_prefix).",
    )
    parser.add_argument(
        "--seed",
        "--train-seed",
        dest="train_seed",
        type=int,
        default=None,
        help="Override training.random_seed (training environment seed).",
    )
    parser.add_argument(
        "--eval-seed",
        dest="eval_seed",
        type=int,
        default=None,
        help="Override training.evaluation_seed (evaluation environment seed).",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging even if the config enables it.",
    )

    # Legacy arguments kept for backwards compatibility (mapped to overrides)
    legacy = parser.add_argument_group("legacy overrides")
    legacy.add_argument("--agents_num", type=int, dest="legacy_agents_num", help=argparse.SUPPRESS)
    legacy.add_argument("--max_steps", type=int, dest="legacy_max_steps", help=argparse.SUPPRESS)
    legacy.add_argument(
        "--reward_closer_to_goal_final",
        dest="legacy_reward_closer_to_goal_final",
        help=argparse.SUPPRESS,
    )
    legacy.add_argument(
        "--reward_final_d_star",
        dest="legacy_reward_final_d_star",
        help=argparse.SUPPRESS,
    )
    legacy.add_argument("--use_d_star_lite", dest="legacy_use_d_star_lite", help=argparse.SUPPRESS)
    legacy.add_argument(
        "--use_cnn_observation",
        dest="legacy_use_cnn_observation",
        help=argparse.SUPPRESS,
    )

    return parser.parse_args(argv)


def resolve_config_path(name: str, config_dir: Optional[str], repo_root: Path) -> Path:
    candidate = Path(name)
    if candidate.is_file():
        return candidate.resolve()

    search_dirs = []
    if config_dir:
        search_dirs.append(Path(config_dir))
    search_dirs.append(repo_root / "configs" / "train")

    for base in search_dirs:
        with_suffix = name if name.endswith(".yaml") else f"{name}.yaml"
        path = (base / with_suffix).resolve()
        if path.exists():
            return path
    raise FileNotFoundError(f"Unable to locate config '{name}'. Looked in: {search_dirs}")


def bool_from_arg(value: Optional[str]) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    lowered = value.strip().lower()
    if lowered in {"true", "1", "yes"}:
        return True
    if lowered in {"false", "0", "no"}:
        return False
    raise ValueError(f"Cannot interpret '{value}' as boolean")


def collect_git_info(repo_root: Path) -> Dict[str, Any]:
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
    except (OSError, subprocess.CalledProcessError):
        return {}
    return {"commit": commit, "status": status}


def collect_overrides(args: argparse.Namespace) -> List[str]:
    overrides: List[str] = list(args.overrides or [])
    if args.train_seed is not None:
        overrides.append(f"training.random_seed={args.train_seed}")
    if args.eval_seed is not None:
        overrides.append(f"training.evaluation_seed={args.eval_seed}")
    if args.legacy_agents_num is not None:
        overrides.append(f"environment.agents_num={args.legacy_agents_num}")
    if args.legacy_max_steps is not None:
        overrides.append(f"environment.max_steps={args.legacy_max_steps}")
    legacy_bool = bool_from_arg(args.legacy_reward_closer_to_goal_final)
    if legacy_bool is not None:
        overrides.append(f"environment.reward_closer_to_goal_final={legacy_bool}")
    legacy_bool = bool_from_arg(args.legacy_reward_final_d_star)
    if legacy_bool is not None:
        overrides.append(f"environment.reward_final_d_star={legacy_bool}")
    legacy_bool = bool_from_arg(args.legacy_use_d_star_lite)
    if legacy_bool is not None:
        overrides.append(f"environment.use_d_star_lite={legacy_bool}")
    legacy_bool = bool_from_arg(args.legacy_use_cnn_observation)
    if legacy_bool is not None:
        overrides.append(f"environment.use_cnn_observation={legacy_bool}")
    return overrides


def _has_wandb_credentials() -> bool:
    """Best-effort check that wandb can run without interactive login."""
    if os.environ.get("WANDB_API_KEY") or os.environ.get("WANDB_API_KEY_FILE"):
        return True
    return (Path.home() / ".netrc").exists()


def build_run_name(config: TrainConfig, explicit_name: Optional[str]) -> str:
    if explicit_name:
        return explicit_name
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    random_suffix = f"{randint(0, 9999):04d}"
    return f"{config.run.name_prefix}-{timestamp}-{random_suffix}"


def build_environment_config(config: TrainConfig) -> Dict[str, Any]:
    env_config = dict(config.environment)
    env_config.setdefault("map_path", str(config.paths.map_root))
    if not env_config.get("maps_names_with_variants"):
        raise ValueError("Config must specify environment.maps_names_with_variants")
    # For training envs we only fix the seed when the user explicitly
    # requested one via the environment config itself. Otherwise RLlib
    # will create env instances with their own RNG streams, improving
    # diversity across episodes and workers.
    if env_config.get("seed") is not None:
        env_config["seed"] = int(env_config["seed"])
    return env_config


def build_evaluation_env_config(config: TrainConfig, seed: Optional[int]) -> Dict[str, Any]:
    env_config = build_environment_config(config)
    eval_overrides = dict(config.evaluation_environment)
    env_config.update(eval_overrides)
    # Evaluation keeps a deterministic seed per run unless the user
    # explicitly overrides it in the evaluation_environment block.
    if env_config.get("seed") is None and seed is not None:
        env_config["seed"] = int(seed)
    return env_config


def _generate_seed() -> int:
    return secrets.randbelow(2**32)


def resolve_run_seeds(config: TrainConfig) -> Tuple[int, int, bool, bool]:
    explicit_train_seed = config.environment.get("seed")
    original_train_seed = config.training.random_seed
    # Training seed is kept for reproducibility metadata; the env itself is only
    # forced to this seed when environment.seed is explicitly set.
    if explicit_train_seed is not None:
        training_seed = int(explicit_train_seed)
        config.training.random_seed = training_seed
    else:
        training_seed = (
            int(original_train_seed)
            if original_train_seed is not None
            else _generate_seed()
        )
        config.training.random_seed = training_seed

    explicit_eval_seed = config.evaluation_environment.get("seed")
    original_eval_seed = config.training.evaluation_seed
    if explicit_eval_seed is not None:
        evaluation_seed = explicit_eval_seed
    else:
        evaluation_seed = original_eval_seed if original_eval_seed is not None else _generate_seed()
    config.training.evaluation_seed = evaluation_seed

    training_generated = explicit_train_seed is None and original_train_seed is None
    evaluation_generated = explicit_eval_seed is None and original_eval_seed is None
    return training_seed, evaluation_seed, training_generated, evaluation_generated


def derive_worker_counts(config: TrainConfig) -> Tuple[int, int]:
    """Determine rollout and evaluation worker counts respecting CPU limits."""
    cpus = max(int(config.hardware.num_cpus), 1)
    max_rollout = max(cpus - 1, 0)
    if config.hardware.num_rollout_workers is not None:
        requested_rollout = max(config.hardware.num_rollout_workers, 0)
    else:
        requested_rollout = max_rollout
    rollout_workers = min(requested_rollout, max_rollout)

    remaining_cpus = max(cpus - rollout_workers - 1, 0)
    if config.hardware.num_evaluation_workers is not None:
        requested_eval = max(config.hardware.num_evaluation_workers, 0)
    else:
        requested_eval = 1 if remaining_cpus > 0 else 0
    evaluation_workers = min(requested_eval, remaining_cpus)
    return rollout_workers, evaluation_workers


def prepare_model_config(config: TrainConfig) -> Dict[str, Any]:
    model_config = dict(config.model.model)
    if config.environment.get("use_cnn_observation") and "conv_filters" not in model_config:
        # Provide sensible CNN defaults when not specified.
        model_config.update({
            "conv_activation": "relu",
            "conv_filters": [
                [32, [3, 3], 1],
                [64, [3, 3], 1],
                [64, [3, 3], 1],
            ],
            "fcnet_hiddens": [512, 256],
        })
    recurrent_cfg = config.model.recurrent
    if recurrent_cfg.enabled:
        model_config["use_lstm"] = True
        if recurrent_cfg.cell_size is not None and "lstm_cell_size" not in model_config:
            model_config["lstm_cell_size"] = recurrent_cfg.cell_size
        if recurrent_cfg.max_seq_len is not None and "max_seq_len" not in model_config:
            model_config["max_seq_len"] = recurrent_cfg.max_seq_len
        if (
            recurrent_cfg.use_prev_action is not None
            and "lstm_use_prev_action" not in model_config
        ):
            model_config["lstm_use_prev_action"] = recurrent_cfg.use_prev_action
        if (
            recurrent_cfg.use_prev_reward is not None
            and "lstm_use_prev_reward" not in model_config
        ):
            model_config["lstm_use_prev_reward"] = recurrent_cfg.use_prev_reward
    return model_config


def extract_metric(results: Dict[str, Any], path: Iterable[str]) -> Any:
    value: Any = results
    for key in path:
        value = value[key]
    return value


def _normalise_metric_value(raw_value: Any) -> Optional[Any]:
    """Convert metric values to JSON/W&B friendly scalar types."""
    if isinstance(raw_value, (str, bool)) or raw_value is None:
        return raw_value

    if isinstance(raw_value, np.generic):
        raw_value = raw_value.item()

    if isinstance(raw_value, numbers.Integral):
        return int(raw_value)

    if isinstance(raw_value, numbers.Real):
        value = float(raw_value)
        return value if math.isfinite(value) else None

    if isinstance(raw_value, numbers.Number):
        try:
            value = float(raw_value)
            return value if math.isfinite(value) else None
        except TypeError:
            return None

    if isinstance(raw_value, np.ndarray):
        if raw_value.size == 1:
            scalar = raw_value.item()
            return _normalise_metric_value(scalar)
        return None

    if hasattr(raw_value, "item"):
        try:
            item_value = raw_value.item()
        except Exception:  # pragma: no cover - defensive fallback
            item_value = None
        if isinstance(item_value, numbers.Real):
            value = float(item_value)
            return value if math.isfinite(value) else None

    return None


def make_wandb_metrics(results: Dict[str, Any], metric_paths: List[List[str]], agents_num: int) -> Dict[str, Any]:
    wandb_metrics: Dict[str, Any] = {}
    for path in metric_paths:
        try:
            value = extract_metric(results, path)
        except KeyError:
            continue
        name = "/".join(path)
        if "reward" in name:
            value = value / max(agents_num, 1)
        normalised_value = _normalise_metric_value(value)
        if normalised_value is None:
            logger.warning(
                "Skipping non-serialisable metric %s (type=%s)",
                name,
                type(value).__name__,
            )
            continue
        wandb_metrics[name] = normalised_value
    return wandb_metrics


def update_best_metrics(
    best_metrics: Dict[str, Dict[str, Any]],
    metrics: Dict[str, Any],
    episode: int,
) -> None:
    for key, value in metrics.items():
        if key not in best_metrics:
            mode = "min" if "len" in key else "max"
            baseline = np.inf if mode == "min" else -np.inf
            best_metrics[key] = {"value": baseline, "episode": -1, "mode": mode}
        entry = best_metrics[key]
        if entry["mode"] == "max":
            if value > entry["value"]:
                entry["value"] = value
                entry["episode"] = episode
        else:
            if value < entry["value"]:
                entry["value"] = value
                entry["episode"] = episode


def serialise_best(best_metrics: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    serialised: Dict[str, Dict[str, Any]] = {}
    for key, info in best_metrics.items():
        value = info.get("value")
        if isinstance(value, float) and not math.isfinite(value):
            value = None
        serialised[key] = {
            "value": value,
            "episode": info.get("episode"),
            "mode": info.get("mode"),
        }
    return serialised


def record_checkpoint(path: Path, episode: Optional[int], index_file: Path) -> None:
    entry = {
        "timestamp": time.time(),
        "episode": episode,
        "path": str(path),
    }
    with index_file.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")


def _sanitize_for_path(value: Optional[str], fallback: str) -> str:
    if not value:
        return fallback
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value.strip())
    safe = safe.strip("_")
    return safe or fallback


def _prepare_video_env_config(
    base_env_config: Dict[str, Any],
    videos_root: Path,
    run_component: str,
    wandb_group: Optional[str],
) -> Tuple[Dict[str, Any], Path]:
    env_config = deepcopy(base_env_config)
    env_config["render_mode"] = "human"

    render_config = dict(env_config.get("render_config", {}))
    render_config.setdefault("title", run_component)
    render_config["save_video"] = True
    render_config["show_render"] = False
    render_config["save_frames"] = False

    videos_root.mkdir(parents=True, exist_ok=True)
    date_dir = videos_root / datetime.now().strftime("%Y-%m-%d")
    date_dir.mkdir(parents=True, exist_ok=True)
    group_component = _sanitize_for_path(wandb_group, "no-group")
    folder_name = run_component if wandb_group is None else f"{run_component}_{group_component}"
    run_dir = date_dir / folder_name
    run_dir.mkdir(parents=True, exist_ok=True)

    video_path = run_dir / f"{run_component}.mp4"
    suffix = 1
    while video_path.exists():
        video_path = run_dir / f"{run_component}_{suffix}.mp4"
        suffix += 1

    render_config["video_path"] = str(video_path)
    render_config.setdefault("video_fps", 10)
    render_config.setdefault("video_dpi", 300)
    render_config.setdefault("legend_position", (0, 0))
    render_config.setdefault("frames_path", str(run_dir / "frames"))
    render_config.setdefault("render_delay", 0.0)
    render_config.setdefault("include_legend", True)
    env_config["render_config"] = render_config
    return env_config, video_path


def evaluate_and_record_video(
    algorithm: Any,
    base_env_config: Dict[str, Any],
    videos_root: Path,
    run_component: str,
    wandb_group: Optional[str],
) -> Tuple[Optional[Path], Optional[int]]:
    """Run `algorithm.evaluate` with render enabled and return the saved video path."""
    if algorithm is None:
        return None, None

    base_eval_config = getattr(algorithm, "evaluation_config", None)
    if base_eval_config is None:
        logger.warning("No evaluation config available; skipping video evaluation for %s", run_component)
        return None, None

    video_env_config, video_path = _prepare_video_env_config(
        base_env_config,
        videos_root,
        run_component,
        wandb_group,
    )
    video_fps = (
        video_env_config.get("render_config", {}).get("video_fps")
        if isinstance(video_env_config.get("render_config"), dict)
        else None
    )

    eval_config = base_eval_config.copy(copy_frozen=False)
    eval_config.env_config = video_env_config
    eval_config.evaluation_num_env_runners = 0
    try:
        eval_config.validate()
        eval_config.freeze()
    except Exception:
        logger.exception("Failed to prepare evaluation config for video recording (%s)", run_component)
        return None, video_fps

    _, env_creator = algorithm._get_env_id_and_creator(eval_config.env, eval_config)
    video_eval_group: Optional[EnvRunnerGroup] = None
    original_eval_group = getattr(algorithm, "eval_env_runner_group", None)
    original_eval_config = getattr(algorithm, "evaluation_config", None)

    def _finalise(env: Any) -> None:
        writer = getattr(env, "_video_writer", None)
        if writer is not None:
            try:
                writer.finish()
            except Exception:
                logger.warning("Unable to finalise video writer for %s", run_component, exc_info=True)
            try:
                delattr(env, "_video_writer")
            except Exception:
                pass
        if hasattr(env, "_fig"):
            try:
                plt.close(env._fig)
            except Exception:
                logger.debug("Failed to close matplotlib figure for %s", run_component, exc_info=True)
            try:
                delattr(env, "_fig")
            except Exception:
                pass

    try:
        video_eval_group = EnvRunnerGroup(
            env_creator=env_creator,
            validate_env=None,
            default_policy_class=algorithm.get_default_policy_class(algorithm.config),
            config=eval_config,
            num_env_runners=0,
            logdir=getattr(algorithm, "logdir", None),
            tune_trial_id=getattr(algorithm, "trial_id", None),
        )

        algorithm.eval_env_runner_group = video_eval_group
        algorithm.evaluation_config = eval_config

        try:
            algorithm.evaluate()
        except Exception:
            logger.exception("Failed to run evaluation for video recording (%s)", run_component)
            return None, video_fps

    finally:
        if video_eval_group is not None:
            try:
                video_eval_group.foreach_env(_finalise)
            except Exception:
                logger.debug("Unable to clean up evaluation video environments for %s", run_component, exc_info=True)
            try:
                video_eval_group.stop()
            except Exception:
                logger.warning("Unable to stop video evaluation env runners for %s", run_component, exc_info=True)
        algorithm.eval_env_runner_group = original_eval_group
        algorithm.evaluation_config = original_eval_config

    if not video_path.exists():
        logger.warning("Video file was not created at %s", video_path)
        return None, video_fps

    return video_path, video_fps


def record_policy_video(
    algorithm: Any,
    base_env_config: Dict[str, Any],
    videos_root: Path,
    run_name: str,
    wandb_group: Optional[str],
    max_render_steps: Optional[int] = None,
) -> Optional[Path]:
    """Render a policy rollout to a video file and return the output path."""
    if algorithm is None:
        return None

    env_config = deepcopy(base_env_config)
    render_config = dict(env_config.get("render_config", {}))
    env_config["render_mode"] = "human"
    render_config.setdefault("title", run_name)
    render_config["save_video"] = True
    render_config["show_render"] = False
    render_config["save_frames"] = False

    videos_root.mkdir(parents=True, exist_ok=True)
    date_dir = videos_root / datetime.now().strftime("%Y-%m-%d")
    date_dir.mkdir(parents=True, exist_ok=True)
    group_component = _sanitize_for_path(wandb_group, "no-group")
    folder_name = run_name if wandb_group is None else f"{run_name}_{group_component}"
    run_dir = date_dir / folder_name
    run_dir.mkdir(parents=True, exist_ok=True)

    video_path = run_dir / f"{run_name}.mp4"
    suffix = 1
    while video_path.exists():
        video_path = run_dir / f"{run_name}_{suffix}.mp4"
        suffix += 1

    render_config["video_path"] = str(video_path)
    render_config.setdefault("video_fps", 10)
    render_config.setdefault("video_dpi", 300)
    render_config.setdefault("legend_position", (0, 0))
    render_config.setdefault("frames_path", str(run_dir / "frames"))
    render_config.setdefault("render_delay", 0.0)
    render_config.setdefault("include_legend", True)
    env_config["render_config"] = render_config

    env = None
    policy = None
    policy_id = DEFAULT_POLICY_ID
    try:
        # For render-only rollouts we do not want to reuse the evaluation seed;
        # drop it so the video samples a fresh map/layout like the live evals.
        recording_seed = env_config.pop("seed", None)
        env = RLMAPF(env_config)
        if recording_seed is not None:
            logger.debug("Record policy video ignoring fixed seed %s to capture varied layouts", recording_seed)
        observations, _ = env.reset()  # Random seed to avoid repeating the same map

        try:
            policy = algorithm.get_policy(policy_id)  # type: ignore[attr-defined]
        except Exception:
            policy = algorithm.get_policy()  # type: ignore[attr-defined]
            policy_id = getattr(policy, "policy_id", DEFAULT_POLICY_ID)
        if policy is None:
            raise RuntimeError("Unable to retrieve policy for video recording")

        state_init = policy.get_initial_state()
        use_lstm = len(state_init) > 0

        def _copy_state_template():
            if not state_init:
                return []
            return [np.copy(s) if isinstance(s, np.ndarray) else s for s in state_init]

        action_template = flatten_to_single_ndarray(policy.action_space.sample())  # type: ignore[attr-defined]
        agent_states: collections.defaultdict[str, List[Any]] = collections.defaultdict(_copy_state_template)
        prev_actions = collections.defaultdict(lambda: np.copy(action_template))
        prev_rewards = collections.defaultdict(float)

        max_steps = max_render_steps or env_config.get("max_steps") or env.max_steps
        step = 0
        while step < max_steps:
            # Compute actions per agent to match the trained policy's expected
            # observation structure (avoid passing a multi-agent obs dict directly).
            int_actions: Dict[str, int] = {}
            for agent_id, obs in observations.items():
                policy_state = agent_states[agent_id]
                action_result = policy.compute_single_action(  # type: ignore[attr-defined]
                    obs,
                    state=policy_state,
                    prev_action=prev_actions[agent_id],
                    prev_reward=prev_rewards[agent_id],
                    explore=False,
                )
                action, new_state, _ = action_result
                if use_lstm:
                    agent_states[agent_id] = new_state

                action = flatten_to_single_ndarray(action)
                prev_actions[agent_id] = action

                # RLlib may return numpy scalars/arrays; normalise to ints for the env.
                if isinstance(action, (list, np.ndarray)):
                    action = np.asarray(action).item()
                int_actions[agent_id] = int(action)
            observations, rewards, terminateds, truncateds, _ = env.step(int_actions)
            if isinstance(rewards, dict):
                for agent_id, reward in rewards.items():
                    prev_rewards[agent_id] = reward if reward is not None else 0.0
            step += 1
            if terminateds.get("__all__", False) or truncateds.get("__all__", False):
                break

        logger.info("Saved evaluation video for %s to %s", run_name, video_path)
        if not video_path.exists():
            raise RuntimeError(f"Video file was not created at {video_path}")
        return video_path
    except Exception:
        logger.exception("Failed to record evaluation video for %s", run_name)
        return None
    finally:
        if env is not None and hasattr(env, "_video_writer"):
            try:
                env._video_writer.finish()
            except Exception:
                logger.warning("Unable to finalise video writer for %s", run_name, exc_info=True)
        if env is not None and hasattr(env, "_fig"):
            try:
                plt.close(env._fig)
            except Exception:
                logger.debug("Failed to close matplotlib figure for %s", run_name, exc_info=True)


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parent

    config_path = resolve_config_path(args.config, args.config_dir, repo_root)
    config = load_train_config(config_path, repo_root)
    if args.no_wandb:
        config.run.use_wandb = False
    overrides = collect_overrides(args)
    if overrides:
        config = apply_overrides(config, overrides, repo_root)

    train_seed, eval_seed, train_generated, eval_generated = resolve_run_seeds(config)
    if train_generated:
        logger.info("Training seed not provided; generated %s", train_seed)
    else:
        logger.info("Training seed: %s", train_seed)
    if eval_generated:
        logger.info("Evaluation seed not provided; generated %s", eval_seed)
    else:
        logger.info("Evaluation seed: %s", eval_seed)

    rollout_workers, evaluation_workers = derive_worker_counts(config)
    logger.info("Rollout workers: %s", rollout_workers)
    if config.training.evaluation_enabled:
        logger.info("Evaluation workers: %s", evaluation_workers)
    else:
        evaluation_workers = 0

    if config.run.use_wandb and wandb is None:
        logger.warning("wandb is not installed. Disabling Weights & Biases logging.")
        config.run.use_wandb = False
    if config.run.use_wandb and not _has_wandb_credentials():
        logger.warning(
            "wandb credentials not configured (no API key or ~/.netrc). Disabling Weights & Biases logging."
        )
        config.run.use_wandb = False

    run_name = build_run_name(config, args.run_name)
    experiment_dir = (config.paths.experiments_root / run_name).resolve()
    checkpoint_dir = experiment_dir / "checkpoints"
    config_dir = experiment_dir / "config"
    metrics_file = experiment_dir / config.logging.local_metrics_file
    checkpoint_index_path = experiment_dir / "checkpoints.jsonl"
    videos_root = (repo_root / "videos").resolve()
    experiment_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)

    dump_config_to_file(config, config_dir / "resolved_config.yaml")
    (config_dir / "resolved_config.json").write_text(serialise_config(config), encoding="utf-8")

    namespace = config.run.namespace or config.run.group
    git_info = collect_git_info(repo_root) if config.logging.log_git_info else {}
    run_metadata = {
        "run_name": run_name,
        "config_path": str(config_path),
        "resolved_config": str((config_dir / "resolved_config.yaml").relative_to(experiment_dir)),
        "start_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "ray_namespace": namespace,
        "metrics_file": config.logging.local_metrics_file,
        "checkpoints_index": checkpoint_index_path.name,
        "git": git_info,
        "training_seed": train_seed,
        "evaluation_seed": eval_seed,
        "seeds_generated": {
            "training": train_generated,
            "evaluation": eval_generated,
        },
        "workers": {
            "rollout": rollout_workers,
            "evaluation": evaluation_workers,
        },
    }
    (experiment_dir / "run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    logger.info("Run name: %s", run_name)
    logger.info("Experiment directory: %s", experiment_dir)
    logger.info("Using config from %s", config_path)
    ray_context = ray.init(
        num_cpus=config.hardware.num_cpus,
        num_gpus=config.hardware.num_gpus,
        namespace=namespace,
    )
    logger.info("Ray initialized (dashboard: %s)", ray_context.dashboard_url)

    env_config = build_environment_config(config)
    eval_env_config = (
        build_evaluation_env_config(config, seed=eval_seed)
        if config.training.evaluation_enabled
        else None
    )
    video_env_config_for_eval = eval_env_config if eval_env_config is not None else env_config
    model_config = prepare_model_config(config)

    ppo_config = (
        PPOConfig()
        .api_stack(**config.model.api_stack)
        .framework(config.model.framework)
        .training(model=model_config)
        .env_runners(num_env_runners=rollout_workers)
        .environment(RLMAPF, env_config=env_config)
        .callbacks(SuccessCallbacks)
    )

    if config.training.evaluation_enabled:
        evaluation_kwargs = {
            "evaluation_interval": config.training.eval_interval,
            "evaluation_duration": config.training.evaluation_duration,
            "evaluation_config": {"env_config": eval_env_config},
            "evaluation_num_workers": evaluation_workers,
        }
        if config.training.evaluation_num_episodes is not None:
            evaluation_kwargs["evaluation_duration_unit"] = "episodes"
            evaluation_kwargs["evaluation_duration"] = config.training.evaluation_num_episodes
        ppo_config = ppo_config.evaluation(**evaluation_kwargs)
    else:
        ppo_config = ppo_config.evaluation(evaluation_interval=None)

    resource_kwargs = dict(config.model.resources)
    ppo_config = ppo_config.resources(num_gpus=config.hardware.num_gpus, **resource_kwargs)

    wandb_run = None
    if config.run.use_wandb:
        wandb_run = wandb.init(
            project=config.run.project,
            group=config.run.group,
            tags=config.run.tags,
            notes=config.run.notes,
            name=run_name,
            config=config.to_nested_dict(),
        )

    metric_paths = [metric.split("/") for metric in config.logging.params_to_log]
    success_metric_primary = ["custom_metrics", "success_rate_pct_mean"]
    success_metric_alt = ["env_runners", "custom_metrics", "success_rate_pct_mean"]
    if success_metric_primary not in metric_paths:
        metric_paths.append(success_metric_primary)
    if success_metric_alt not in metric_paths:
        metric_paths.append(success_metric_alt)
    evaluation_metric_paths = [["evaluation"] + path for path in metric_paths]
    best_metrics: Dict[str, Dict[str, Any]] = {}

    algorithm = None
    return_code = 0
    video_output_path: Optional[Path] = None

    try:
        with metrics_file.open("a", encoding="utf-8") as metrics_handle:
            algorithm = ppo_config.build()
            logger.info("Starting training for %s episodes", config.training.episodes)

            agents_num = env_config.get("agents_num", 1)

            try:
                for episode in range(config.training.episodes):
                    results = algorithm.train()
                    train_metrics = make_wandb_metrics(results, metric_paths, agents_num)
                    update_best_metrics(best_metrics, train_metrics, episode)

                    log_entry = {
                        "timestamp": time.time(),
                        "episode": episode,
                        "metrics": train_metrics,
                    }
                    metrics_handle.write(json.dumps(log_entry) + "\n")
                    metrics_handle.flush()

                    logger.info("-" * 40)
                    logger.info("Episode: %s", episode)
                    for key, value in train_metrics.items():
                        logger.info("  %s: %s", key, value)

                    eval_video_path: Optional[Path] = None

                    if "evaluation" in results and config.training.evaluation_enabled:
                        eval_metrics = make_wandb_metrics(
                            {"evaluation": results["evaluation"]},
                            evaluation_metric_paths,
                            agents_num,
                        )
                        update_best_metrics(best_metrics, eval_metrics, episode)
                        logger.info("Evaluation metrics:")
                        for key, value in eval_metrics.items():
                            logger.info("  %s: %s", key, value)
                        train_metrics.update(eval_metrics)
                        if config.run.use_wandb and wandb_run is not None:
                            eval_run_component = f"{run_name}_eval_ep{episode}"
                            eval_video_path, eval_video_fps = evaluate_and_record_video(
                                algorithm,
                                video_env_config_for_eval,
                                videos_root,
                                eval_run_component,
                                config.run.group,
                            )
                            if eval_video_path is None:
                                logger.warning("Evaluation video recording failed for episode %s", episode)
                            else:
                                wandb_run.log(
                                    {
                                        "evaluation/video": wandb.Video(
                                            str(eval_video_path),
                                            fps=eval_video_fps or 10,
                                            format="mp4",
                                        )
                                    },
                                    step=episode,
                                    commit=False,
                                )

                    if config.run.use_wandb and wandb_run is not None:
                        wandb_payload = dict(train_metrics)
                        wandb_run.log(wandb_payload, step=episode)

                    if (
                        config.training.save_interval
                        and episode > 0
                        and episode % config.training.save_interval == 0
                    ):
                        checkpoint_path = checkpoint_dir / f"{run_name}_ep{episode}"
                        _ = algorithm.save(str(checkpoint_path))
                        logger.info("Checkpoint saved at %s", checkpoint_path)
                        record_checkpoint(checkpoint_path, episode, checkpoint_index_path)
                        if config.run.use_wandb and wandb_run is not None:
                            wandb_run.log({"checkpoints/latest": str(checkpoint_path)}, step=episode)

                logger.info("%s Training finished %s", "=" * 20, "=" * 20)
                final_checkpoint = checkpoint_dir / f"{run_name}_final"
                algorithm.save(str(final_checkpoint))
                logger.info("Final model saved at %s", final_checkpoint)
                record_checkpoint(final_checkpoint, config.training.episodes, checkpoint_index_path)
                summary_path = experiment_dir / "run_summary.json"
                summary_payload = {
                    "run_name": run_name,
                    "episodes": config.training.episodes,
                    "final_checkpoint": str(final_checkpoint),
                    "best_metrics": serialise_best(best_metrics),
                }
                video_output_path = record_policy_video(
                    algorithm,
                    video_env_config_for_eval,
                    videos_root,
                    run_name,
                    config.run.group,
                )
                if video_output_path is not None:
                    summary_payload["video_path"] = str(video_output_path)
                summary_path.write_text(
                    json.dumps(summary_payload, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
                if config.run.use_wandb and wandb_run is not None:
                    wandb_run.summary["final_checkpoint"] = str(final_checkpoint)
                    best_for_logging = serialise_best(best_metrics)
                    for key, entry in best_for_logging.items():
                        wandb_run.summary[f"best/{key}"] = entry["value"]
                        wandb_run.summary[f"best_episode/{key}"] = entry["episode"]
                    if video_output_path is not None:
                        video_fps = (
                            video_env_config_for_eval.get("render_config", {}).get("video_fps")
                            if isinstance(video_env_config_for_eval.get("render_config"), dict)
                            else None
                        )
                        wandb_run.log(
                            {
                                "video/demo": wandb.Video(
                                    str(video_output_path),
                                    fps=video_fps or 10,
                                    format="mp4",
                                )
                            },
                            step=config.training.episodes,
                        )
                        wandb_run.summary["video/demo_path"] = str(video_output_path)
            except KeyboardInterrupt:
                logger.warning("Interrupted by user. Saving latest checkpoint before exit.")
                if algorithm is not None:
                    final_checkpoint = checkpoint_dir / f"{run_name}_interrupt"
                    algorithm.save(str(final_checkpoint))
                    logger.info("Interrupted checkpoint saved at %s", final_checkpoint)
                    record_checkpoint(final_checkpoint, None, checkpoint_index_path)
                    summary_path = experiment_dir / "run_summary.json"
                    summary_payload = {
                        "run_name": run_name,
                        "episodes": config.training.episodes,
                        "final_checkpoint": str(final_checkpoint),
                        "status": "interrupted",
                        "best_metrics": serialise_best(best_metrics),
                    }
                    video_output_path = record_policy_video(
                        algorithm,
                        video_env_config_for_eval,
                        videos_root,
                        run_name,
                        config.run.group,
                    )
                    if video_output_path is not None:
                        summary_payload["video_path"] = str(video_output_path)
                    summary_path.write_text(
                        json.dumps(summary_payload, indent=2, sort_keys=True),
                        encoding="utf-8",
                    )
                return_code = 130
            except Exception:
                logger.exception("Training failed due to an unexpected error.")
                return_code = 1
                raise
            else:
                return_code = 0
    finally:
        if config.run.use_wandb and wandb_run is not None:
            wandb_run.finish()
        ray.shutdown()

    return return_code


if __name__ == "__main__":
    sys.exit(main())
