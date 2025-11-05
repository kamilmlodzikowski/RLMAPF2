#!/usr/bin/env python3
"""Train PPO agents for RLMAPF using structured experiment configs."""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from random import randint
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import ray
from ray.rllib.algorithms.ppo import PPOConfig

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
        type=int,
        default=None,
        help="Override training.random_seed.",
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
    if args.seed is not None:
        overrides.append(f"training.random_seed={args.seed}")
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


def detect_available_gpus() -> int:
    """Detect the number of visible CUDA devices without requiring Ray to be initialised."""
    try:
        import torch

        count = torch.cuda.device_count()
        if isinstance(count, int) and count >= 0:
            return count
    except Exception:
        # Fall back to environment inspection if torch is unavailable or misconfigured.
        logger.debug("Unable to query GPU count via torch", exc_info=True)

    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible:
        devices = [
            device.strip()
            for device in cuda_visible.split(",")
            if device.strip() and device.strip() not in {"-1"}
        ]
        return len(devices)
    return 0


def reconcile_gpu_config(config: TrainConfig) -> None:
    """Adjust GPU-related configuration based on the actual hardware availability."""
    available_gpus = detect_available_gpus()
    requested_gpus = config.hardware.num_gpus

    if available_gpus == 0:
        if requested_gpus and requested_gpus > 0:
            logger.warning(
                "No GPUs detected on this machine. Overriding hardware.num_gpus from %s to 0.",
                requested_gpus,
            )
            config.hardware.num_gpus = 0.0
        for key, value in list(config.model.resources.items()):
            if "gpu" in key.lower() and isinstance(value, (int, float)) and value > 0:
                logger.warning(
                    "No GPUs detected. Overriding model.resources.%s from %s to 0.",
                    key,
                    value,
                )
                config.model.resources[key] = 0
    elif requested_gpus > available_gpus:
        logger.warning(
            "Requested %s GPU(s) but only %s detected. Using %s GPU(s) instead.",
            requested_gpus,
            available_gpus,
            available_gpus,
        )
        config.hardware.num_gpus = float(available_gpus)


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
    if config.training.random_seed is not None:
        env_config.setdefault("seed", config.training.random_seed)
    return env_config


def build_evaluation_env_config(config: TrainConfig) -> Dict[str, Any]:
    env_config = build_environment_config(config)
    eval_overrides = dict(config.evaluation_environment)
    if config.training.random_seed is not None:
        eval_overrides.setdefault("seed", config.training.random_seed)
    else:
        eval_overrides.setdefault("seed", 42)
    env_config.update(eval_overrides)
    return env_config


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
    return model_config


def extract_metric(results: Dict[str, Any], path: Iterable[str]) -> Any:
    value: Any = results
    for key in path:
        value = value[key]
    return value


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
        wandb_metrics[name] = value
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

    if config.run.use_wandb and wandb is None:
        logger.warning("wandb is not installed. Disabling Weights & Biases logging.")
        config.run.use_wandb = False

    reconcile_gpu_config(config)

    run_name = build_run_name(config, args.run_name)
    experiment_dir = (config.paths.experiments_root / run_name).resolve()
    checkpoint_dir = experiment_dir / "checkpoints"
    config_dir = experiment_dir / "config"
    metrics_file = experiment_dir / config.logging.local_metrics_file
    checkpoint_index_path = experiment_dir / "checkpoints.jsonl"
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
    eval_env_config = build_evaluation_env_config(config) if config.training.evaluation_enabled else None
    model_config = prepare_model_config(config)

    ppo_config = (
        PPOConfig()
        .api_stack(**config.model.api_stack)
        .framework(config.model.framework)
        .training(model=model_config)
        .environment(RLMAPF, env_config=env_config)
    )

    if config.training.evaluation_enabled:
        evaluation_kwargs = {
            "evaluation_interval": config.training.eval_interval,
            "evaluation_duration": config.training.evaluation_duration,
            "evaluation_config": {"env_config": eval_env_config},
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
    evaluation_metric_paths = [["evaluation"] + path for path in metric_paths]
    best_metrics: Dict[str, Dict[str, Any]] = {}

    algorithm = None
    return_code = 0

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

                    print("-" * 40)
                    print(f"Episode: {episode}")
                    for key, value in train_metrics.items():
                        print(f"\t{key}: {value}")

                    if "evaluation" in results and config.training.evaluation_enabled:
                        eval_metrics = make_wandb_metrics(
                            {"evaluation": results["evaluation"]},
                            evaluation_metric_paths,
                            agents_num,
                        )
                        update_best_metrics(best_metrics, eval_metrics, episode)
                        print("Evaluation metrics:")
                        for key, value in eval_metrics.items():
                            print(f"\t{key}: {value}")
                        train_metrics.update(eval_metrics)

                    if config.run.use_wandb and wandb_run is not None:
                        wandb_run.log(train_metrics, step=episode)

                    if (
                        config.training.save_interval
                        and episode > 0
                        and episode % config.training.save_interval == 0
                    ):
                        checkpoint_path = checkpoint_dir / f"{run_name}_ep{episode}"
                        state = algorithm.save(str(checkpoint_path))
                        logger.info("Checkpoint saved at %s", checkpoint_path)
                        record_checkpoint(checkpoint_path, episode, checkpoint_index_path)
                        if config.run.use_wandb and wandb_run is not None:
                            wandb_run.log({"checkpoints/latest": state}, step=episode)

                print("=" * 20, "Training finished", "=" * 20)
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
