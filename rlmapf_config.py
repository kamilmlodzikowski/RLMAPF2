#!/usr/bin/env python3
"""Configuration helpers and evaluation CLI for RLMAPF2."""

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import ray
import yaml
from ray.rllib.algorithms.ppo import PPOConfig

from rlmapf2 import RLMAPF

# Optional wandb
try:
    import wandb  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    wandb = None  # type: ignore


def _deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``updates`` into ``base`` (modifies and returns ``base``)."""
    for key, value in updates.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _cast_value(value: str) -> Any:
    """Best-effort cast of string overrides to Python objects."""
    value = value.strip()
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.lower() in {"null", "none"}:
        return None
    try:
        if value.startswith("0") and value not in {"0", "0.0"}:
            # Keep strings with leading zeros (likely IDs) intact
            raise ValueError
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def _set_by_path(data: Dict[str, Any], key_path: Iterable[str], value: Any) -> None:
    """Set ``value`` in ``data`` following ``key_path`` (dot-separated)."""
    current = data
    key_list = list(key_path)
    for key in key_list[:-1]:
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]
    current[key_list[-1]] = value


@dataclass
class RunConfig:
    name_prefix: str = "run"
    project: Optional[str] = None
    group: Optional[str] = None
    use_wandb: bool = True
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    resume_from: Optional[str] = None
    namespace: Optional[str] = None


@dataclass
class HardwareConfig:
    num_cpus: int = 4
    num_gpus: float = 1.0
    num_rollout_workers: Optional[int] = None
    num_evaluation_workers: Optional[int] = None
    extra_resources: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelConfig:
    framework: str = "torch"
    api_stack: Dict[str, Any] = field(
        default_factory=lambda: {
            "enable_rl_module_and_learner": False,
            "enable_env_runner_and_connector_v2": False,
        }
    )
    model: Dict[str, Any] = field(
        default_factory=lambda: {
            "fcnet_hiddens": [1024, 512, 512],
            "fcnet_activation": "relu",
        }
    )
    resources: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingSection:
    episodes: int = 500
    save_interval: int = 500
    eval_interval: int = 50
    evaluation_duration: int = 10
    evaluation_enabled: bool = True
    evaluation_num_episodes: Optional[int] = None
    stop_reward_mean: Optional[float] = None
    random_seed: Optional[int] = None
    evaluation_seed: Optional[int] = 42


@dataclass
class PathsConfig:
    save_dir: Path = Path("saved_models")
    experiments_root: Path = Path("experiments")
    map_root: Path = Path("maps")

    def resolve(self, repo_root: Path) -> None:
        self.save_dir = (repo_root / self.save_dir).resolve()
        self.experiments_root = (repo_root / self.experiments_root).resolve()
        self.map_root = (repo_root / self.map_root).resolve()


@dataclass
class LoggingConfig:
    params_to_log: List[str] = field(
        default_factory=lambda: [
            "env_runners/episode_reward_mean",
            "env_runners/episode_reward_min",
            "env_runners/episode_reward_max",
            "env_runners/episode_len_mean",
        ]
    )
    best_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    local_metrics_file: str = "metrics.jsonl"
    log_git_info: bool = True


@dataclass
class TrainConfig:
    run: RunConfig = field(default_factory=RunConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingSection = field(default_factory=TrainingSection)
    environment: Dict[str, Any] = field(default_factory=dict)
    evaluation_environment: Dict[str, Any] = field(default_factory=dict)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)

    @classmethod
    def from_dict(cls, data: Dict[str, Any], repo_root: Path) -> "TrainConfig":
        base = cls()
        merged_dict = asdict(base)
        merged_dict["paths"] = {
            "save_dir": str(base.paths.save_dir),
            "experiments_root": str(base.paths.experiments_root),
            "map_root": str(base.paths.map_root),
        }
        merged_dict["hardware"]["extra_resources"] = dict(base.hardware.extra_resources)
        merged_dict["model"]["api_stack"] = dict(base.model.api_stack)
        merged_dict["model"]["model"] = dict(base.model.model)
        merged_dict["model"]["resources"] = dict(base.model.resources)
        merged_dict["logging"]["best_metrics"] = dict(base.logging.best_metrics)

        _deep_update(merged_dict, data)

        train_config = cls(
            run=RunConfig(**merged_dict["run"]),
            hardware=HardwareConfig(**merged_dict["hardware"]),
            model=ModelConfig(
                framework=merged_dict["model"].get("framework", "torch"),
                api_stack=dict(merged_dict["model"].get("api_stack", {})),
                model=dict(merged_dict["model"].get("model", {})),
                resources=dict(merged_dict["model"].get("resources", {})),
            ),
            training=TrainingSection(**merged_dict["training"]),
            environment=dict(merged_dict.get("environment", {})),
            evaluation_environment=dict(merged_dict.get("evaluation_environment", {})),
            logging=LoggingConfig(**merged_dict["logging"]),
            paths=PathsConfig(
                save_dir=Path(merged_dict["paths"]["save_dir"]),
                experiments_root=Path(merged_dict["paths"]["experiments_root"]),
                map_root=Path(merged_dict["paths"]["map_root"]),
            ),
        )
        train_config.paths.resolve(repo_root)
        return train_config

    def to_nested_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["paths"] = {
            "save_dir": str(self.paths.save_dir),
            "experiments_root": str(self.paths.experiments_root),
            "map_root": str(self.paths.map_root),
        }
        return data


def load_train_config(config_path: Path, repo_root: Path) -> TrainConfig:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return TrainConfig.from_dict(data, repo_root=repo_root)


def apply_overrides(base_config: TrainConfig, overrides: List[str], repo_root: Path) -> TrainConfig:
    if not overrides:
        return base_config
    config_dict = base_config.to_nested_dict()
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid override '{override}'. Use key=value format.")
        key, raw_value = override.split("=", 1)
        key = key.strip()
        value = _cast_value(raw_value)
        _set_by_path(config_dict, key.split("."), value)
    return TrainConfig.from_dict(config_dict, repo_root=repo_root)


def dump_config_to_file(config: TrainConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config.to_nested_dict(), f, sort_keys=False)


def serialise_config(config: TrainConfig) -> str:
    return json.dumps(config.to_nested_dict(), indent=2, sort_keys=True)


__all__ = [
    "TrainConfig",
    "load_train_config",
    "apply_overrides",
    "dump_config_to_file",
    "serialise_config",
]


LOG_LEVEL = logging.INFO
logger = logging.getLogger("eval")


def resolve_config_path(name, config_dir, repo_root):
    """Resolve config path from name or explicit path."""
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


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate PPO policies with RLMAPF2 configs.")
    parser.add_argument("--config", default="baseline",
                        help="Config name (relative to configs/train) or explicit path to YAML file.")
    parser.add_argument("--config-dir", default=None,
                        help="Base directory for config lookup when --config is a name.")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="KEY=VALUE", help="Override config values, e.g. --set environment.agents_num=40.")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to the trained PPO checkpoint to evaluate.")
    parser.add_argument("--episodes", type=int, default=1,
                        help="Number of evaluation episodes (default: 1).")
    parser.add_argument("--repeats", type=int, default=10,
                        help="Number of times to repeat evaluation for each agent count (default: 10).")
    parser.add_argument("--agents-range", type=str, default="4-20",
                        help="Range of agents to test, e.g. 4-20.")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable Weights & Biases logging even if the config enables it.")
    parser.add_argument("--num-threads", type=int, default=1,
                        help="Number of parallel threads for evaluation.")
    parser.add_argument("--render-video", action="store_true",
                        help="Generate video recording of evaluation episodes.")
    parser.add_argument("--video-output", type=str, default="eval_videos",
                        help="Directory to save video outputs (default: eval_videos).")
    parser.add_argument("--video-fps", type=int, default=10,
                        help="Frames per second for video output (default: 10).")
    return parser.parse_args(argv)


@dataclass
class EvaluationStats:
    """Track evaluation metrics for each agent count and repeat."""

    time_results: Dict[Tuple[int, int], float] = field(default_factory=dict)
    length_results: Dict[Tuple[int, int], float] = field(default_factory=dict)
    deadlocks: Dict[int, int] = field(default_factory=dict)

    def add_measurement(self, agents_num: int, repeat: int, elapsed: float, episode_len: float) -> None:
        key = (agents_num, repeat)
        self.time_results[key] = elapsed
        self.length_results[key] = episode_len

    def record_deadlock(self, agents_num: int, deadlock_count: int) -> None:
        self.deadlocks.setdefault(agents_num, 0)
        self.deadlocks[agents_num] += deadlock_count

    def averages_for_agent(self, agents_num: int) -> Tuple[float, float]:
        times = [v for (a, _), v in self.time_results.items() if a == agents_num]
        lengths = [v for (a, _), v in self.length_results.items() if a == agents_num]
        avg_time = float(np.mean(times)) if times else 0.0
        avg_length = float(np.mean(lengths)) if lengths else 0.0
        return avg_time, avg_length


class ExperimentSetup:
    """Encapsulate config resolution and reusable experiment state."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.repo_root = Path(__file__).resolve().parent
        self.config_path = resolve_config_path(args.config, args.config_dir, self.repo_root)
        self.config = self._load_config()
        self.agents_range = self._parse_agents_range(args.agents_range)
        self.env_config_base = self._build_environment_config()
        self.ppo_config = self._build_ppo_config()
        self.results_folder = Path(f"eval_results_{time.strftime('%Y%m%d_%H%M%S')}")
        self.results_folder.mkdir(exist_ok=True)
        logger.info(f"Results will be saved to {self.results_folder}")
        self.video_folder = self._prepare_video_folder()

    def _load_config(self):
        config = load_train_config(self.config_path, self.repo_root)
        if self.args.no_wandb:
            config.run.use_wandb = False
        if self.args.overrides:
            config = apply_overrides(config, self.args.overrides, self.repo_root)
        return config

    def _parse_agents_range(self, spec: str) -> range:
        try:
            start, end = map(int, spec.split("-"))
        except ValueError as exc:  # pragma: no cover - user input validation
            raise ValueError("Invalid --agents-range format. Use 'start-end', e.g. 4-20.") from exc
        if start > end:
            raise ValueError("--agents-range start must be <= end")
        return range(start, end + 1)

    def _build_environment_config(self) -> Dict[str, object]:
        env_config = dict(self.config.environment)
        env_config.setdefault("map_path", str(self.config.paths.map_root))
        if not env_config.get("maps_names_with_variants"):
            raise ValueError("Config must specify environment.maps_names_with_variants")
        return env_config

    def _build_ppo_config(self) -> PPOConfig:
        return (
            PPOConfig()
            .api_stack(**self.config.model.api_stack)
            .framework(self.config.model.framework)
            .training(model=dict(self.config.model.model))
            .resources(num_gpus=self.config.hardware.num_gpus, num_cpus=self.config.hardware.num_cpus)
        )

    def _prepare_video_folder(self) -> Optional[Path]:
        if not self.args.render_video:
            return None
        folder = Path(self.args.video_output)
        folder.mkdir(exist_ok=True)
        logger.info(f"Videos will be saved to {folder}")
        return folder

    def create_env_config(self, agents_num: int, seed: int, render_mode: str) -> Dict[str, object]:
        env_config = self.env_config_base.copy()
        env_config["agents_num"] = agents_num
        env_config["seed"] = seed
        env_config["render_mode"] = render_mode
        return env_config

    def build_algorithm(self, env_config: Dict[str, object]):
        algo_config = self.ppo_config.copy(copy_frozen=False)
        algo_config.environment(RLMAPF, env_config=env_config)
        return algo_config.build()


class EvaluationRunner:
    """Coordinate evaluation runs and metrics aggregation."""

    def __init__(self, setup: ExperimentSetup):
        self.setup = setup
        self.stats = EvaluationStats()

    def run(self) -> None:
        self._evaluate_agents()
        if self.setup.args.render_video and self.setup.video_folder:
            VideoRenderer(self.setup).render()
        self._write_summary()

    def _evaluate_agents(self) -> None:
        for agents_num in self.setup.agents_range:
            self.stats.deadlocks[agents_num] = 0
            logger.info(f"Evaluating {agents_num} agents...")
            with ThreadPoolExecutor(max_workers=max(1, self.setup.args.num_threads)) as executor:
                futures = [executor.submit(self._run_single_repeat, agents_num, repeat)
                           for repeat in range(self.setup.args.repeats)]
                for future in as_completed(futures):
                    ag, rep, elapsed, ep_len, dead = future.result()
                    if elapsed is not None and dead == 0:
                        self.stats.add_measurement(ag, rep, elapsed, ep_len)
                    self.stats.record_deadlock(ag, dead)

            self._log_intermediate_results(agents_num)
            self._persist_agent_results(agents_num)

    def _run_single_repeat(self, agents_num: int, repeat: int) -> Tuple[int, int, Optional[float], Optional[float], int]:
        env_config = self.setup.create_env_config(agents_num, seed=42 + repeat, render_mode="none")
        algo = self.setup.build_algorithm(env_config)
        algo.restore(self.setup.args.checkpoint)
        start_time = time.time()
        try:
            results = algo.evaluate()
            elapsed_time = time.time() - start_time
            episode_len = results["env_runners"]["episode_len_mean"]
            deadlock = 0 if episode_len < env_config.get("max_steps", 250) else 1
            return agents_num, repeat, elapsed_time, episode_len, deadlock
        except Exception as exc:  # pragma: no cover - runtime safety
            logger.error(f"Error during evaluation for {agents_num} agents (repeat {repeat}): {exc}")
            return agents_num, repeat, None, None, 1
        finally:
            algo.stop()

    def _persist_agent_results(self, agents_num: int) -> None:
        rows = [
            {"agents_num": a, "repeat": r, "elapsed_time": self.stats.time_results[(a, r)],
             "episode_length": self.stats.length_results[(a, r)]}
            for (a, r) in self.stats.time_results.keys() if a == agents_num
        ]
        df = pd.DataFrame(rows)
        df.to_csv(self.setup.results_folder / f"results_{agents_num}_agents.csv", index=False)

    def _log_intermediate_results(self, agents_num: int) -> None:
        avg_time, avg_len = self.stats.averages_for_agent(agents_num)
        deadlocks = self.stats.deadlocks.get(agents_num, 0)
        logger.info(
            f"Intermediate results for {agents_num} agents: avg time={avg_time:.3f}s, "
            f"avg len={avg_len:.2f}, deadlocks={deadlocks}"
        )

    def _write_summary(self) -> None:
        logger.info("=" * 60)
        logger.info("Final aggregated results:")
        summary_data = []
        for agents_num in self.setup.agents_range:
            avg_time, avg_len = self.stats.averages_for_agent(agents_num)
            deadlocks = self.stats.deadlocks.get(agents_num, 0)
            summary_data.append({
                "agents_num": agents_num,
                "avg_time": avg_time,
                "avg_length": avg_len,
                "deadlocks": deadlocks,
            })
            logger.info(
                f"Agents {agents_num}: avg time={avg_time:.3f}s, avg len={avg_len:.2f}, deadlocks={deadlocks}"
            )

        pd.DataFrame(summary_data).to_csv(self.setup.results_folder / "final_summary.csv", index=False)
        logger.info(f"Saved final summary to {self.setup.results_folder / 'final_summary.csv'}")


class VideoRenderer:
    """Handle optional RGB video generation for evaluation episodes."""

    def __init__(self, setup: ExperimentSetup):
        if setup.video_folder is None:
            raise ValueError("Video folder not configured")
        self.setup = setup

    def render(self) -> None:
        logger.info("=" * 60)
        logger.info("Generating evaluation videos...")
        for agents_num in self.setup.agents_range:
            logger.info(f"Creating video for {agents_num} agents...")
            env_config = self.setup.create_env_config(agents_num, seed=42, render_mode="rgb_array")
            algo = self.setup.build_algorithm(env_config)
            algo.restore(self.setup.args.checkpoint)
            video_filename = f"eval_{agents_num}_agents_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
            video_path = self.setup.video_folder / video_filename
            try:
                frames = self._record_episode(algo, env_config, video_path)
                logger.info(f"Video created: {video_path} ({frames} frames)")
            except Exception as exc:  # pragma: no cover - optional feature
                logger.error(f"Failed to create video for {agents_num} agents: {exc}")
            finally:
                algo.stop()

    def _record_episode(self, algo, env_config: Dict[str, object], video_path: Path) -> int:
        import cv2  # Imported lazily to avoid hard dependency when not rendering videos

        env = RLMAPF(env_config)
        obs, _ = env.reset()
        done = False
        truncated = False
        frames = []

        while not (done or truncated):
            frame = env.render()
            if frame is not None:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            action = algo.compute_single_action(obs)
            obs, _, done, truncated, _ = env.step(action)

        frame = env.render()
        if frame is not None:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        env.close()

        if not frames:
            logger.warning("No frames captured for video")
            return 0

        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(video_path), fourcc, self.setup.args.video_fps, (width, height))
        for frame in frames:
            out.write(frame)
        out.release()
        return len(frames)


def main(argv=None):
    logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args(argv)

    try:
        setup = ExperimentSetup(args)
    except (FileNotFoundError, ValueError) as exc:
        logger.error(str(exc))
        return 1

    logger.info(f"Loaded config: {setup.config_path}")
    logger.info(f"Checkpoint: {args.checkpoint}")

    ray.init(num_cpus=setup.config.hardware.num_cpus, num_gpus=setup.config.hardware.num_gpus)
    try:
        EvaluationRunner(setup).run()
    finally:
        ray.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
