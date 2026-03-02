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
from collections import defaultdict
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union
import re

import numpy as np
import math
import pandas as pd
import ray
try:  # Ray versions prior to 2.7 don't expose this symbol
    from ray._private.utils import RayDeprecationWarning  # type: ignore
except ImportError:  # pragma: no cover - fallback for older Ray releases
    class RayDeprecationWarning(DeprecationWarning):
        pass
import yaml
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.utils.spaces.space_utils import flatten_to_single_ndarray

from rlmapf2 import RLMAPF
from plots import (
    plot_collisions_vs_agents,
    plot_cross_map_comparison,
    plot_deadlock_vs_agents,
    plot_efficiency_vs_agents,
    plot_goal_completion_boxplot,
    plot_goal_completion_heatmap_detailed,
    plot_maps_success_heatmap,
    plot_success_vs_agents,
    plot_throughput_vs_agents,
    plot_makespan_vs_agents,
    plot_dashboard_behavior,
    plot_dashboard_reliability,
    plot_tradeoff_efficiency_vs_collisions,
    plot_goal_completion_heatmap,
    plot_wait_fraction_vs_agents,
    plot_wait_fraction_vs_agents_extra,
    plot_steps_to_half_completion_vs_agents,
    plot_progress_rate_vs_agents,
    plot_collision_diagnostic_hist,
    create_cross_map_summary,
)

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


@dataclass(frozen=True)
class MapSpec:
    name: str
    variants: Optional[List[int]]
    label: str


@dataclass(frozen=True)
class RunInfo:
    path: Path
    name: str
    name_prefix: Optional[str]
    start_time: float


@dataclass(frozen=True)
class CheckpointRecord:
    path: Path
    episode: Optional[int]
    timestamp: float


@dataclass
class EpisodeResult:
    map_name: str
    checkpoint: str
    seed: int
    agents_num: int
    repeat: int
    episode_runtime_seconds: float
    episode_length_steps: int
    goals_completed: int
    total_reward: float
    total_collisions: int
    collision_agent_agent: int
    collision_agent_obstacle: int
    throughput_steps_per_sec: float
    throughput_goals_per_step: float
    wait_actions: int
    wait_fraction: float
    goal_completion_rate_percent: float
    average_steps_to_goal: float
    completion_step_deviation: float
    success: bool
    path_efficiency: float
    collision_agent_agent_per_agent_step: float
    collision_agent_obstacle_per_agent_step: float
    collision_total_per_agent_step: float
    collisions_per_1000_steps: float
    success_episode_length_steps: float
    deadlock: int
    makespan_steps: float
    steps_to_half_completion: float


class EnvConfigBuilder:
    """Build env configs for evaluation rollouts and videos."""

    def __init__(self, config: Dict[str, Any]):
        self._config = config
        self._base_env = dict(config.get('environment', {}))
        self._paths = dict(config.get('paths', {}))

    def build(
        self,
        agents_num: int,
        seed: int,
        render_video: bool,
        video_path: Optional[str],
        render_mode: Optional[str],
        map_spec: Optional[MapSpec],
    ) -> Dict[str, Any]:
        env_config = dict(self._base_env)

        if map_spec is not None:
            env_config['maps_names_with_variants'] = {map_spec.name: map_spec.variants}

        env_config['agents_num'] = agents_num
        env_config['seed'] = seed

        map_root = self._paths.get('map_root', 'maps')
        if not os.path.isabs(map_root):
            map_root = os.path.join(os.getcwd(), map_root)
        env_config['map_path'] = map_root

        if render_video:
            render_mode = "human"
        else:
            render_mode = render_mode if render_mode is not None else env_config.get('render_mode', 'none')
        env_config['render_mode'] = render_mode

        if video_path is None:
            video_path = f"evaluation_video_{agents_num}agents.mp4"

        render_config = dict(env_config.get('render_config', {}))
        render_config.update({
            "show_render": False,
            "save_video": render_video,
            "include_legend": True,
            "legend_position": (0, 0),
            "video_path": video_path,
            "video_fps": self._config.get('eval_video_fps', 10),
            "video_dpi": 300,
            "render_delay": 0.2,
            "title": (
                f"RLMAPF Evaluation - {agents_num} agents"
                + (" (D*)" if env_config.get('use_d_star_lite', False) else "")
            ),
            "save_frames": False,
            "frames_path": "frames/",
            "smooth_motion": self._config.get('eval_smooth_motion', False),
            "motion_frames": self._config.get('eval_motion_frames', 5),
        })
        env_config['render_config'] = render_config
        return env_config


class AlgorithmFactory:
    """Create PPO algorithms for evaluation with consistent model settings."""

    def __init__(self, model_config: Dict[str, Any], num_gpus: float):
        self._model_config = model_config
        self._num_gpus = num_gpus

    def build(self, env_config: Dict[str, Any]) -> Any:
        config = (
            PPOConfig()
            .api_stack(
                enable_rl_module_and_learner=self._model_config['api_stack'].get('enable_rl_module_and_learner', False),
                enable_env_runner_and_connector_v2=self._model_config['api_stack'].get('enable_env_runner_and_connector_v2', False),
            )
            .framework(self._model_config.get('framework', 'torch'))
            .training(model=self._model_config.get('model', {}))
            .environment(RLMAPF, env_config=env_config)
            .env_runners(num_env_runners=0)  # Run rollouts on the driver to avoid extra Ray workers
            .evaluation(evaluation_num_env_runners=0)
            .resources(num_gpus=self._num_gpus)
        )
        return config.build()


class PolicyController:
    """Manage per-agent policy state and action selection."""

    def __init__(self, policy: Any, explore: bool = False):
        self._policy = policy
        self._explore = explore
        self._state_init = policy.get_initial_state()
        self._use_state = len(self._state_init) > 0
        self._action_template = flatten_to_single_ndarray(policy.action_space.sample())
        self._agent_states = defaultdict(self._copy_state_template)
        self._prev_actions = defaultdict(lambda: np.copy(self._action_template))
        self._prev_rewards = defaultdict(float)

    def _copy_state_template(self) -> List[Any]:
        if not self._state_init:
            return []
        return [np.copy(s) if isinstance(s, np.ndarray) else s for s in self._state_init]

    def act(self, agent_id: str, obs: Any) -> int:
        if self._use_state:
            result = self._policy.compute_single_action(
                obs,
                state=self._agent_states[agent_id],
                prev_action=self._prev_actions[agent_id],
                prev_reward=self._prev_rewards[agent_id],
                explore=self._explore,
            )
            if isinstance(result, tuple):
                action = result[0]
                new_state = result[1] if len(result) > 1 else []
            else:
                action = result
                new_state = []
            self._agent_states[agent_id] = new_state
        else:
            result = self._policy.compute_single_action(obs, explore=self._explore)
            action = result[0] if isinstance(result, tuple) else result

        action = flatten_to_single_ndarray(action)
        self._prev_actions[agent_id] = action

        if isinstance(action, (list, np.ndarray)):
            action = np.asarray(action).item()
        return int(action)

    def update_reward(self, agent_id: str, reward: float) -> None:
        self._prev_rewards[agent_id] = reward


class ResultsTracker:
    def __init__(self) -> None:
        self._results: List[EpisodeResult] = []

    def add(self, result: EpisodeResult) -> None:
        self._results.append(result)

    def dataframe(self) -> pd.DataFrame:
        if not self._results:
            return pd.DataFrame()
        return pd.DataFrame([result.__dict__ for result in self._results])

    def agent_dataframe(self, agents_num: int) -> pd.DataFrame:
        df = self.dataframe()
        if df.empty:
            return df
        return df[df['agents_num'] == agents_num].copy()

    def summary_dataframe(self) -> pd.DataFrame:
        df = self.dataframe()
        if df.empty:
            return df
        grouped = df.groupby('agents_num')
        summary = grouped.agg({
            'episode_runtime_seconds': 'mean',
            'episode_length_steps': 'mean',
            'success_episode_length_steps': 'mean',
            'total_reward': 'mean',
            'total_collisions': 'mean',
            'collision_agent_agent': 'mean',
            'collision_agent_obstacle': 'mean',
            'collision_agent_agent_per_agent_step': 'mean',
            'collision_agent_obstacle_per_agent_step': 'mean',
            'collision_total_per_agent_step': 'mean',
            'throughput_steps_per_sec': 'mean',
            'wait_actions': 'mean',
            'goal_completion_rate_percent': 'mean',
            'completion_step_deviation': 'mean',
            'path_efficiency': 'mean',
            'makespan_steps': 'mean',
            'steps_to_half_completion': 'mean',
            'deadlock': 'sum',
            'success': 'mean',
        }).reset_index()

        summary = summary.rename(columns={
            'episode_runtime_seconds': 'avg_time_seconds',
            'episode_length_steps': 'avg_length_steps',
            'success_episode_length_steps': 'avg_success_length_steps',
            'total_reward': 'avg_reward',
            'total_collisions': 'avg_total_collisions',
            'collision_agent_agent': 'avg_collision_agent_agent',
            'collision_agent_obstacle': 'avg_collision_agent_obstacle',
            'collision_agent_agent_per_agent_step': 'avg_collision_agent_agent_per_agent_step',
            'collision_agent_obstacle_per_agent_step': 'avg_collision_agent_obstacle_per_agent_step',
            'collision_total_per_agent_step': 'avg_collision_total_per_agent_step',
            'throughput_steps_per_sec': 'avg_throughput_steps_per_sec',
            'wait_actions': 'avg_wait_actions',
            'goal_completion_rate_percent': 'avg_goal_completion_rate_percent',
            'completion_step_deviation': 'avg_completion_step_deviation',
            'path_efficiency': 'avg_path_efficiency',
            'makespan_steps': 'avg_makespan_steps',
            'steps_to_half_completion': 'avg_steps_to_half_completion',
            'deadlock': 'deadlocks',
            'success': 'success_rate_percent',
        })

        summary['success_rate_percent'] = summary['success_rate_percent'] * 100.0
        summary['deadlock_rate_percent'] = (
            summary['deadlocks'] / grouped.size().values * 100.0
        )
        summary['total_runs'] = grouped.size().values
        return summary


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
        "--checkpoint-group",
        default=None,
        help=(
            "Run directory or run name prefix to group checkpoints (e.g. cnn-20260103-..., or cnn). "
            "When set, numeric --checkpoint selects the Nth most recent run in this group."
        ),
    )
    parser.add_argument(
        "--checkpoint-strategy",
        choices=["best", "latest", "oldest"],
        default=None,
        help="How to choose a checkpoint inside a run group when --checkpoint is numeric.",
    )
    parser.add_argument(
        "--checkpoint-success-tolerance",
        type=float,
        default=None,
        help=(
            "Tolerance in success-rate points when selecting the best checkpoint. "
            "If multiple checkpoints are within this range, pick the oldest."
        ),
    )
    parser.add_argument(
        "--sync-train-config",
        action="store_true",
        help=(
            "Load the training config stored next to the checkpoint and overwrite "
            "evaluation environment/model settings to match training."
        ),
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
        "--success-threshold",
        type=float,
        default=None,
        help="Goal completion rate (%%) required to mark an episode as success (default: 99.9).",
    )
    parser.add_argument(
        "--clamp-agents-to-train",
        action="store_true",
        help="Override eval agent range to the training agents_num found in the checkpoint config.",
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
    parser.add_argument(
        "--video-once-per-agent",
        action="store_true",
        help="Render at most one video per agent count across repeats (default: render each repeat).",
    )
    parser.add_argument(
        "--video-best-repeat-per-agent",
        action="store_true",
        help="Render only the best repeat for each agent count (implies one video per agent count).",
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

        current = config
        for key in keys[:-1]:
            # Ensure nested containers exist (and reset non-dicts like None) so we can assign deeper keys.
            if key not in current or current[key] is None or not isinstance(current[key], dict):
                current[key] = {}
            current = current[key]

        final_key = keys[-1]
        try:
            lower_val = value.lower()
            if lower_val in ("none", "null"):
                current[final_key] = None
            elif lower_val in ("true", "false"):
                current[final_key] = lower_val == "true"
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
        if "-" in range_str:
            start, end = map(int, range_str.split('-'))
            return range(start, end + 1)
        val = int(range_str)
        return range(val, val + 1)
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


def _parse_run_start_time(run_dir: Path) -> float:
    metadata_path = run_dir / "run_metadata.json"
    if metadata_path.exists():
        try:
            data = json.loads(metadata_path.read_text(encoding="utf-8"))
            start_time = data.get("start_time")
            if isinstance(start_time, str):
                return time.mktime(time.strptime(start_time, "%Y-%m-%dT%H:%M:%S"))
        except Exception:
            pass
    try:
        return run_dir.stat().st_mtime
    except FileNotFoundError:
        return 0.0


def _load_run_name_prefix(run_dir: Path) -> Optional[str]:
    config_path = run_dir / "config" / "resolved_config.yaml"
    if not config_path.exists():
        return None
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
        return config.get("run", {}).get("name_prefix")
    except Exception:
        return None


def _discover_run_infos(roots: Iterable[Union[str, Path]]) -> List[RunInfo]:
    run_dirs: List[Path] = []
    for root in roots:
        root_path = Path(root)
        if not root_path.exists():
            continue
        for checkpoints_index in root_path.rglob("checkpoints.jsonl"):
            run_dirs.append(checkpoints_index.parent)

    seen = set()
    unique_run_dirs: List[Path] = []
    for run_dir in run_dirs:
        resolved = run_dir.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique_run_dirs.append(run_dir)

    run_infos: List[RunInfo] = []
    for run_dir in unique_run_dirs:
        run_infos.append(
            RunInfo(
                path=run_dir,
                name=run_dir.name,
                name_prefix=_load_run_name_prefix(run_dir),
                start_time=_parse_run_start_time(run_dir),
            )
        )
    return run_infos


def _infer_group_name(run_info: RunInfo) -> str:
    if run_info.name_prefix:
        return run_info.name_prefix
    match = re.match(r"^(.+)-\d{8}-\d{6}-\d{4}$", run_info.name)
    if match:
        return match.group(1)
    return run_info.name


def _filter_run_infos(
    run_infos: List[RunInfo],
    group_hint: Optional[str],
) -> List[RunInfo]:
    if not group_hint:
        return run_infos

    group_path = Path(group_hint)
    if group_path.exists():
        if group_path.name == "checkpoints":
            run_dir = group_path.parent
        else:
            run_dir = group_path
        for info in run_infos:
            if info.path.resolve() == run_dir.resolve():
                return [info]
        if (run_dir / "checkpoints.jsonl").exists():
            return [
                RunInfo(
                    path=run_dir,
                    name=run_dir.name,
                    name_prefix=_load_run_name_prefix(run_dir),
                    start_time=_parse_run_start_time(run_dir),
                )
            ]

    filtered = [
        info for info in run_infos
        if info.name.startswith(group_hint) or info.name_prefix == group_hint
    ]
    return filtered


def _format_checkpoint_label(record: CheckpointRecord) -> str:
    if record.episode is not None:
        return f"ep{record.episode}"
    name = record.path.name.lower()
    if "final" in name:
        return "final"
    return record.path.name


def _build_checkpoint_group_listing(run_infos: List[RunInfo]) -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for run_info in run_infos:
        group_name = _infer_group_name(run_info)
        records = _load_checkpoint_records(run_info.path)
        metrics_entries = _load_metrics_entries(run_info.path)
        checkpoints = []
        for record in sorted(records, key=lambda r: (r.episode if r.episode is not None else float("inf"), r.timestamp)):
            entry = _find_metrics_at_or_before(metrics_entries, record.episode)
            checkpoints.append({
                "label": _format_checkpoint_label(record),
                "path": str(record.path),
                "episode": record.episode,
                "timestamp": record.timestamp,
                "success_rate_pct": _extract_success_rate(entry or {}),
            })
        groups.setdefault(group_name, []).append({
            "run_name": run_info.name,
            "run_path": str(run_info.path),
            "start_time": run_info.start_time,
            "checkpoints": checkpoints,
        })

    for group_runs in groups.values():
        group_runs.sort(key=lambda item: item.get("start_time", 0.0), reverse=True)
    return groups


def _load_checkpoint_records(run_dir: Path) -> List[CheckpointRecord]:
    records: List[CheckpointRecord] = []
    index_path = run_dir / "checkpoints.jsonl"
    if index_path.exists():
        with index_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                path = Path(entry.get("path", "")).expanduser()
                if not path.is_absolute():
                    path = (run_dir / path).resolve()
                if not path.exists():
                    candidate = run_dir / "checkpoints" / path.name
                    if candidate.exists():
                        path = candidate.resolve()
                    else:
                        continue
                records.append(
                    CheckpointRecord(
                        path=path,
                        episode=entry.get("episode"),
                        timestamp=float(entry.get("timestamp", 0.0)),
                    )
                )

    if records:
        return records

    checkpoints_dir = run_dir / "checkpoints"
    if checkpoints_dir.exists():
        episode_pattern = re.compile(r"_ep(\d+)")
        for ckpt_state in checkpoints_dir.rglob("algorithm_state.pkl"):
            episode = None
            match = episode_pattern.search(ckpt_state.parent.name)
            if match:
                try:
                    episode = int(match.group(1))
                except ValueError:
                    episode = None
            records.append(
                CheckpointRecord(
                    path=ckpt_state.parent,
                    episode=episode,
                    timestamp=ckpt_state.parent.stat().st_mtime,
                )
            )
    return records


def _load_metrics_entries(run_dir: Path) -> List[Dict[str, Any]]:
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.exists():
        return []
    entries: List[Dict[str, Any]] = []
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "episode" in entry and "metrics" in entry:
                entries.append(entry)
    entries.sort(key=lambda e: e.get("episode", 0))
    return entries


def _find_metrics_at_or_before(
    entries: List[Dict[str, Any]],
    episode: Optional[int],
) -> Optional[Dict[str, Any]]:
    if not entries:
        return None
    if episode is None:
        return entries[-1]
    last = None
    for entry in entries:
        entry_episode = entry.get("episode")
        if entry_episode is None:
            continue
        if entry_episode <= episode:
            last = entry
        else:
            break
    return last


def _extract_success_rate(metrics: Dict[str, Any]) -> Optional[float]:
    if not metrics:
        return None
    values = metrics.get("metrics", metrics)
    if not isinstance(values, dict):
        return None
    keys = [
        "evaluation/env_runners/custom_metrics/success_rate_pct_mean",
        "evaluation/custom_metrics/success_rate_pct_mean",
        "env_runners/custom_metrics/success_rate_pct_mean",
        "custom_metrics/success_rate_pct_mean",
    ]
    for key in keys:
        value = values.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _select_checkpoint_by_strategy(
    records: List[CheckpointRecord],
    metrics_entries: List[Dict[str, Any]],
    strategy: str,
    success_tolerance: float,
) -> Path:
    if not records:
        raise FileNotFoundError("No checkpoints found for selected run.")

    if strategy == "latest":
        record = max(records, key=lambda r: (r.timestamp, r.episode or -1))
        return record.path

    if strategy == "oldest":
        record = min(records, key=lambda r: (r.episode if r.episode is not None else float("inf"), r.timestamp))
        return record.path

    scored: List[Tuple[CheckpointRecord, Optional[float]]] = []
    for record in records:
        entry = _find_metrics_at_or_before(metrics_entries, record.episode)
        scored.append((record, _extract_success_rate(entry or {})))

    valid_scores = [score for _, score in scored if score is not None]
    if not valid_scores:
        record = max(records, key=lambda r: (r.timestamp, r.episode or -1))
        logger.warning("No success metrics found; falling back to latest checkpoint %s", record.path)
        return record.path

    best_score = max(valid_scores)
    candidates = [
        record for record, score in scored
        if score is not None and score >= best_score - success_tolerance
    ]
    if not candidates:
        record = max(records, key=lambda r: (r.timestamp, r.episode or -1))
        return record.path

    record = min(candidates, key=lambda r: (r.episode if r.episode is not None else float("inf"), r.timestamp))
    return record.path


def resolve_checkpoint_path(
    checkpoint_arg: str,
    experiments_roots: Union[str, Path, Iterable[Union[str, Path]]] = ("experiments/train", "experiments"),
    group_hint: Optional[str] = None,
    strategy: str = "best",
    success_tolerance: float = 1.0,
) -> str:
    """
    Resolve checkpoint argument to full path.

    If checkpoint_arg is a digit (1, 2, 3...), selects the Nth most recent run
    (after applying group filtering) and then chooses a checkpoint within that run
    based on the provided strategy.
    Otherwise, returns the checkpoint path as-is.
    """
    arg_str = str(checkpoint_arg).strip()

    if isinstance(experiments_roots, (str, Path)):
        roots: List[Union[str, Path]] = [experiments_roots]
    else:
        roots = list(experiments_roots)
    if not roots:
        roots = ["experiments/train", "experiments"]

    if arg_str.isdigit():
        n = int(arg_str)
        run_infos = _discover_run_infos(roots)
        filtered_runs = _filter_run_infos(run_infos, group_hint)
        if not filtered_runs:
            filtered_runs = run_infos

        if not filtered_runs:
            raise FileNotFoundError(
                "No run directories found in any search roots: " + ", ".join(str(r) for r in roots)
            )

        filtered_runs.sort(key=lambda info: info.start_time, reverse=True)

        if n > len(filtered_runs):
            raise ValueError(f"Requested run #{n} but only {len(filtered_runs)} runs found")

        selected_run = filtered_runs[n - 1]
        logger.info("Selected run: %s", selected_run.path)

        records = _load_checkpoint_records(selected_run.path)
        metrics_entries = _load_metrics_entries(selected_run.path)
        selected_checkpoint = _select_checkpoint_by_strategy(
            records,
            metrics_entries,
            strategy=strategy,
            success_tolerance=success_tolerance,
        )
        logger.info("Checkpoint strategy '%s' resolved to: %s", strategy, selected_checkpoint)
        return str(selected_checkpoint)

    path = Path(arg_str).expanduser()
    if not path.is_absolute():
        path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint path not found: {path}")
    return str(path)


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

    config_file = experiment_dir / "resolved_config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def _get_by_path(config: Dict[str, Any], path: Iterable[str]) -> Any:
    current: Any = config
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _load_training_config_from_checkpoint(checkpoint_path: Path) -> Optional[Dict[str, Any]]:
    run_dir = checkpoint_path.parent.parent
    resolved_config_path = run_dir / "config" / "resolved_config.yaml"
    if not resolved_config_path.exists():
        return None
    try:
        with resolved_config_path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    except Exception:
        logger.debug("Failed to read training config from %s", resolved_config_path, exc_info=True)
        return None


def _warn_config_mismatches(eval_config: Dict[str, Any], train_config: Dict[str, Any]) -> None:
    checks = [
        ("environment.use_cnn_observation", ["environment", "use_cnn_observation"]),
        ("environment.use_d_star_lite", ["environment", "use_d_star_lite"]),
        ("environment.observation_size", ["environment", "observation_size"]),
        ("environment.observation_type", ["environment", "observation_type"]),
        ("environment.start_goal_on_periphery", ["environment", "start_goal_on_periphery"]),
        ("environment.d_star_iterations", ["environment", "d_star_iterations"]),
        ("environment.d_star_congestion_weight", ["environment", "d_star_congestion_weight"]),
        ("environment.d_star_path_progress_weight", ["environment", "d_star_path_progress_weight"]),
        ("model.model.conv_filters", ["model", "model", "conv_filters"]),
        ("model.model.fcnet_hiddens", ["model", "model", "fcnet_hiddens"]),
        ("model.model.use_lstm", ["model", "model", "use_lstm"]),
    ]
    mismatches = []
    for label, path in checks:
        eval_value = _get_by_path(eval_config, path)
        train_value = _get_by_path(train_config, path)
        if train_value is None:
            continue
        if eval_value != train_value:
            mismatches.append((label, train_value, eval_value))

    eval_maps = _get_by_path(eval_config, ["environment", "maps_names_with_variants"])
    train_maps = _get_by_path(train_config, ["environment", "maps_names_with_variants"])
    if isinstance(eval_maps, dict) and isinstance(train_maps, dict):
        eval_keys = set(eval_maps.keys())
        train_keys = set(train_maps.keys())
        if eval_keys != train_keys:
            mismatches.append(("environment.maps_names_with_variants", train_keys, eval_keys))

    if mismatches:
        logger.warning("Evaluation config does not match training config from checkpoint:")
        for label, train_value, eval_value in mismatches:
            logger.warning("  %s: train=%s eval=%s", label, train_value, eval_value)


def sync_eval_config_with_train_config(
    eval_config: Dict[str, Any],
    train_config: Dict[str, Any],
) -> Dict[str, Any]:
    """Override eval config with environment/model settings from training config."""
    merged = deepcopy(eval_config)

    train_env = deepcopy(train_config.get("environment") or {})
    eval_env_overrides = train_config.get("evaluation_environment") or {}
    if isinstance(eval_env_overrides, dict):
        train_env.update(eval_env_overrides)
    if train_env:
        merged["environment"] = train_env

    train_model = deepcopy(train_config.get("model") or {})
    if train_model:
        merged["model"] = train_model

    train_paths = train_config.get("paths") or {}
    if isinstance(train_paths, dict) and train_paths.get("map_root") is not None:
        merged.setdefault("paths", {})
        merged["paths"]["map_root"] = train_paths["map_root"]

    return merged



class Evaluator:
    def __init__(self, args: argparse.Namespace):
        self._args = args
        self._repo_root = Path(__file__).resolve().parent
        self._config_path = resolve_config_path(args.config, args.config_dir, self._repo_root)
        self._config = load_config(self._config_path)
        self._paths_config = self._config.get('paths', {})

        checkpoint_search_roots = self._checkpoint_search_roots()
        group_hint = args.checkpoint_group or self._config.get('eval_checkpoint_group')
        if group_hint is None:
            group_hint = self._config.get('run', {}).get('name_prefix')
        checkpoint_strategy = args.checkpoint_strategy or self._config.get('eval_checkpoint_strategy', 'best')
        success_tolerance = args.checkpoint_success_tolerance
        if success_tolerance is None:
            success_tolerance = float(self._config.get('eval_checkpoint_success_tolerance', 1.0))

        self._checkpoint_group_hint = group_hint
        self._checkpoint_strategy = checkpoint_strategy
        self._checkpoint_success_tolerance = success_tolerance

        resolved_checkpoint = resolve_checkpoint_path(
            args.checkpoint,
            checkpoint_search_roots,
            group_hint=self._checkpoint_group_hint,
            strategy=self._checkpoint_strategy,
            success_tolerance=self._checkpoint_success_tolerance,
        )
        self._checkpoint_path = Path(resolved_checkpoint).resolve()

        self._train_config: Optional[Dict[str, Any]] = None
        auto_sync_enabled = bool(self._config.get('eval_sync_train_config', False))
        if args.sync_train_config or auto_sync_enabled:
            self._train_config = _load_training_config_from_checkpoint(self._checkpoint_path)
            if self._train_config is None:
                raise ValueError(
                    "Unable to locate training config next to checkpoint. "
                    "Expected config/resolved_config.yaml in the training run directory."
                )
            self._config = sync_eval_config_with_train_config(self._config, self._train_config)
            train_config_path = self._checkpoint_path.parent.parent / "config" / "resolved_config.yaml"
            logger.info(
                "Synced eval config with training config at %s (%s)",
                train_config_path,
                "auto" if auto_sync_enabled and not args.sync_train_config else "requested",
            )
        if self._train_config is None:
            self._train_config = _load_training_config_from_checkpoint(self._checkpoint_path)

        if args.overrides:
            logger.info("Applying %d override(s)", len(args.overrides))
            self._config = apply_overrides(self._config, args.overrides)

        self._video_agent_selection = parse_video_agent_selection(args.video_agents)
        self._run_name = build_run_name(self._config, args.run_name)
        self._results_dir = self._resolve_results_dir(args.results_dir)

        self._hardware_config = self._config.get('hardware', {})
        self._model_config = self._config.get('model', {})
        self._env_config = self._config.get('environment', {})
        self._paths_config = self._config.get('paths', {})
        self._success_threshold = (
            args.success_threshold
            if args.success_threshold is not None
            else float(self._config.get('eval_success_threshold_percent', 99.9))
        )

        self._num_cpus = self._hardware_config.get('num_cpus', 2)
        self._num_gpus = self._hardware_config.get('num_gpus', 0)

        self._agents_range = parse_agent_range(self._config.get('eval_agents_range', '4-20'))
        if args.clamp_agents_to_train and self._train_config:
            train_agents = self._train_config.get('environment', {}).get('agents_num')
            if train_agents is not None:
                self._agents_range = range(int(train_agents), int(train_agents) + 1)
                logger.info("Clamped agent range to training agents_num=%s", train_agents)
            else:
                logger.warning("Could not clamp agents to training config; agents_num missing in training config.")
        self._num_threads = args.num_threads if args.num_threads is not None else self._config.get('eval_num_threads', 1)
        self._repeats = args.repeats if args.repeats is not None else self._config.get('eval_repeats', 10)
        self._render_video = args.render_video
        if self._render_video and args.repeats is None:
            self._repeats = 1
            logger.info("Video rendering enabled - setting repeats to 1 (override with --repeats if needed)")
        self._video_once_per_agent = bool(
            args.video_once_per_agent
            or self._config.get('eval_video_once_per_agent', False)
        )
        self._video_best_repeat_per_agent = bool(
            args.video_best_repeat_per_agent
            or self._config.get('eval_video_best_repeat_per_agent', False)
        )
        if self._video_best_repeat_per_agent:
            self._video_once_per_agent = True

        self._env_builder = EnvConfigBuilder(self._config)
        self._algo_factory = AlgorithmFactory(self._model_config, self._num_gpus)
        self._map_specs, self._multi_map_enabled = self._resolve_map_specs()

        self._warn_if_config_mismatch()

    def _resolve_results_dir(self, results_dir: Optional[str]) -> Path:
        if results_dir:
            return Path(results_dir).resolve()
        results_base = Path(self._config.get('paths', {}).get('experiments_root', 'experiments'))
        name_prefix = self._config.get('run', {}).get('name_prefix')
        if name_prefix:
            return (results_base / name_prefix / self._run_name).resolve()
        return (results_base / self._run_name).resolve()

    def _checkpoint_search_roots(self) -> List[Union[str, Path]]:
        experiments_root = self._paths_config.get('experiments_root')
        train_experiments_root = self._paths_config.get('train_experiments_root', 'experiments/train')
        roots: List[Union[str, Path]] = []
        if experiments_root:
            roots.append(experiments_root)
        if train_experiments_root and all(Path(train_experiments_root) != Path(existing) for existing in roots):
            roots.append(train_experiments_root)
        if all(Path('experiments') != Path(existing) for existing in roots):
            roots.append('experiments')
        return roots

    def _resolve_map_specs(self) -> Tuple[List[MapSpec], bool]:
        eval_maps_config = self._config.get('eval_maps', {})
        multi_map_enabled = eval_maps_config.get('enabled', False)
        if multi_map_enabled:
            maps_to_evaluate = eval_maps_config.get('maps', [])
            map_specs = []
            for item in maps_to_evaluate:
                map_specs.append(
                    MapSpec(
                        name=item['name'],
                        variants=item.get('variants'),
                        label=item.get('label', item['name']),
                    )
                )
            return map_specs, True

        current_map = list(self._env_config.get('maps_names_with_variants', {}).keys())[0]
        map_spec = MapSpec(
            name=current_map,
            variants=self._env_config.get('maps_names_with_variants', {}).get(current_map),
            label=current_map,
        )
        return [map_spec], False

    def _warn_if_config_mismatch(self) -> None:
        train_config = self._train_config or _load_training_config_from_checkpoint(self._checkpoint_path)
        if train_config is None:
            return
        _warn_config_mismatches(self._config, train_config)

    def run(self) -> int:
        self._results_dir.mkdir(parents=True, exist_ok=True)

        git_info = collect_git_info(self._repo_root) if self._config.get('logging', {}).get('log_git_info', True) else {}
        save_metadata(self._results_dir, self._config, self._config_path, str(self._checkpoint_path), self._run_name, git_info)

        self._log_checkpoint_groups()
        self._log_header()

        return_code = 0
        map_summary_paths: Dict[str, Path] = {}

        try:
            ray_context = ray.init(
                num_cpus=self._num_cpus,
                num_gpus=self._num_gpus,
            )
            logger.info("Ray initialized (dashboard: %s)", ray_context.dashboard_url)

            for map_idx, map_spec in enumerate(self._map_specs):
                if self._multi_map_enabled:
                    logger.info("")
                    logger.info("=" * 80)
                    logger.info("EVALUATING MAP %d/%d: %s", map_idx + 1, len(self._map_specs), map_spec.label)
                    logger.info("Map: %s", map_spec.name)
                    logger.info("=" * 80)

                if self._multi_map_enabled:
                    current_results_dir = self._results_dir / f"map_{map_spec.label.replace(' ', '_').lower()}"
                    current_results_dir.mkdir(parents=True, exist_ok=True)
                    logger.info("Map results directory: %s", current_results_dir)
                else:
                    current_results_dir = self._results_dir

                summary_path = self._evaluate_map(map_spec, current_results_dir)
                if self._multi_map_enabled and summary_path is not None:
                    map_summary_paths[map_spec.label] = summary_path

            if self._multi_map_enabled and len(map_summary_paths) > 1:
                self._run_cross_map_comparison(map_summary_paths)

        except KeyboardInterrupt:
            logger.warning("Interrupted by user.")
            return_code = 130
        except Exception as exc:
            logger.exception("Evaluation failed due to an unexpected error: %s", exc)
            return_code = 1
        finally:
            ray.shutdown()

        return return_code

    def _log_checkpoint_groups(self) -> None:
        checkpoint_search_roots = self._checkpoint_search_roots()
        run_infos = _discover_run_infos(checkpoint_search_roots)
        if not run_infos:
            logger.warning("No checkpoint runs discovered in %s", checkpoint_search_roots)
            return

        groups = _build_checkpoint_group_listing(run_infos)
        groups_path = self._results_dir / "checkpoint_groups.json"
        try:
            groups_path.write_text(json.dumps(groups, indent=2, sort_keys=True), encoding="utf-8")
            logger.info("Checkpoint groups saved to %s", groups_path)
        except Exception:
            logger.debug("Failed to write checkpoint group listing to %s", groups_path, exc_info=True)

        logger.info("Available checkpoint groups:")
        for group_name, runs in sorted(groups.items()):
            logger.info("  Group: %s (%d run(s))", group_name, len(runs))
            for run in runs:
                checkpoints = run.get("checkpoints", [])
                if not checkpoints:
                    logger.info("    %s -> no checkpoints", run.get("run_name"))
                    continue
                labels = []
                for checkpoint in checkpoints:
                    label = checkpoint.get("label", "?")
                    success = checkpoint.get("success_rate_pct")
                    if success is not None:
                        label = f"{label}:{success:.2f}%"
                    labels.append(label)
                logger.info("    %s -> %s", run.get("run_name"), ", ".join(labels))

    def _log_header(self) -> None:
        logger.info("=" * 80)
        logger.info("Evaluation Configuration")
        logger.info("=" * 80)
        logger.info("Run name: %s", self._run_name)
        logger.info("Results directory: %s", self._results_dir)
        logger.info("Checkpoint: %s", self._checkpoint_path)
        if self._args.checkpoint.isdigit():
            logger.info(
                "Checkpoint selection: group=%s strategy=%s tolerance=%.3f",
                self._checkpoint_group_hint or "auto",
                self._checkpoint_strategy,
                self._checkpoint_success_tolerance,
            )
        logger.info("Agent range: %s", list(self._agents_range))
        logger.info("Threads: %d, Repeats: %d", self._num_threads, self._repeats)
        logger.info("CPUs: %d, GPUs: %d", self._num_cpus, self._num_gpus)
        logger.info("Success threshold: %.2f%%", self._success_threshold)
        if self._render_video:
            if self._video_agent_selection:
                selection_str = ", ".join(str(num) for num in sorted(self._video_agent_selection))
                scope_note = "only for agents: %s" % selection_str
            else:
                scope_note = "for all agent counts"
            if self._video_best_repeat_per_agent:
                logger.info("Video rendering: ENABLED (%s; one video per agent count, best repeat)", scope_note)
            elif self._video_once_per_agent:
                logger.info("Video rendering: ENABLED (%s; one video per agent count)", scope_note)
            else:
                logger.info("Video rendering: ENABLED (%s)", scope_note)
        else:
            logger.info("Video rendering: DISABLED")
        logger.info("=" * 80)

    def _evaluate_map(self, map_spec: MapSpec, results_dir: Path) -> Optional[Path]:
        results_tracker = ResultsTracker()

        for agents_num in self._agents_range:
            logger.info("")
            logger.info("=" * 80)
            logger.info("Evaluating with %d agents", agents_num)
            logger.info("=" * 80)

            repeat_results = self._run_repeats_for_agent_count(map_spec, agents_num, results_dir)
            for result in repeat_results:
                results_tracker.add(result)

            self._write_intermediate_results(results_tracker, agents_num, results_dir)

        summary_df = results_tracker.summary_dataframe()
        if summary_df.empty:
            logger.warning("No results collected for map %s", map_spec.label)
            return None

        self._write_final_results(results_tracker, summary_df, results_dir)
        self._generate_plots(results_dir)

        logger.info("")
        logger.info("=" * 80)
        logger.info("Evaluation completed successfully!")
        logger.info("Results saved to: %s", results_dir)
        if self._render_video:
            logger.info("Videos saved to: %s/*.mp4", results_dir)
        logger.info("=" * 80)

        return results_dir / 'summary.csv'

    def _run_repeats_for_agent_count(self, map_spec: MapSpec, agents_num: int, results_dir: Path) -> List[EpisodeResult]:
        results: List[EpisodeResult] = []
        render_during_repeats = self._render_video and not self._video_best_repeat_per_agent
        video_repeat = None
        if render_during_repeats:
            selection_allows_video = self._video_agent_selection is None or agents_num in self._video_agent_selection
            if selection_allows_video and self._video_once_per_agent:
                video_repeat = 0  # render using the first repeat for this agent count
        with ThreadPoolExecutor(max_workers=self._num_threads) as executor:
            futures = [
                executor.submit(
                    self._run_repeat,
                    map_spec,
                    agents_num,
                    repeat,
                    results_dir,
                    video_repeat,
                    disable_video=not render_during_repeats,
                )
                for repeat in range(self._repeats)
            ]
            for future in as_completed(futures):
                results.append(future.result())
        if self._render_video and self._video_best_repeat_per_agent:
            best_repeat = self._select_best_repeat_for_agent(results, agents_num)
            if best_repeat is not None:
                self._render_best_repeat_video(map_spec, agents_num, best_repeat, results_dir)
        return results

    def _run_repeat(
        self,
        map_spec: MapSpec,
        agents_num: int,
        repeat: int,
        results_dir: Path,
        video_repeat: Optional[int] = None,
        disable_video: bool = False,
    ) -> EpisodeResult:
        logger.info("Evaluating with %d agents, repeat %d/%d", agents_num, repeat + 1, self._repeats)

        should_render = False
        if not disable_video:
            should_render = self._render_video and (
                self._video_agent_selection is None or agents_num in self._video_agent_selection
            )
            if should_render and self._video_once_per_agent:
                should_render = video_repeat is not None and repeat == video_repeat

        video_path = None
        if should_render:
            video_filename = f"evaluation_{agents_num}agents_repeat{repeat}.mp4"
            video_path = str(results_dir / video_filename)
            logger.info("Rendering video to: %s", video_path)

        env_config = self._env_builder.build(
            agents_num=agents_num,
            seed=42 + repeat,
            render_video=False,
            video_path=None,
            render_mode="none",
            map_spec=map_spec,
        )

        algorithm = self._algo_factory.build(env_config)
        algorithm.restore(str(self._checkpoint_path))
        policy = algorithm.get_policy()

        try:
            if should_render and video_path is not None:
                self._render_video_episode(policy, map_spec, agents_num, repeat, video_path)

            result = self._rollout_episode(policy, map_spec, agents_num, repeat)
        finally:
            algorithm.stop()

        return result

    def _select_best_repeat_for_agent(self, results: List[EpisodeResult], agents_num: int) -> Optional[int]:
        agent_results = [r for r in results if r.agents_num == agents_num]
        if not agent_results:
            return None

        def safe(val: Any, default: float = 0.0) -> float:
            if val is None:
                return default
            try:
                if isinstance(val, float) and math.isnan(val):
                    return default
                return float(val)
            except (TypeError, ValueError):
                return default

        def score(result: EpisodeResult) -> Tuple:
            return (
                1 if result.success else 0,  # prefer successful episodes
                safe(result.goal_completion_rate_percent),  # higher completion
                -safe(result.collision_total_per_agent_step),  # fewer collisions
                -safe(result.wait_fraction),  # less waiting
                safe(result.throughput_goals_per_step),  # higher throughput
                -safe(result.episode_length_steps),  # shorter episodes
            )

        best = max(agent_results, key=score)
        logger.info(
            "Selected best repeat for %d agents: repeat %d (score=%s)",
            agents_num,
            best.repeat,
            score(best),
        )
        return int(best.repeat)

    def _render_best_repeat_video(self, map_spec: MapSpec, agents_num: int, repeat: int, results_dir: Path) -> None:
        logger.info("Rendering best repeat video for %d agents using repeat %d", agents_num, repeat)
        video_filename = f"evaluation_{agents_num}agents_repeat{repeat}.mp4"
        video_path = str(results_dir / video_filename)

        env_config = self._env_builder.build(
            agents_num=agents_num,
            seed=42 + repeat,
            render_video=False,
            video_path=None,
            render_mode="none",
            map_spec=map_spec,
        )

        algorithm = self._algo_factory.build(env_config)
        algorithm.restore(str(self._checkpoint_path))
        policy = algorithm.get_policy()
        try:
            self._render_video_episode(policy, map_spec, agents_num, repeat, video_path)
        finally:
            algorithm.stop()

    def _render_video_episode(
        self,
        policy: Any,
        map_spec: MapSpec,
        agents_num: int,
        repeat: int,
        video_path: str,
    ) -> None:
        env_config = self._env_builder.build(
            agents_num=agents_num,
            seed=42 + repeat,
            render_video=True,
            video_path=video_path,
            render_mode=None,
            map_spec=map_spec,
        )

        env = RLMAPF(env_config)
        controller = PolicyController(policy, explore=False)

        try:
            obs, _ = env.reset()
            step_count = 0
            max_steps = env_config.get('max_steps', env.max_steps)
            while step_count < max_steps:
                actions = {agent_id: controller.act(agent_id, agent_obs) for agent_id, agent_obs in obs.items()}
                obs, rewards, dones, truncated, _ = env.step(actions)
                if isinstance(rewards, dict):
                    for agent_id, reward in rewards.items():
                        controller.update_reward(agent_id, reward if reward is not None else 0.0)
                step_count += 1
                if dones.get('__all__', False) or truncated.get('__all__', False):
                    break
            env.finalize_video()
            logger.info("Video saved to: %s", video_path)
        except Exception as exc:
            logger.warning("Video rendering failed: %s", exc)
        finally:
            env.close()

    def _rollout_episode(
        self,
        policy: Any,
        map_spec: MapSpec,
        agents_num: int,
        repeat: int,
    ) -> EpisodeResult:
        env_config = self._env_builder.build(
            agents_num=agents_num,
            seed=42 + repeat,
            render_video=False,
            video_path=None,
            render_mode="none",
            map_spec=map_spec,
        )

        env = RLMAPF(env_config)
        controller = PolicyController(policy, explore=False)

        start_time = time.time()
        obs, info = env.reset()
        step_count = 0
        max_steps = env_config.get('max_steps', env.max_steps)
        done = False

        agents_reached_goal: Set[str] = set()
        agent_completion_steps: Dict[str, int] = {}
        total_reward = 0.0
        wait_actions = 0

        half_completion_step: Optional[int] = None
        agent_agent_collision_total = 0
        agent_obstacle_collision_total = 0

        total_agents = env_config.get('agents_num', agents_num)
        half_threshold = max(1, int(np.ceil(total_agents * 0.5)))
        agent_positions_dict = getattr(env, "agent_positions", {})
        agent_ids_for_counters = [
            agent_id for agent_id in agent_positions_dict.keys()
            if agent_id != "__all__" and agent_id != "__common__"
        ]
        collision_counters = {
            agent_id: info.get(agent_id, {}).get("number_of_collisions", 0)
            for agent_id in agent_ids_for_counters
        }

        supports_positions = hasattr(env, "agent_positions")
        prev_positions: Dict[str, Tuple[int, int]] = {}
        if supports_positions:
            for agent_id in agent_ids_for_counters:
                pos = env.agent_positions.get(agent_id)
                if pos is not None:
                    prev_positions[agent_id] = tuple(pos)

        while not done and step_count < max_steps:
            if supports_positions:
                for agent_id in agent_ids_for_counters:
                    pos = env.agent_positions.get(agent_id)
                    if pos is not None:
                        prev_positions[agent_id] = tuple(pos)

            actions = {}
            intended_positions = {}
            for agent_id, agent_obs in obs.items():
                action = controller.act(agent_id, agent_obs)
                actions[agent_id] = action
                delta = ACTION_DELTAS.get(action, (0, 0))
                prev = prev_positions.get(agent_id, (0, 0))
                intended_positions[agent_id] = (prev[0] + delta[0], prev[1] + delta[1])

            wait_actions += sum(1 for action in actions.values() if action == 4)

            obs, rewards, dones, truncated, info = env.step(actions)
            total_reward += sum(rewards.values())
            for agent_id, reward in rewards.items():
                controller.update_reward(agent_id, reward if reward is not None else 0.0)

            for agent_id, is_done in dones.items():
                if agent_id == "__all__":
                    continue
                if is_done and agent_id not in agents_reached_goal:
                    agents_reached_goal.add(agent_id)
                    agent_completion_steps[agent_id] = step_count
                    if half_completion_step is None and len(agents_reached_goal) >= half_threshold:
                        half_completion_step = step_count

            collision_deltas = {}
            info_dict = info if isinstance(info, dict) else {}
            if collision_counters and info_dict:
                for agent_id, agent_info in info_dict.items():
                    if agent_id not in collision_counters:
                        continue
                    prev_count = collision_counters[agent_id]
                    new_count = agent_info.get("number_of_collisions", prev_count)
                    delta = new_count - prev_count
                    if delta > 0:
                        collision_deltas[agent_id] = delta
                    collision_counters[agent_id] = new_count

            if collision_deltas:
                collision_agents = list(collision_deltas.keys())
                intended_subset = {agent_id: intended_positions.get(agent_id) for agent_id in collision_agents}
                prev_subset = {agent_id: prev_positions.get(agent_id) for agent_id in collision_agents}
                actions_subset = {agent_id: actions.get(agent_id, 4) for agent_id in collision_agents}

                intended_pos_to_agents: Dict[Tuple[int, int], List[str]] = defaultdict(list)
                prev_pos_to_agents: Dict[Tuple[int, int], List[str]] = defaultdict(list)
                waiting_prev_positions: Dict[Tuple[int, int], List[str]] = defaultdict(list)

                for agent_id in collision_agents:
                    intended_pos = intended_subset.get(agent_id)
                    prev_pos = prev_subset.get(agent_id)
                    if intended_pos is not None:
                        intended_pos_to_agents[intended_pos].append(agent_id)
                    if prev_pos is not None:
                        prev_pos_to_agents[prev_pos].append(agent_id)
                        if actions_subset.get(agent_id, 4) == 4:
                            waiting_prev_positions[prev_pos].append(agent_id)

                for agent_id, delta in collision_deltas.items():
                    intended_pos = intended_subset.get(agent_id)
                    prev_pos = prev_subset.get(agent_id)

                    # Multiple agents moving into the same cell
                    if intended_pos is not None and len(intended_pos_to_agents.get(intended_pos, [])) > 1:
                        agent_agent_collision_total += delta
                        continue

                    is_agent_collision = False
                    if intended_pos is not None and prev_pos is not None:
                        # Swap positions check
                        for other_id in prev_pos_to_agents.get(intended_pos, []):
                            if other_id == agent_id:
                                continue
                            if intended_subset.get(other_id) == prev_pos:
                                is_agent_collision = True
                                break
                        # Moving into a waiting agent's cell
                        if not is_agent_collision:
                            waiting_agents = waiting_prev_positions.get(intended_pos, [])
                            if any(other_id != agent_id for other_id in waiting_agents):
                                is_agent_collision = True

                    if is_agent_collision:
                        agent_agent_collision_total += delta
                    else:
                        agent_obstacle_collision_total += delta

            step_count += 1
            done = dones.get("__all__", False) or truncated.get("__all__", False)

        env.close()
        elapsed_time = time.time() - start_time

        episode_len = step_count
        goal_completion_rate = (len(agents_reached_goal) / total_agents * 100.0) if total_agents > 0 else 0.0
        goal_completion_rate = max(0.0, min(goal_completion_rate, 100.0))
        success = (
            episode_len < max_steps
            and total_agents > 0
            and goal_completion_rate >= self._success_threshold
        )
        avg_steps_to_goal = (
            sum(agent_completion_steps.values()) / len(agent_completion_steps)
            if agent_completion_steps
            else np.nan
        )
        if agent_completion_steps:
            mean_completion = float(np.mean(list(agent_completion_steps.values())))
            completion_deviation = float(np.mean([abs(v - mean_completion) for v in agent_completion_steps.values()]))
        else:
            completion_deviation = np.nan

        total_collisions = agent_agent_collision_total + agent_obstacle_collision_total
        throughput = episode_len / elapsed_time if elapsed_time > 0 else 0.0
        throughput_goals_per_step = len(agents_reached_goal) / max(episode_len, 1)
        wait_fraction = wait_actions / max(episode_len * total_agents, 1)
        collisions_per_1000_steps = total_collisions / max(episode_len, 1) * 1000.0

        denom = max(episode_len * total_agents, 1)
        collision_agent_agent_rate = agent_agent_collision_total / denom
        collision_agent_obstacle_rate = agent_obstacle_collision_total / denom
        collision_total_rate = total_collisions / denom

        makespan_steps = episode_len if success else np.nan
        steps_to_half_completion = float(half_completion_step if half_completion_step is not None else max_steps)

        path_efficiency = np.nan
        # Option B: only successful agents contribute; failures are excluded (NaN efficiency).
        if getattr(env, "start_d_star_paths", None) and agent_completion_steps:
            efficiencies = []
            for agent_id, steps_to_goal in agent_completion_steps.items():
                start_path = env.start_d_star_paths.get(agent_id)
                if start_path:
                    optimal_len = max(len(start_path), 1)
                    efficiencies.append(steps_to_goal / optimal_len)
            if efficiencies:
                path_efficiency = float(np.mean(efficiencies))
        elif agent_completion_steps:
            # No D* paths available; use average steps-to-goal for successful agents only.
            path_efficiency = float(avg_steps_to_goal)

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
            agent_agent_collision_total,
            agent_obstacle_collision_total,
            wait_actions,
            goal_completion_rate,
            throughput,
            success,
        )

        return EpisodeResult(
            map_name=map_spec.label,
            checkpoint=self._checkpoint_path.name,
            seed=42 + repeat,
            agents_num=agents_num,
            repeat=repeat,
            episode_runtime_seconds=elapsed_time,
            episode_length_steps=episode_len,
            goals_completed=len(agents_reached_goal),
            total_reward=total_reward,
            total_collisions=total_collisions,
            collision_agent_agent=agent_agent_collision_total,
            collision_agent_obstacle=agent_obstacle_collision_total,
            throughput_steps_per_sec=throughput,
            throughput_goals_per_step=throughput_goals_per_step,
            wait_actions=wait_actions,
            wait_fraction=wait_fraction,
            goal_completion_rate_percent=goal_completion_rate,
            average_steps_to_goal=avg_steps_to_goal,
            completion_step_deviation=completion_deviation,
            success=success,
            path_efficiency=path_efficiency,
            collision_agent_agent_per_agent_step=collision_agent_agent_rate,
            collision_agent_obstacle_per_agent_step=collision_agent_obstacle_rate,
            collision_total_per_agent_step=collision_total_rate,
            collisions_per_1000_steps=collisions_per_1000_steps,
            success_episode_length_steps=success_episode_length,
            deadlock=deadlock,
            makespan_steps=makespan_steps,
            steps_to_half_completion=steps_to_half_completion,
        )

    def _write_intermediate_results(self, tracker: ResultsTracker, agents_num: int, results_dir: Path) -> None:
        agent_df = tracker.agent_dataframe(agents_num)
        if agent_df.empty:
            return

        avg_time = agent_df['episode_runtime_seconds'].mean()
        avg_length = agent_df['episode_length_steps'].mean()
        avg_success_length = agent_df['success_episode_length_steps'].mean()
        success_rate = agent_df['success'].mean() * 100.0
        deadlocks = agent_df['deadlock'].sum()

        logger.info("")
        logger.info("Results for %d agents:", agents_num)
        logger.info("  Average episode time: %.5f s", avg_time)
        logger.info("  Average episode length: %.2f steps", avg_length)
        logger.info("  Success rate: %.1f%%", success_rate)
        logger.info("  Deadlocks: %d/%d", deadlocks, len(agent_df))

        intermediate_name = results_dir / f'intermediate_results_{agents_num}_agents'
        intermediate_name.parent.mkdir(parents=True, exist_ok=True)

        with open(str(intermediate_name) + '.txt', 'w') as f:
            f.write(f"Results for {agents_num} agents:\n")
            f.write(f"Average episode time: {avg_time:.5f} seconds\n")
            f.write(f"Average episode length: {avg_length:.2f} steps\n")
            f.write(f"Average success episode length: {avg_success_length:.2f} steps\n")
            f.write(f"Success rate: {success_rate:.1f}%\n")
            f.write(f"Deadlocks: {deadlocks}/{len(agent_df)}\n")
            f.write("\nDetailed results:\n")
            for _, row in agent_df.sort_values(['repeat']).iterrows():
                f.write(
                    f"Repeat {int(row['repeat'])}: Time={row['episode_runtime_seconds']:.5f}s, "
                    f"Length={row['episode_length_steps']:.2f} steps, "
                    f"Success_len={row['success_episode_length_steps']}\n"
                )

        agent_df.to_csv(str(intermediate_name) + '.csv', index=False)

    def _write_final_results(self, tracker: ResultsTracker, summary_df: pd.DataFrame, results_dir: Path) -> None:
        final_name = results_dir / 'final_results'
        df = tracker.dataframe()
        df.to_csv(str(final_name) + '.csv', index=False)

        summary_df.to_csv(str(results_dir / 'summary.csv'), index=False)

        headline_cols = [
            'agents_num',
            'success_rate_percent',
            'deadlock_rate_percent',
            'avg_success_length_steps',
            'avg_makespan_steps',
            'avg_steps_to_half_completion',
            'avg_throughput_steps_per_sec',
            'avg_collision_agent_agent_per_agent_step',
            'avg_collision_agent_obstacle_per_agent_step',
            'avg_collision_total_per_agent_step',
            'avg_path_efficiency',
        ]
        existing_cols = [c for c in headline_cols if c in summary_df.columns]
        if existing_cols:
            summary_df[existing_cols].to_csv(str(results_dir / 'at_a_glance.csv'), index=False)

        with open(str(final_name) + '.txt', 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("Final Evaluation Results\n")
            f.write("=" * 80 + "\n\n")
            for _, item in summary_df.iterrows():
                f.write(f"Agents: {int(item['agents_num']):2d} | ")
                f.write(f"Time: {item['avg_time_seconds']:7.5f}s | ")
                f.write(f"Length: {item['avg_length_steps']:6.2f} steps | ")
                f.write(f"Success: {item['success_rate_percent']:.1f}% | ")
                f.write(f"Deadlocks: {int(item['deadlocks'])}/{int(item['total_runs'])}")
                f.write(f" ({item['deadlock_rate_percent']:.1f}% deadlock)\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write("Detailed Results\n")
            f.write("=" * 80 + "\n\n")
            for _, row in df.sort_values(['agents_num', 'repeat']).iterrows():
                f.write(
                    f"Agents: {int(row['agents_num'])}, Repeat: {int(row['repeat'])}, "
                    f"Time: {row['episode_runtime_seconds']:.5f}s, "
                    f"Length: {row['episode_length_steps']:.2f} steps\n"
                )

    def _generate_plots(self, results_dir: Path) -> None:
        logger.info("")
        logger.info("=" * 80)
        logger.info("GENERATING PLOTS")
        logger.info("=" * 80)

        plots_dir = results_dir / "plots"
        plots_dir.mkdir(exist_ok=True)

        episodes_csv = results_dir / 'final_results.csv'
        if episodes_csv.exists():
            episodes_df = pd.read_csv(episodes_csv)
            plot_success_vs_agents(episodes_df, plots_dir)
            plot_deadlock_vs_agents(episodes_df, plots_dir)
            plot_throughput_vs_agents(episodes_df, plots_dir)
            plot_collisions_vs_agents(episodes_df, plots_dir)
            plot_efficiency_vs_agents(episodes_df, plots_dir)
            plot_wait_fraction_vs_agents(episodes_df, plots_dir)
            plot_wait_fraction_vs_agents_extra(episodes_df, plots_dir)
            plot_makespan_vs_agents(episodes_df, plots_dir)
            plot_goal_completion_boxplot(episodes_df, plots_dir)
            plot_goal_completion_heatmap_detailed(episodes_df, plots_dir)
            plot_steps_to_half_completion_vs_agents(episodes_df, plots_dir)
            plot_progress_rate_vs_agents(episodes_df, plots_dir)
            plot_collision_diagnostic_hist(episodes_df, plots_dir)
            plot_dashboard_reliability(episodes_df, plots_dir)
            plot_dashboard_behavior(episodes_df, plots_dir)
            plot_goal_completion_heatmap(episodes_df, plots_dir)
            plot_tradeoff_efficiency_vs_collisions(episodes_df, plots_dir)
            if self._multi_map_enabled:
                plot_maps_success_heatmap(episodes_df, plots_dir)
            logger.info("Plots generated in: %s", plots_dir)
        else:
            logger.warning("final_results.csv not found, skipping plotting")

    def _run_cross_map_comparison(self, map_summary_paths: Dict[str, Path]) -> None:
        logger.info("")
        logger.info("=" * 80)
        logger.info("CROSS-MAP COMPARISON")
        logger.info("=" * 80)

        cross_map_dir = self._results_dir / "cross_map_comparison"
        cross_map_dir.mkdir(exist_ok=True)

        eval_maps_config = self._config.get('eval_maps', {})
        if eval_maps_config.get('generate_comparison_plots', True):
            plot_cross_map_comparison(map_summary_paths, cross_map_dir)

        if eval_maps_config.get('aggregate_results', True):
            create_cross_map_summary(map_summary_paths, cross_map_dir / 'all_maps_summary.csv')

        logger.info("Cross-map comparison saved to: %s", cross_map_dir)
        logger.info("=" * 80)


def main(argv: Optional[List[str]] = None) -> int:
    """Main evaluation function."""
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
    )

    args = parse_args(argv)
    try:
        evaluator = Evaluator(args)
    except ValueError as exc:
        logger.error(str(exc))
        return 1

    return evaluator.run()


if __name__ == "__main__":
    sys.exit(main())
