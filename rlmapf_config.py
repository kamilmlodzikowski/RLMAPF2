"""Configuration helpers for RLMAPF training scripts."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import json

import yaml


def _deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``updates`` into ``base`` (modifies and returns ``base``)."""
    for key, value in updates.items():
        if (
            key in base
            and isinstance(base[key], dict)
            and isinstance(value, dict)
        ):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _cast_value(value: str) -> Any:
    """Best-effort cast of string overrides to Python objects."""
    value = value.strip()
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.lower() == "null" or value.lower() == "none":
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
    # Allows Ray resource overrides without editing script
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
        # Convert PathsConfig back to serialisable dict for merging
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
