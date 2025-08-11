#!/usr/bin/env python3
"""Launch multiple training runs using Ray Tune.

This script reads a configuration file where each line contains command line
arguments for ``train.py``. Ray Tune schedules each line as a separate trial
and manages parallel execution.
"""

import argparse
import subprocess
from pathlib import Path

import ray
from ray import tune


TRAIN_SCRIPT = Path(__file__).parent / "train.py"


def train_with_args(config: dict) -> None:
    """Run ``train.py`` with arguments provided by Ray Tune."""
    args = config["args"].split()
    cmd = ["python", str(TRAIN_SCRIPT)] + args
    subprocess.run(cmd, check=True)


def parse_config_file(path: Path) -> list[str]:
    """Return non-empty, non-comment lines from the config file."""
    lines = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                lines.append(line)
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch multiple training runs with Ray Tune")
    parser.add_argument("config_file", type=Path, help="File with one set of args per line")
    parser.add_argument("--max_concurrent_trials", type=int, default=5,
                        help="Maximum number of concurrent training runs")
    args = parser.parse_args()

    lines = parse_config_file(args.config_file)

    ray.init()
    trainable = tune.with_resources(train_with_args, {"cpu": 1})
    tuner = tune.Tuner(
        trainable,
        param_space={"args": tune.grid_search(lines)},
        tune_config=tune.TuneConfig(max_concurrent_trials=args.max_concurrent_trials),
    )
    tuner.fit()


if __name__ == "__main__":
    main()
