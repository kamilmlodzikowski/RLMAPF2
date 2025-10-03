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
    parser.add_argument("--gpus-per-trial", type=float, default=1,
                        help="Number of GPUs to allocate per trial (default: 1)")
    parser.add_argument(
        "--use_cnn_observation",
        action="store_true",
        help="Append '--use_cnn_observation True' to each train.py invocation unless already present.",
    )
    args = parser.parse_args()

    lines = parse_config_file(args.config_file)

    # Optionally enforce CNN observation for all trials unless already specified per-line
    if args.use_cnn_observation:
        enforced_lines = []
        for line in lines:
            if "--use_cnn_observation" in line:
                enforced_lines.append(line)
            else:
                enforced_lines.append(line + " --use_cnn_observation True")
        lines = enforced_lines

    ray.init()
    resources = {"cpu": 4}
    if args.gpus_per_trial > 0:
        resources["gpu"] = args.gpus_per_trial
    trainable = tune.with_resources(train_with_args, resources)
    tuner = tune.Tuner(
        trainable,
        param_space={"args": tune.grid_search(lines)},
        tune_config=tune.TuneConfig(max_concurrent_trials=args.max_concurrent_trials),
    )
    tuner.fit()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received. Shutting down Ray and cleaning up subprocesses.")
        ray.shutdown()
