#!/usr/bin/env python3
"""Launch multiple training runs using Ray Tune.

This script reads a configuration file where each line contains command line
arguments for ``train.py``. Ray Tune schedules each line as a separate trial
and manages parallel execution.
"""

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

import ray
from ray import tune


TRAIN_SCRIPT = Path(__file__).resolve().parent / "train.py"


def train_with_args(config: dict) -> None:
    """Run ``train.py`` with arguments provided by Ray Tune."""
    raw_args = config["args"]
    tokens = shlex.split(raw_args)
    if tokens and not any(token in {"--config", "-c"} for token in tokens):
        tokens = ["--config", *tokens]
    cmd = [sys.executable, str(TRAIN_SCRIPT)] + tokens
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
    parser.add_argument("--gpus-per-trial", type=float, default=0.25,
                        help="Number of GPUs to allocate per trial (default: 0.25)")
    parser.add_argument(
        "--use_cnn_observation",
        action="store_true",
        help="Ensure each run sets environment.use_cnn_observation=True via config override.",
    )
    args = parser.parse_args()

    lines = parse_config_file(args.config_file)

    # Optionally enforce CNN observation for all trials unless already specified per-line
    if args.use_cnn_observation:
        enforced_lines = []
        for line in lines:
            tokens = shlex.split(line)
            has_override = False
            for idx, token in enumerate(tokens):
                if token in {"--set", "-s"} and idx + 1 < len(tokens):
                    if tokens[idx + 1].startswith("environment.use_cnn_observation"):
                        has_override = True
                        break
                if token.startswith("environment.use_cnn_observation"):
                    has_override = True
                    break
                if token in {"--use_cnn_observation", "--use-cnn-observation"}:
                    has_override = True
                    break
            if not has_override:
                tokens.extend(["--set", "environment.use_cnn_observation=True"])
            enforced_lines.append(shlex.join(tokens))
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
