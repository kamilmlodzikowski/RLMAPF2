#!/usr/bin/env python3
"""Batch multi-model evaluator wrapper for eval.py."""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import yaml


def _load_spec(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Spec file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict) or "runs" not in data or not isinstance(data["runs"], list):
        raise ValueError("Spec must be a dict with a 'runs' list")
    return data


def _ensure_list(val: Any) -> List[str]:
    if val is None:
        return []
    if isinstance(val, list):
        return [str(v) for v in val]
    return [str(val)]


def _build_args(entry: Dict[str, Any], ts: str, results_dir: Optional[Path]) -> List[str]:
    required = ["config", "checkpoint"]
    for key in required:
        if key not in entry:
            raise ValueError(f"Missing required key '{key}' in run entry: {entry}")

    args: List[str] = [sys.executable, "eval.py", "--config", str(entry["config"]), "--checkpoint", str(entry["checkpoint"])]

    if entry.get("checkpoint_group"):
        args += ["--checkpoint-group", str(entry["checkpoint_group"])]
    if entry.get("checkpoint_strategy"):
        args += ["--checkpoint-strategy", str(entry["checkpoint_strategy"])]
    if entry.get("checkpoint_success_tolerance") is not None:
        args += ["--checkpoint-success-tolerance", str(entry["checkpoint_success_tolerance"])]

    if entry.get("sync_train_config"):
        args.append("--sync-train-config")
    if entry.get("clamp_agents_to_train"):
        args.append("--clamp-agents-to-train")
    if entry.get("render_video"):
        args.append("--render-video")

    if entry.get("video_agents"):
        args += ["--video-agents", str(entry["video_agents"])]
    if entry.get("success_threshold") is not None:
        args += ["--success-threshold", str(entry["success_threshold"])]
    if entry.get("num_threads") is not None:
        args += ["--num-threads", str(entry["num_threads"])]
    if entry.get("repeats") is not None:
        args += ["--repeats", str(entry["repeats"])]
    if results_dir:
        args += ["--results-dir", str(results_dir)]

    overrides = _ensure_list(entry.get("overrides"))
    for ov in overrides:
        args += ["--set", ov]

    # Derive run name if provided
    run_name = entry.get("run_name")
    if not run_name and entry.get("name_suffix"):
        run_name = f"{entry['config']}-{entry['name_suffix']}-{ts}"
    if run_name:
        args += ["--run-name", str(run_name)]

    return args


def _aggregate_models(model_summaries: Dict[str, Path], output_dir: Path) -> None:
    if len(model_summaries) < 2:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for label, path in model_summaries.items():
        if path.exists():
            df = pd.read_csv(path)
            df["model"] = label
            frames.append(df)
    if not frames:
        return
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(output_dir / "all_models_summary.csv", index=False)

    metrics = [
        ("success_rate_percent", "Success rate (%)"),
        ("deadlock_rate_percent", "Deadlock rate (%)"),
        ("avg_success_length_steps", "Episode length (success)"),
        ("avg_throughput_steps_per_sec", "Throughput (steps/s)"),
        ("avg_collision_agent_agent_per_agent_step", "Agent–Agent collisions/agent-step"),
        ("avg_collision_agent_obstacle_per_agent_step", "Agent–Obstacle collisions/agent-step"),
    ]

    # Save individual plots (mean only)
    for metric, title in metrics:
        fig, ax = plt.subplots(figsize=(8, 5))
        for label, group in combined.groupby("model"):
            if metric not in group.columns or "agents_num" not in group.columns:
                continue
            g = group.sort_values("agents_num")
            ax.plot(g["agents_num"], g[metric], marker="o", label=label)
        ax.set_xlabel("Number of agents")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"{metric}.png", dpi=200, bbox_inches="tight")
        plt.close()

    # Combined dashboard into one PNG
    rows, cols = 2, 3
    fig, axes = plt.subplots(rows, cols, figsize=(18, 10))
    axes = axes.flatten()
    for idx, (metric, title) in enumerate(metrics):
        ax = axes[idx]
        for label, group in combined.groupby("model"):
            if metric not in group.columns or "agents_num" not in group.columns:
                continue
            g = group.sort_values("agents_num")
            ax.plot(g["agents_num"], g[metric], marker="o", label=label)
        ax.set_xlabel("Number of agents")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "cross_model_comparison.png", dpi=200, bbox_inches="tight")
    plt.close()


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="Run multiple model evaluations via eval.py")
    parser.add_argument("spec", help="YAML spec listing runs")
    parser.add_argument("--dry-run", action="store_true", help="Print planned commands without executing")
    parser.add_argument(
        "--output-root",
        default="experiments/eval/multi_model",
        help="Base directory to store all model runs (default: experiments/eval/multi_model)",
    )
    args = parser.parse_args(argv)

    spec_path = Path(args.spec).resolve()
    spec = _load_spec(spec_path)
    ts = time.strftime("%Y%m%d-%H%M%S")
    base_output = Path(args.output_root).resolve() / ts
    base_output.mkdir(parents=True, exist_ok=True)

    model_summaries: Dict[str, Path] = {}
    for idx, entry in enumerate(spec.get("runs", []), start=1):
        label = str(entry.get("name_suffix") or entry.get("run_name") or entry.get("config") or f"model_{idx}")
        results_dir = base_output / label
        cmd = _build_args(entry, ts, results_dir)
        print(f"[{idx}/{len(spec['runs'])}] Running: {' '.join(cmd)}")
        if args.dry_run:
            continue
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            print(f"Run {idx} failed with exit code {result.returncode}", file=sys.stderr)
            return result.returncode
        summary_path = results_dir / "summary.csv"
        model_summaries[label] = summary_path

    _aggregate_models(model_summaries, base_output / "cross_model_comparison")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
