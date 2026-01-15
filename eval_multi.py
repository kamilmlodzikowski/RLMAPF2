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
import numpy as np


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


def _build_args(
    entry: Dict[str, Any],
    ts: str,
    results_dir: Optional[Path],
    global_maps: Optional[List[str]] = None,
) -> List[str]:
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

    maps = entry.get("maps") or entry.get("map_overrides") or global_maps
    if maps:
        if isinstance(maps, (str, Path)):
            maps = [str(maps)]
        # Clear existing map list, then set overrides
        args += ["--set", "environment.maps_names_with_variants=null"]
        for m in maps:
            args += ["--set", f"environment.maps_names_with_variants.{m}=null"]

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


def _aggregate_models(model_summaries: Dict[str, Path], output_dir: Path, model_results: Optional[Dict[str, Path]] = None) -> None:
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
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "cross_model_comparison.png", dpi=200, bbox_inches="tight")
    plt.close()

    if model_results:
        _plot_cross_model_diagnostics(model_results, output_dir)


def _find_column(df: pd.DataFrame, aliases) -> Optional[str]:
    for col in aliases:
        if col in df.columns:
            return col
    return None


def _agg_mean_ci(df: pd.DataFrame, value_col: str) -> Optional[pd.DataFrame]:
    if "agents_num" not in df.columns or value_col not in df.columns:
        return None
    grouped = df[["agents_num", value_col]].dropna().groupby("agents_num")[value_col].agg(["mean", "count", "std"]).reset_index()
    if grouped.empty:
        return None
    grouped["std"] = grouped["std"].fillna(0.0)
    grouped["ci"] = 1.96 * grouped["std"] / grouped["count"].clip(lower=1) ** 0.5
    return grouped


def _plot_cross_model_diagnostics(model_results: Dict[str, Path], output_dir: Path) -> None:
    """Diagnostic 4-up figure for cross-model evaluation."""
    frames = []
    for label, path in model_results.items():
        if path and Path(path).exists():
            df = pd.read_csv(path)
            if df.empty:
                continue
            df = df.copy()
            df["model"] = label
            frames.append(df)
    if not frames:
        print("No final_results.csv found for cross-model diagnostics")
        return
    data = pd.concat(frames, ignore_index=True)
    if data.empty:
        print("No data available for cross-model diagnostics")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    comp_col = _find_column(data, ["goal_completion_rate_percent", "goal_completion_percent", "goal_completion"])

    # Goal completion heatmap (rows = model-repeat, cols = agents)
    ax = axes[0]
    if comp_col and "agents_num" in data.columns:
        heat_df = data[["agents_num", comp_col, "model"]].dropna()
        repeat_col = _find_column(data, ["repeat", "repeat_idx", "episode", "episode_idx"])
        if repeat_col is None:
            heat_df["repeat_idx"] = heat_df.groupby(["model", "agents_num"]).cumcount()
        else:
            heat_df["repeat_idx"] = data.loc[heat_df.index, repeat_col]
        heat_df["row_id"] = heat_df["model"].astype(str) + "_r" + heat_df["repeat_idx"].astype(str)
        pivot = heat_df.pivot_table(index="row_id", columns="agents_num", values=comp_col, aggfunc="mean")
        if pivot.empty:
            ax.text(0.5, 0.5, "No heatmap data", ha="center", va="center")
            ax.axis("off")
        else:
            im = ax.imshow(pivot.values, aspect="auto", origin="lower", cmap="viridis")
            ax.set_xticks(range(pivot.shape[1]))
            ax.set_xticklabels([str(c) for c in pivot.columns])
            ax.set_yticks(range(pivot.shape[0]))
            ax.set_yticklabels(pivot.index)
            ax.set_xlabel("Number of agents")
            ax.set_ylabel("Model/repeat")
            ax.set_title("Goal completion heatmap")
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label("Goal completion rate (%)")
    else:
        ax.text(0.5, 0.5, "Goal completion data missing", ha="center", va="center")
        ax.axis("off")

    # Wait fraction vs agents
    ax = axes[1]
    wait_col = _find_column(data, ["wait_fraction", "wait_ratio", "frac_wait", "wait_frac"])
    if wait_col and "agents_num" in data.columns:
        has_any = False
        for model, df_m in data.groupby("model"):
            agg = _agg_mean_ci(df_m, wait_col)
            if agg is None:
                continue
            has_any = True
            ax.plot(agg["agents_num"], agg["mean"], marker="o", label=model)
            ax.fill_between(
                agg["agents_num"],
                agg["mean"] - agg["ci"],
                agg["mean"] + agg["ci"],
                alpha=0.15,
            )
        if has_any:
            ax.set_xlabel("Number of agents")
            ax.set_ylabel("Wait fraction")
            ax.set_title("Wait fraction vs agents")
            ax.grid(True, alpha=0.3)
            ax.legend()
        else:
            ax.text(0.5, 0.5, "No wait fraction data", ha="center", va="center")
            ax.axis("off")
    else:
        ax.text(0.5, 0.5, "Wait fraction data missing", ha="center", va="center")
        ax.axis("off")

    # Steps to 50% completion
    ax = axes[2]
    steps_col = _find_column(
        data,
        ["steps_to_half_completion", "steps_to_50pct_completion", "half_completion_steps"],
    )
    if steps_col and "agents_num" in data.columns:
        has_any = False
        for model, df_m in data.groupby("model"):
            agg = _agg_mean_ci(df_m, steps_col)
            if agg is None:
                continue
            has_any = True
            ax.plot(agg["agents_num"], agg["mean"], marker="o", label=model)
            ax.fill_between(
                agg["agents_num"],
                agg["mean"] - agg["ci"],
                agg["mean"] + agg["ci"],
                alpha=0.15,
            )
        if has_any:
            ax.set_xlabel("Number of agents")
            ax.set_ylabel("Steps to 50% completion")
            ax.set_title("Steps to 50% completion vs agents")
            ax.grid(True, alpha=0.3)
            ax.legend()
        else:
            ax.text(0.5, 0.5, "No steps-to-half data", ha="center", va="center")
            ax.axis("off")
    else:
        ax.text(0.5, 0.5, "Steps-to-half data missing", ha="center", va="center")
        ax.axis("off")

    # Progress rate vs agents
    ax = axes[3]
    length_col = _find_column(
        data,
        ["episode_length_steps", "episode_length", "episode_len", "steps", "timesteps"],
    )
    if comp_col and length_col and "agents_num" in data.columns:
        prog_data = data[[comp_col, length_col, "agents_num", "model"]].dropna()
        prog_data = prog_data[prog_data[length_col] > 0]
        if prog_data.empty:
            ax.text(0.5, 0.5, "No progress data", ha="center", va="center")
            ax.axis("off")
        else:
            has_any = False
            prog_data = prog_data.copy()
            prog_data["progress_rate"] = prog_data[comp_col] / prog_data[length_col]
            for model, df_m in prog_data.groupby("model"):
                agg = _agg_mean_ci(df_m, "progress_rate")
                if agg is None:
                    continue
                has_any = True
                ax.plot(agg["agents_num"], agg["mean"], marker="o", label=model)
                ax.fill_between(
                    agg["agents_num"],
                    agg["mean"] - agg["ci"],
                    agg["mean"] + agg["ci"],
                    alpha=0.15,
                )
            if has_any:
                ax.set_xlabel("Number of agents")
                ax.set_ylabel("Completion percent per step")
                ax.set_title("Progress rate vs agents")
                ax.grid(True, alpha=0.3)
                ax.legend()
            else:
                ax.text(0.5, 0.5, "No progress data", ha="center", va="center")
                ax.axis("off")
    else:
        ax.text(0.5, 0.5, "Progress data missing", ha="center", va="center")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_dir / "cross_model_diagnostics.png", dpi=300, bbox_inches="tight")
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
    global_maps = spec.get("maps") or spec.get("map_overrides")
    ts = time.strftime("%Y%m%d-%H%M%S")
    base_output = Path(args.output_root).resolve() / ts
    base_output.mkdir(parents=True, exist_ok=True)

    model_summaries: Dict[str, Path] = {}
    model_results: Dict[str, Path] = {}
    for idx, entry in enumerate(spec.get("runs", []), start=1):
        label = str(entry.get("name_suffix") or entry.get("run_name") or entry.get("config") or f"model_{idx}")
        results_dir = base_output / label
        cmd = _build_args(entry, ts, results_dir, global_maps=global_maps)
        print(f"[{idx}/{len(spec['runs'])}] Running: {' '.join(cmd)}")
        if args.dry_run:
            continue
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            print(f"Run {idx} failed with exit code {result.returncode}", file=sys.stderr)
            return result.returncode
        summary_path = results_dir / "summary.csv"
        model_summaries[label] = summary_path
        model_results[label] = results_dir / "final_results.csv"

    _aggregate_models(model_summaries, base_output / "cross_model_comparison", model_results)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
