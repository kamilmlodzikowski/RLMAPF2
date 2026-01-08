"""MAPF-style plotting helpers used by eval.py."""
from pathlib import Path
from typing import Dict, Optional

import logging
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger("eval")


def _line_with_ci(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    out_path: Path,
    title: str,
    ylabel: str,
    y_lim: Optional[tuple[float, float]] = None,
) -> None:
    if df.empty or y_col not in df.columns:
        return
    grouped = df.groupby(x_col)[y_col].agg(["mean", "count", "std"]).reset_index()
    grouped["ci"] = 1.96 * grouped["std"] / grouped["count"].clip(lower=1) ** 0.5
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(grouped[x_col], grouped["mean"], yerr=grouped["ci"], fmt="-o", capsize=3)
    ax.set_xlabel("Number of agents")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if y_lim is not None:
        ax.set_ylim(*y_lim)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def _multiline_with_ci(df: pd.DataFrame, x_col: str, series: Dict[str, str], out_path: Path, title: str, ylabel: str) -> None:
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for col, label in series.items():
        if col not in df.columns:
            continue
        grouped = df.groupby(x_col)[col].agg(["mean", "count", "std"]).reset_index()
        grouped["ci"] = 1.96 * grouped["std"] / grouped["count"].clip(lower=1) ** 0.5
        ax.errorbar(grouped[x_col], grouped["mean"], yerr=grouped["ci"], fmt="-o", capsize=3, label=label)
    ax.set_xlabel("Number of agents")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def _aggregate_mean_ci(df: pd.DataFrame, x_col: str, y_col: str) -> Optional[pd.DataFrame]:
    if df.empty or y_col not in df.columns:
        return None
    grouped = df[[x_col, y_col]].dropna().groupby(x_col)[y_col].agg(["mean", "count", "std"]).reset_index()
    if grouped.empty:
        return None
    grouped["ci"] = 1.96 * grouped["std"] / grouped["count"].clip(lower=1) ** 0.5
    return grouped


# --------------------------------------------------------------------------- #
# Core plots (literature-aligned)
# --------------------------------------------------------------------------- #

def plot_success_vs_agents(df: pd.DataFrame, out_dir: Path) -> None:
    """Success rate vs agents (mean ± 95% CI)."""
    if "goal_completion_rate_percent" not in df.columns:
        logger.warning("goal_completion_rate_percent missing; skipping success plot")
        return
    _line_with_ci(
        df,
        x_col="agents_num",
        y_col="goal_completion_rate_percent",
        out_path=out_dir / "success_vs_agents.png",
        title="Success vs agents",
        ylabel="Success rate (%)",
        y_lim=(0, 100),
    )
    if "episode_success" in df.columns:
        _line_with_ci(
            df,
            x_col="agents_num",
            y_col="episode_success",
            out_path=out_dir / "episode_success_vs_agents.png",
            title="Episode success vs agents",
            ylabel="Episode success rate",
            y_lim=(0, 1),
        )


def plot_deadlock_vs_agents(df: pd.DataFrame, out_dir: Path) -> None:
    """Deadlock/timeout rate vs agents (mean ± 95% CI)."""
    if "deadlock" not in df.columns:
        logger.warning("deadlock column missing; skipping deadlock plot")
        return
    _line_with_ci(
        df,
        x_col="agents_num",
        y_col="deadlock",
        out_path=out_dir / "timeout_or_deadlock_vs_agents.png",
        title="Deadlock/timeout vs agents",
        ylabel="Deadlock/timeout rate",
        y_lim=(0, 1),
    )


def plot_throughput_vs_agents(df: pd.DataFrame, out_dir: Path) -> None:
    """Throughput vs agents (mean ± 95% CI)."""
    if "throughput_goals_per_step" in df.columns:
        col = "throughput_goals_per_step"
        ylabel = "Goals per step"
    elif "throughput_steps_per_sec" in df.columns:
        col = "throughput_steps_per_sec"
        ylabel = "Steps per second"
    else:
        logger.warning("No throughput column found; skipping throughput plot")
        return
    _line_with_ci(
        df,
        x_col="agents_num",
        y_col=col,
        out_path=out_dir / "throughput_vs_agents.png",
        title="Throughput vs agents",
        ylabel=ylabel,
    )


def plot_collisions_vs_agents(df: pd.DataFrame, out_dir: Path) -> None:
    """Collisions per 1000 steps vs agents (split by type, mean ± CI, shaded)."""
    if "collisions_per_1000_steps" not in df.columns:
        df["collisions_per_1000_steps"] = df["total_collisions"] / df["episode_length_steps"].clip(lower=1) * 1000.0
    if "collision_agent_agent" in df.columns and "collision_agent_agent_per_1000_steps" not in df.columns:
        df["collision_agent_agent_per_1000_steps"] = (
            df["collision_agent_agent"] / df["episode_length_steps"].clip(lower=1) * 1000.0
        )
    if "collision_agent_obstacle" in df.columns and "collision_agent_obstacle_per_1000_steps" not in df.columns:
        df["collision_agent_obstacle_per_1000_steps"] = (
            df["collision_agent_obstacle"] / df["episode_length_steps"].clip(lower=1) * 1000.0
        )

    series = [
        ("collisions_per_1000_steps", "Total (per 1000 steps)"),
        ("collision_agent_agent_per_1000_steps", "Agent-Agent (per 1000 steps)"),
        ("collision_agent_obstacle_per_1000_steps", "Agent-Obstacle (per 1000 steps)"),
    ]
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for idx, (col, label) in enumerate(series):
        if col not in df.columns:
            continue
        g = df[[ "agents_num", col ]].dropna().groupby("agents_num")[col].agg(["mean", "count", "std"]).reset_index()
        if g.empty:
            continue
        g["std"] = g["std"].fillna(0.0)
        g["ci"] = 1.96 * g["std"] / g["count"].clip(lower=1) ** 0.5
        color = colors[idx % len(colors)]
        ax.plot(g["agents_num"], g["mean"], marker="o", label=label, color=color)
        ax.fill_between(
            g["agents_num"],
            g["mean"] - g["ci"],
            g["mean"] + g["ci"],
            color=color,
            alpha=0.15,
            linewidth=0,
        )

    ax.set_xlabel("Number of agents")
    ax.set_ylabel("Collisions (normalized)")
    ax.set_title("Collisions vs agents")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "collisions_vs_agents.png", dpi=200, bbox_inches="tight")
    plt.close()


def plot_efficiency_vs_agents(df: pd.DataFrame, out_dir: Path) -> None:
    """Path efficiency vs agents (mean ± CI)."""
    if "path_efficiency" not in df.columns:
        logger.warning("path_efficiency missing; skipping efficiency plot")
        return
    _line_with_ci(
        df,
        x_col="agents_num",
        y_col="path_efficiency",
        out_path=out_dir / "efficiency_vs_agents.png",
        title="Path efficiency vs agents",
        ylabel="Path efficiency (actual/optimal)",
    )


def plot_wait_fraction_vs_agents(df: pd.DataFrame, out_dir: Path) -> None:
    """Wait fraction vs agents (mean ± CI)."""
    if "wait_fraction" not in df.columns:
        logger.warning("wait_fraction missing; skipping wait plot")
        return
    _line_with_ci(
        df,
        x_col="agents_num",
        y_col="wait_fraction",
        out_path=out_dir / "wait_fraction_vs_agents.png",
        title="Wait fraction vs agents",
        ylabel="Wait fraction (wait actions / total actions)",
    )


def plot_tradeoff_efficiency_vs_collisions(df: pd.DataFrame, out_dir: Path) -> None:
    """Trade-off scatter: efficiency vs collisions (per rollout)."""
    if df.empty or "path_efficiency" not in df.columns or "collisions_per_1000_steps" not in df.columns:
        logger.warning("Missing columns for tradeoff plot; skipping")
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(
        df["collisions_per_1000_steps"],
        df["path_efficiency"],
        c=df["agents_num"],
        cmap="viridis",
        edgecolors="k",
        alpha=0.8,
    )
    ax.set_xlabel("Collisions per 1000 steps")
    ax.set_ylabel("Path efficiency (actual/optimal)")
    ax.set_title("Efficiency vs Collisions")
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Agents")
    plt.tight_layout()
    plt.savefig(out_dir / "tradeoff_efficiency_vs_collisions.png", dpi=200, bbox_inches="tight")
    plt.close()


def plot_makespan_vs_agents(df: pd.DataFrame, out_dir: Path) -> None:
    """Makespan/time-to-solve vs agents with fallback to 50% completion."""
    if df.empty:
        return

    # Ensure fallback column exists
    if "steps_to_half_completion" not in df.columns:
        logger.warning("steps_to_half_completion missing; skipping makespan plot")
        return

    if "makespan_steps" not in df.columns:
        df["makespan_steps"] = np.nan

    if "steps_to_half_completion" in df.columns:
        df["steps_to_half_completion"] = df["steps_to_half_completion"]

    series = {
        "makespan_steps": "Makespan (success only)",
        "steps_to_half_completion": "Steps to 50% completion / capped",
    }
    _multiline_with_ci(
        df,
        x_col="agents_num",
        series=series,
        out_path=out_dir / "makespan_vs_agents.png",
        title="Makespan / Time-to-solve",
        ylabel="Steps",
    )


def _plot_line_to_axis(ax, grouped: Optional[pd.DataFrame], title: str, ylabel: str) -> None:
    if grouped is None or grouped.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_title(title)
        ax.axis("off")
        return
    ax.errorbar(grouped.iloc[:, 0], grouped["mean"], yerr=grouped["ci"], fmt="-o", capsize=3)
    ax.set_title(title)
    ax.set_xlabel("Number of agents")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.axis("on")


def plot_dashboard_reliability(df: pd.DataFrame, out_dir: Path) -> None:
    """4-up dashboard: success, deadlock, throughput, makespan/time-to-solve."""
    if df.empty:
        return
    # Ensure needed derived columns exist
    if "collisions_per_1000_steps" not in df.columns and "total_collisions" in df.columns:
        df["collisions_per_1000_steps"] = df["total_collisions"] / df["episode_length_steps"].clip(lower=1) * 1000.0
    makespan_fallback_col = "steps_to_half_completion"
    if "makespan_steps" not in df.columns:
        df["makespan_steps"] = np.nan
    if makespan_fallback_col not in df.columns:
        makespan_fallback_col = None

    throughput_col = "throughput_goals_per_step" if "throughput_goals_per_step" in df.columns else "throughput_steps_per_sec"

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    # Success
    success_g = _aggregate_mean_ci(df, "agents_num", "goal_completion_rate_percent")
    _plot_line_to_axis(axes[0, 0], success_g, "Success vs agents", "Success rate (%)")
    # Deadlock
    deadlock_col = "deadlock_rate_percent"
    if "deadlock" in df.columns:
        df[deadlock_col] = df["deadlock"] * 100.0
    deadlock_g = _aggregate_mean_ci(df, "agents_num", deadlock_col)
    _plot_line_to_axis(axes[0, 1], deadlock_g, "Deadlock/timeout vs agents", "Deadlock/timeout rate (%)")
    # Throughput
    throughput_g = _aggregate_mean_ci(df, "agents_num", throughput_col) if throughput_col else None
    _plot_line_to_axis(axes[1, 0], throughput_g, "Throughput vs agents", throughput_col.replace("_", " "))
    # Makespan
    series = {}
    series["makespan_steps"] = "Makespan (success)"
    if makespan_fallback_col:
        series[makespan_fallback_col] = "Steps to 50% completion"
    ax_ms = axes[1, 1]
    has_any = False
    for col, label in series.items():
        g = _aggregate_mean_ci(df, "agents_num", col)
        if g is None:
            continue
        has_any = True
        ax_ms.errorbar(g["agents_num"], g["mean"], yerr=g["ci"], fmt="-o", capsize=3, label=label)
    if has_any:
        ax_ms.set_title("Makespan / time-to-solve")
        ax_ms.set_xlabel("Number of agents")
        ax_ms.set_ylabel("Steps")
        ax_ms.grid(True, alpha=0.3)
        ax_ms.legend()
    else:
        ax_ms.text(0.5, 0.5, "No data", ha="center", va="center")
        ax_ms.axis("off")

    plt.tight_layout()
    plt.savefig(out_dir / "dashboard_reliability.png", dpi=250, bbox_inches="tight")
    plt.close()


def plot_dashboard_behavior(df: pd.DataFrame, out_dir: Path) -> None:
    """4-up dashboard: collisions, wait fraction, path efficiency, efficiency vs collisions scatter."""
    if df.empty:
        return
    if "collisions_per_1000_steps" not in df.columns and "total_collisions" in df.columns:
        df["collisions_per_1000_steps"] = df["total_collisions"] / df["episode_length_steps"].clip(lower=1) * 1000.0

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    # Collisions
    coll_g = _aggregate_mean_ci(df, "agents_num", "collisions_per_1000_steps")
    _plot_line_to_axis(axes[0, 0], coll_g, "Collisions vs agents", "Collisions per 1000 steps")
    # Wait
    wait_g = _aggregate_mean_ci(df, "agents_num", "wait_fraction")
    _plot_line_to_axis(axes[0, 1], wait_g, "Wait fraction vs agents", "Wait fraction")
    # Efficiency
    eff_g = _aggregate_mean_ci(df, "agents_num", "path_efficiency")
    _plot_line_to_axis(axes[1, 0], eff_g, "Path efficiency vs agents", "Path efficiency")
    # Scatter efficiency vs collisions
    ax_scatter = axes[1, 1]
    if not df.empty and "path_efficiency" in df.columns and "collisions_per_1000_steps" in df.columns:
        sc = ax_scatter.scatter(
            df["collisions_per_1000_steps"],
            df["path_efficiency"],
            c=df["agents_num"],
            cmap="viridis",
            edgecolors="k",
            alpha=0.8,
        )
        ax_scatter.set_xlabel("Collisions per 1000 steps")
        ax_scatter.set_ylabel("Path efficiency")
        ax_scatter.set_title("Efficiency vs collisions")
        ax_scatter.grid(True, alpha=0.3)
        cbar = plt.colorbar(sc, ax=ax_scatter)
        cbar.set_label("Agents")
    else:
        ax_scatter.text(0.5, 0.5, "No data", ha="center", va="center")
        ax_scatter.axis("off")

    plt.tight_layout()
    plt.savefig(out_dir / "dashboard_behavior.png", dpi=250, bbox_inches="tight")
    plt.close()


def plot_goal_completion_heatmap(df: pd.DataFrame, out_dir: Path) -> None:
    """Heatmap of agents reaching the goal per rollout (agents x repeat)."""
    if df.empty or "agents_num" not in df.columns:
        return
    if "goals_completed" not in df.columns:
        if "goal_completion_rate_percent" not in df.columns:
            logger.warning("No goals_completed or completion rate; skipping goal completion heatmap")
            return
        df = df.copy()
        df["goals_completed"] = df["goal_completion_rate_percent"] * df["agents_num"] / 100.0

    repeat_col = "repeat" if "repeat" in df.columns else None
    if repeat_col is None:
        logger.warning("Repeat column missing; skipping goal completion heatmap")
        return

    pivot = df.pivot_table(index="agents_num", columns=repeat_col, values="goals_completed", aggfunc="mean")
    if pivot.empty:
        return

    fig_w = max(6, 0.4 * pivot.shape[1] + 4)
    fig_h = max(4, 0.3 * pivot.shape[0] + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(pivot.values, aspect="auto", origin="lower", cmap="viridis")
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels([str(c) for c in pivot.columns])
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels([str(r) for r in pivot.index])
    ax.set_xlabel("Repeat")
    ax.set_ylabel("Number of agents")
    ax.set_title("Agents reaching goal (per rollout)")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Agents reached goal (avg)")
    plt.tight_layout()
    plt.savefig(out_dir / "goal_completion_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close()


# --------------------------------------------------------------------------- #
# Multi-map plots
# --------------------------------------------------------------------------- #

def plot_maps_success_heatmap(df: pd.DataFrame, out_dir: Path) -> None:
    """Heatmap of success across maps vs agents."""
    if df.empty or "map_name" not in df.columns or "goal_completion_rate_percent" not in df.columns:
        logger.warning("Insufficient data for maps heatmap; skipping")
        return
    pivot = df.pivot_table(
        index="map_name",
        columns="agents_num",
        values="goal_completion_rate_percent",
        aggfunc="mean",
    )
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 4 + 0.4 * len(pivot)))
    im = ax.imshow(pivot.values, aspect="auto", origin="lower", cmap="viridis")
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels([str(c) for c in pivot.columns])
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Number of agents")
    ax.set_ylabel("Map")
    ax.set_title("Success across maps")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Success rate (%)")
    plt.tight_layout()
    plt.savefig(out_dir / "maps_success_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close()


def plot_maps_success_boxplot(df: pd.DataFrame, out_dir: Path) -> None:
    """Boxplot of map-wise success per agent count."""
    if df.empty or "map_name" not in df.columns or "goal_completion_rate_percent" not in df.columns:
        return
    grouped = df.groupby(["map_name", "agents_num"])["goal_completion_rate_percent"].mean().reset_index()
    if grouped.empty:
        return
    agent_groups = sorted(grouped["agents_num"].unique())
    data = [grouped[grouped["agents_num"] == n]["goal_completion_rate_percent"] for n in agent_groups]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(data, labels=agent_groups, patch_artist=True)
    ax.set_xlabel("Number of agents")
    ax.set_ylabel("Success rate (%)")
    ax.set_title("Success across maps (boxplot)")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "maps_success_boxplot.png", dpi=200, bbox_inches="tight")
    plt.close()


def plot_cross_map_comparison(map_summaries: Dict[str, Path], plots_dir: Path):
    """Legacy cross-map comparison (line plots per map)."""
    try:
        dfs = {}
        for label, summary_path in map_summaries.items():
            if summary_path.exists():
                dfs[label] = pd.read_csv(summary_path)
        if len(dfs) < 2:
            logger.warning("Need at least 2 maps for cross-map comparison")
            return
        metrics = [
            ("success_rate_percent", "Success rate (%)"),
            ("deadlock_rate_percent", "Deadlock rate (%)"),
            ("avg_success_length_steps", "Episode length (success)"),
            ("avg_throughput_steps_per_sec", "Throughput (steps/s)"),
            ("avg_collision_agent_agent_per_agent_step", "Agent–Agent collisions/agent-step"),
            ("avg_collision_agent_obstacle_per_agent_step", "Agent–Obstacle collisions/agent-step"),
        ]
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        for idx, (metric, title) in enumerate(metrics):
            for map_label, df in dfs.items():
                if metric in df.columns:
                    axes[idx].plot(df["agents_num"], df[metric], marker="o", linewidth=2, label=map_label)
            axes[idx].set_xlabel("Number of agents")
            axes[idx].set_ylabel(title)
            axes[idx].set_title(title)
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / "cross_map_comparison.png", dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("Cross-map comparison plot saved")
    except Exception as e:
        logger.warning("Could not generate cross-map comparison plot: %s", e)


# --------------------------------------------------------------------------- #
# Legacy helpers still used elsewhere
# --------------------------------------------------------------------------- #

def create_cross_map_summary(map_summaries: Dict[str, Path], output_path: Path):
    """Aggregate all map results into single CSV."""
    try:
        all_data = []
        for map_label, summary_path in map_summaries.items():
            if summary_path.exists():
                df = pd.read_csv(summary_path)
                df["map"] = map_label
                all_data.append(df)
        if not all_data:
            logger.warning("No map summaries found for aggregation")
            return
        combined_df = pd.concat(all_data, ignore_index=True)
        cols = ["map"] + [col for col in combined_df.columns if col != "map"]
        combined_df = combined_df[cols]
        combined_df.to_csv(output_path, index=False)
        logger.info("Cross-map summary saved to %s", output_path)
    except Exception as e:
        logger.warning("Could not create cross-map summary: %s", e)
