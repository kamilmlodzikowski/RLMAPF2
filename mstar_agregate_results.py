#!/usr/bin/env python3
"""
Plot average elapsed_time for each agent count (4–20) and show success rates.

• Left y-axis  : mean elapsed time ± σ
• Right y-axis : success rate (bars, drawn behind the line)
"""

import argparse
import ast
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

EXPECTED_REPEATS = 10      # planned runs per agent count
MAX_AGENTS       = 20      # highest agent count to show


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────
def gather(folder: Path):
    """Return agents, means, stds, success rates for all counts 4 … MAX_AGENTS."""
    stats: dict[int, tuple[float | None, float, float]] = {}

    for csv in folder.glob("*.csv"):
        df = pd.read_csv(csv)
        n_agents  = int(ast.literal_eval(df["agents_num"].iloc[0])[0])
        finished  = len(df)
        success_rate = (finished / EXPECTED_REPEATS) * 100
        stats[n_agents] = (
            df["elapsed_time"].mean(),
            df["elapsed_time"].std(ddof=0),
            success_rate,
        )

    agents, means, stds, success_rates = [], [], [], []
    for a in range(4, MAX_AGENTS + 1):
        if a in stats:
            m, s, sr = stats[a]
        else:
            m, s, sr = None, 0.0, 0.0
        agents.append(a)
        means.append(m)
        stds.append(s)
        success_rates.append(sr)

    return agents, means, stds, success_rates


# ──────────────────────────────────────────────────────────────────────────────
# Main plotting routine
# ──────────────────────────────────────────────────────────────────────────────
def main(folder: Path) -> None:
    agents, means, stds, success_rates = gather(folder)

    # ── create axes ─────────────────────────────────────────────────────
    fig, ax_time = plt.subplots()     # left axis  (elapsed time)
    ax_sr = ax_time.twinx()           # right axis (success rates)

    # Make the LEFT axis face transparent so the bars are visible behind it
    ax_time.patch.set_alpha(0)        # transparent foreground
    ax_time.set_zorder(2)             # draw this axis LAST
    ax_sr.set_zorder(1)               # background axis

    # ── draw success rate bars on the RIGHT axis (background) ──────────
    ax_sr.bar(
        agents,
        success_rates,
        width=0.6,
        color="green",
        alpha=0.5,
        label="success rate (%)",
        zorder=0,                     # behind the line
    )
    ax_sr.set_ylabel("Success rate (%)")
    ax_sr.set_ylim(0, 100)

    # ── draw elapsed-time line on the LEFT axis (foreground) ───────────
    valid = [i for i, m in enumerate(means) if m is not None]
    ax_time.errorbar(
        [agents[i] for i in valid],
        [means[i]  for i in valid],
        yerr=[stds[i] for i in valid],
        fmt="o-",
        capsize=4,
        label="mean elapsed time",
        zorder=3,
    )
    ax_time.set_ylabel("Average elapsed time (s)")
    ax_time.set_xlabel("Number of agents")
    ax_time.set_xticks(range(4, MAX_AGENTS + 1))
    ax_time.grid(True, linestyle="--", alpha=0.6)

    # ── combined legend ────────────────────────────────────────────────
    h1, l1 = ax_time.get_legend_handles_labels()
    h2, l2 = ax_sr.get_legend_handles_labels()
    ax_time.legend(h1 + h2, l1 + l2, loc="upper left", frameon=False)

    plt.title("Simulation performance vs. agent count for M*")
    fig.tight_layout()

    # ── save plot ───────────────────────────────────────────────────────
    out_file = folder / "avg_elapsed_time"
    if out_file.exists():
        print(f"Warning: {out_file} exists — overwriting.")
    fig.savefig(out_file.with_suffix(".png"), dpi=300)
    fig.savefig(out_file.with_suffix(".eps"), dpi=300)
    print(f"Plot saved to {out_file}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot elapsed time and success rates for agent counts 4-20."
    )
    parser.add_argument("folder", type=Path, help="Folder containing CSV files")
    args = parser.parse_args()

    if not args.folder.is_dir():
        raise SystemExit(f"{args.folder} is not a directory")

    main(args.folder)
