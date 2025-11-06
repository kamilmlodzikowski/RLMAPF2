"""
Utility script to run the modified D* Lite planner on an RLMAPF2 JSON map.

This script loads a single map variant, performs iterative congestion-aware
replanning using the `iterative_congestion_d_star` logic with cumulative, per-agent
penalties (excluding each agent’s own prior routes), and optionally visualizes
per-iteration paths to inspect how weights shift the resulting routes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from d_star_lite import DStarLite, OBSTACLE, OccupancyGridMap


AgentId = str
GridPos = Tuple[int, int]


def calculate_overlap_percentages(
    paths: Dict[AgentId, List[GridPos]]
) -> Tuple[Dict[AgentId, float], float, int, float]:
    """Compute per-agent and overall overlap percentages for the supplied paths."""
    if not paths:
        return {}, 0.0, 0, 0.0

    path_sets: Dict[AgentId, set[GridPos]] = {agent: set(path) for agent, path in paths.items()}

    # Per-agent overlap relative to other agents
    per_agent: Dict[AgentId, float] = {}
    for agent, cells in path_sets.items():
        if not cells:
            per_agent[agent] = 0.0
            continue

        other_union = set().union(*(path_sets[a] for a in path_sets if a != agent))
        overlap_count = len(cells & other_union)
        per_agent[agent] = 100.0 * overlap_count / len(cells)

    # Overall overlap relative to total unique path cells
    cell_counts: Dict[GridPos, int] = {}
    for cells in path_sets.values():
        for cell in cells:
            cell_counts[cell] = cell_counts.get(cell, 0) + 1

    overlapped_cells = sum(1 for count in cell_counts.values() if count > 1)
    total_unique_cells = len(cell_counts)
    overall = 100.0 * overlapped_cells / total_unique_cells if total_unique_cells else 0.0
    average = float(np.mean(list(per_agent.values()))) if per_agent else 0.0

    return per_agent, overall, overlapped_cells, average


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run D* Lite with congestion iterations on a specific RLMAPF2 map."
    )
    parser.add_argument(
        "--map",
        type=Path,
        default=Path("maps/crashtest_2-2a-5x15.json"),
        help="Path to map JSON file (default: %(default)s).",
    )
    parser.add_argument(
        "--variant",
        type=int,
        default=None,
        help="Map variant id to use. Defaults to the first available variant.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Number of D* Lite congestion iterations to perform (>=1).",
    )
    parser.add_argument(
        "--congestion-weight",
        type=float,
        default=1.0,
        help="Multiplier applied to congestion counts when updating traversal costs.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Display matplotlib visualization of each iteration.",
    )
    parser.add_argument(
        "--pause",
        type=float,
        default=1.0,
        help="Seconds to pause between frames when visualizing (default: %(default)s).",
    )
    parser.add_argument(
        "--no-block",
        action="store_true",
        help="When set with --visualize, close the figure after the final frame instead of blocking.",
    )
    parser.add_argument(
        "--save-frames",
        type=Path,
        default=None,
        help="Optional directory to save per-iteration PNG frames (does not require --visualize).",
    )
    return parser.parse_args()


def load_map(
    map_path: Path,
    variant_id: int | None,
) -> Tuple[int, int, np.ndarray, Dict[AgentId, GridPos], Dict[AgentId, GridPos]]:
    with map_path.open("r") as fh:
        data = json.load(fh)

    width = int(data["metadata"]["width"])
    height = int(data["metadata"]["height"])

    variants = data["map_variant"]
    variant_key = None
    if variant_id is not None:
        variant_key = str(variant_id)
        if variant_key not in variants:
            available = ", ".join(sorted(variants.keys()))
            raise ValueError(f"Variant {variant_id} not found. Available variants: {available}")
    else:
        variant_key = sorted(variants.keys(), key=int)[0]

    variant = variants[variant_key]

    obstacle_grid = np.zeros((width, height), dtype=np.uint8)
    for x, y in variant.get("obstacles", []):
        obstacle_grid[x, y] = OBSTACLE

    starts_raw = variant.get("starting_positions", {})
    goals_raw = variant.get("goal_positions", {})

    if set(starts_raw.keys()) != set(goals_raw.keys()):
        raise ValueError("Starting and goal position keys do not match in the chosen variant.")

    starts = {str(agent): tuple(int(v) for v in pos) for agent, pos in starts_raw.items()}
    goals = {str(agent): tuple(int(v) for v in pos) for agent, pos in goals_raw.items()}

    return width, height, obstacle_grid, starts, goals


def run_iterative_planning(
    x_dim: int,
    y_dim: int,
    obstacle_grid: np.ndarray,
    starts: Dict[AgentId, GridPos],
    goals: Dict[AgentId, GridPos],
    iterations: int,
    congestion_weight: float,
) -> List[Dict]:
    if iterations < 1:
        raise ValueError("iterations must be at least 1")

    base_map = OccupancyGridMap(x_dim, y_dim)
    base_map.set_map(obstacle_grid)

    traversal_costs = np.ones((x_dim, y_dim), dtype=np.float32)
    history: List[Dict] = []
    agent_penalties: Dict[AgentId, np.ndarray] = {
        agent: np.zeros((x_dim, y_dim), dtype=np.float32) for agent in starts
    }

    for iteration_idx in range(iterations):
        base_map.set_traversal_costs(traversal_costs)

        paths: Dict[AgentId, List[GridPos]] = {}
        for agent_id, start in starts.items():
            if congestion_weight != 0.0:
                prior_penalty = agent_penalties[agent_id]
                if prior_penalty.any():
                    traversal_costs -= prior_penalty
                    prior_penalty.fill(0.0)
                    base_map.set_traversal_costs(traversal_costs)

            goal = goals[agent_id]
            planner = DStarLite(base_map, start, goal)
            path, _, _ = planner.move_and_replan(start)
            paths[agent_id] = path

            if congestion_weight != 0.0:
                penalty = agent_penalties[agent_id]
                penalty.fill(0.0)
                for cell in path:
                    penalty[cell] += congestion_weight
                traversal_costs += penalty
                base_map.set_traversal_costs(traversal_costs)

        if congestion_weight == 0.0:
            base_map.set_traversal_costs(traversal_costs)

        history.append(
            {
                "iteration": iteration_idx + 1,
                "paths": paths,
                "traversal_costs": base_map.traversal_costs.copy(),
            }
        )

    return history


def visualize_iterations(
    x_dim: int,
    y_dim: int,
    obstacle_grid: np.ndarray,
    starts: Dict[AgentId, GridPos],
    goals: Dict[AgentId, GridPos],
    history: List[Dict],
    display: bool,
    pause_seconds: float,
    save_dir: Path | None,
    block: bool,
) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for visualization. Install it or run without --visualize."
        ) from exc

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    agent_ids = sorted(starts.keys())
    color_map = plt.colormaps.get_cmap("tab10")

    if display:
        plt.ion()
    fig, ax = plt.subplots(figsize=(6, 6))
    cbar = None

    for idx, record in enumerate(history, start=1):
        ax.cla()
        costs = record["traversal_costs"].T

        im = ax.imshow(costs, origin="lower", cmap="viridis", interpolation="none")
        if cbar is None:
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Traversal multiplier")
        else:
            cbar.update_normal(im)

        obstacle_positions = np.argwhere(obstacle_grid == OBSTACLE)
        for ox, oy in obstacle_positions:
            rect = Rectangle(
                (ox - 0.5, oy - 0.5),
                1.0,
                1.0,
                facecolor="dimgray",
                edgecolor="black",
            )
            ax.add_patch(rect)

        for agent_idx, agent_id in enumerate(agent_ids):
            path = record["paths"][agent_id]
            color = color_map(agent_idx % color_map.N)
            xs = [cell[0] for cell in path]
            ys = [cell[1] for cell in path]
            ax.plot(xs, ys, "-o", color=color, linewidth=2, markersize=4, label=agent_id)

            start = starts[agent_id]
            goal = goals[agent_id]
            ax.scatter(start[0], start[1], marker="s", s=80, color=color, edgecolor="black")
            ax.scatter(goal[0], goal[1], marker="*", s=120, color=color, edgecolor="black")

        ax.set_title(f"Iteration {idx} / {len(history)}")
        ax.set_xlim(-0.5, x_dim - 0.5)
        ax.set_ylim(-0.5, y_dim - 0.5)
        ax.set_aspect("equal")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.legend(loc="upper right")
        ax.grid(False)

        fig.canvas.draw()
        if display:
            fig.canvas.flush_events()
            pause = max(0.0, pause_seconds)
            plt.pause(pause if pause > 0 else 0.001)

        if save_dir is not None:
            output_path = save_dir / f"iteration_{idx:02d}.png"
            fig.savefig(output_path, dpi=200, bbox_inches="tight")

    if display:
        plt.ioff()
        if block:
            plt.show()
        else:
            plt.close(fig)
    else:
        plt.close(fig)


def main() -> None:
    args = parse_args()

    width, height, obstacle_grid, starts, goals = load_map(args.map, args.variant)

    print(f"Loaded map {args.map} (variant={args.variant if args.variant is not None else 'auto'})")
    print(f"Grid size: {width} x {height}, agents: {len(starts)}")
    print(f"Running {args.iterations} iteration(s) with congestion weight {args.congestion_weight}")

    history = run_iterative_planning(
        width,
        height,
        obstacle_grid,
        starts,
        goals,
        iterations=args.iterations,
        congestion_weight=args.congestion_weight,
    )

    for record in history:
        iteration = record["iteration"]
        traversal_costs = record["traversal_costs"]
        min_cost = float(np.min(traversal_costs))
        max_cost = float(np.max(traversal_costs))
        print(f"\nIteration {iteration}: traversal multiplier range [{min_cost:.2f}, {max_cost:.2f}]")
        for agent_id, path in record["paths"].items():
            length = len(path)
            if length > 0:
                print(f"  {agent_id}: path length {length}, start {path[0]} -> goal {path[-1]}")
            else:
                print(f"  {agent_id}: path length {length}, empty path")
        per_agent_overlap, overall_overlap, total_overlap_cells, avg_overlap = calculate_overlap_percentages(record["paths"])
        formatted_agents = ", ".join(
            f"{agent}: {pct:.1f}%" for agent, pct in sorted(per_agent_overlap.items())
        )
        print(
            f"  Overlap: {total_overlap_cells} cells ({overall_overlap:.1f}%) "
            f"| avg {avg_overlap:.1f}% | {formatted_agents}"
        )

    if args.visualize or args.save_frames is not None:
        visualize_iterations(
            width,
            height,
            obstacle_grid,
            starts,
            goals,
            history,
            display=args.visualize,
            pause_seconds=args.pause,
            save_dir=args.save_frames,
            block=(not args.no_block) if args.visualize else False,
        )


if __name__ == "__main__":
    main()
