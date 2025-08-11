from __future__ import annotations

class MStarAgent:
    """
    Mimics an RL agent interface: given an observation (with agent position, goal, obstacles),
    computes a path using M*, and returns the next action (0=Up, 1=Down, 2=Left, 3=Right, 4=Wait).
    """
    ACTIONS = {
        (0, -1): 0,  # Up
        (0, 1): 1,   # Down
        (-1, 0): 2,  # Left
        (1, 0): 3,   # Right
        (0, 0): 4,   # Wait
    }

    def __init__(self, agent_id, grid_size, all_starts, all_goals, obstacles):
        """
        agent_id: index of this agent in the joint state
        grid_size: (nrows, ncols)
        all_starts: list of all agent start positions
        all_goals: list of all agent goal positions
        obstacles: set of (row, col)
        """
        self.agent_id = agent_id
        self.nrows, self.ncols = grid_size
        self.all_starts = list(all_starts)
        self.all_goals = list(all_goals)
        self.obstacles = set(obstacles)
        self.path = None
        self.path_idx = 0

    def reset(self, obs):
        """
        obs: dict with keys 'current_position', 'goal_position', 'obstacles'
        """
        # Update own start/goal in joint state
        self.all_starts[self.agent_id] = obs['current_position']
        self.all_goals[self.agent_id] = obs['goal_position']
        self.obstacles = set(obs['obstacles'])
        # Compute joint path
        try:
            # Convert to M* format
            agent_positions = {i: pos for i, pos in enumerate(self.all_starts)}
            agent_goals = {i: pos for i, pos in enumerate(self.all_goals)}
            
            joint_paths = mstar(
                obstacles=self.obstacles,
                agent_positions=agent_positions,
                agent_goals=agent_goals,
                R=self.nrows,
                C=self.ncols
            )
            self.path = joint_paths[self.agent_id]
            self.path_idx = 0
        except Exception as e:
            raise RuntimeError(f"Failed to compute path for agent {self.agent_id}: {e}")

    def act(self, obs, current_position):
        """
        Returns the next action (int) for the agent.
        current_position: the agent's current (row, col) position
        """
        if self.path is None:
            raise RuntimeError("Path not computed or exhausted")
        # Only update path_idx if current_position matches the next step in the path
        if self.path_idx >= len(self.path):
            return 4  # Path exhausted, wait

        # If current_position does not match where we think we are, try to sync up
        if current_position != self.path[self.path_idx]:
            # Try to find the closest matching index at or after current path_idx
            try:
                # Only search forward to avoid going back in time
                idx = self.path.index(current_position, self.path_idx)
                self.path_idx = idx
            except ValueError:
                print(f"Warning: Current position {current_position} not in path, defaulting to wait.")
                return 4

        if self.path_idx >= len(self.path) - 1:
            # Reached the end of the path, wait at the goal
            return 4

        cur = self.path[self.path_idx]
        nxt = self.path[self.path_idx + 1]
        move = (nxt[0] - cur[0], nxt[1] - cur[1])
        action = self.ACTIONS.get(move, 4)
        # Only advance path_idx if the agent is moving to the next step
        if current_position == nxt:
            self.path_idx += 1
        elif current_position == cur:
            # If waiting at the current position, only advance if next step is also the same
            if nxt == cur:
                self.path_idx += 1
        # if action == 4:
        #     print(f"Agent {self.agent_id} is waiting at {cur}.")
        #     print(f"Full path: {self.path}")
        # else:
        #     print(f"Agent {self.agent_id} moving from {cur} to {nxt} with action {action}.")
        return action
    
###############################################################################
# M* multi-agent path-finding (independence-detection variant)                #
# Works directly with the MStarAgent class given in the prompt.               #
###############################################################################
import sys
import time
from typing import NamedTuple, Dict, Set, Tuple, List
from heapq import heappush, heappop
from collections import defaultdict
from itertools import product, combinations
from typing import Dict, List, Tuple, Set, FrozenSet
from concurrent.futures import ThreadPoolExecutor, as_completed

Pos         = Tuple[int, int]          # (row, col)
Path        = List[Pos]                # list of positions, one per time-step
PathsDict   = Dict[int, Path]          # agent_id → Path
ObstacleMap = Dict[int, Set[Pos]]      # time → occupied cells

# --------------------------------------------------------------------------- #
# Low-level utilities                                                         #
# --------------------------------------------------------------------------- #

MOVES: Tuple[Pos, ...] = ((0, 0), (0, 1), (0, -1), (1, 0), (-1, 0))  # Wait, D, U, R, L

def manhattan(a: Pos, b: Pos) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def in_bounds(p: Pos, R: int, C: int) -> bool:
    return 0 <= p[0] < R and 0 <= p[1] < C

def pad_path(path: Path, length: int) -> Path:
    """Extend `path` with goal repetitions so that len(path) == length."""
    if len(path) >= length:
        return path
    goal = path[-1]
    return path + [goal] * (length - len(path))

# --------------------------------------------------------------------------- #
# Single-agent space-time A* (with dynamic obstacles coming from other paths) #
# --------------------------------------------------------------------------- #

def astar_single(start: Pos, goal: Pos,
                 static_obstacles: Set[Pos],
                 dynamic_obstacles: ObstacleMap,
                 R: int, C: int,
                 time_horizon: int = 512) -> Path | None:
    """
    4-connected grid A* in space-time:
      • waits are allowed,
      • cannot step onto static obstacles,
      • cannot step onto a cell occupied by another agent at the same time,
      • cannot swap edges with another agent between consecutive time-steps.
    """
    Node = Tuple[int, int, Pos, Pos]  # (f = g+h, g, pos, prev_pos)
    open_list: List[Node] = []
    heappush(open_list, (manhattan(start, goal), 0, start, start))
    came_from: Dict[Tuple[Pos, int], Pos] = {(start, 0): start}
    g_cost: Dict[Tuple[Pos, int], int] = {(start, 0): 0}

    while open_list:
        f, g, current, prev = heappop(open_list)

        if current == goal:
            # Reconstruct path (stop when current == start)
            t = g
            path: Path = [current]
            while (current, t) != (start, 0):
                current = came_from[(current, t)]
                t -= 1
                path.append(current)
            return list(reversed(path))

        if g >= time_horizon:
            continue  # give up on this branch

        t_next = g + 1
        for d_row, d_col in MOVES:
            nxt = (current[0] + d_row, current[1] + d_col)
            if not in_bounds(nxt, R, C) or nxt in static_obstacles:
                continue

            # Dynamic conflicts: vertex & edge
            if nxt in dynamic_obstacles.get(t_next, set()):
                continue
            if (nxt == prev and current in dynamic_obstacles.get(t_next, set())):
                continue  # edge swap with another agent

            key = (nxt, t_next)
            new_g = g + 1
            if key not in g_cost or new_g < g_cost[key]:
                g_cost[key] = new_g
                h = manhattan(nxt, goal)
                heappush(open_list, (new_g + h, new_g, nxt, current))
                came_from[key] = current
    return None  # no path found within horizon

# --------------------------------------------------------------------------- #
# Joint A* for a *small* coupled set of agents                                #
# --------------------------------------------------------------------------- #

def astar_joint(group: Tuple[int, ...],
                starts: Dict[int, Pos],
                goals: Dict[int, Pos],
                static_obstacles: Set[Pos],
                other_paths: PathsDict,
                R: int, C: int,
                time_horizon: int = 512) -> Dict[int, Path] | None:
    """
    Joint A* in the *multi-agent* space-time of the coupled `group`
    (size is expected to be tiny – typically ≤3 in practice).
    """
    # Helper to extract a group's joint position from a full tuple
    def joint_heuristic(joint_pos: Tuple[Pos, ...]) -> int:
        return sum(manhattan(p, goals[a]) for a, p in zip(group, joint_pos))

    # Pre-compute dynamic obstacles from already-fixed other agents
    dyn: ObstacleMap = defaultdict(set)
    if other_paths:
        L_max = max(len(p) for p in other_paths.values())
        for t in range(L_max):
            for pos in (p[t] if t < len(p) else p[-1] for p in other_paths.values()):
                dyn[t].add(pos)

    Node = Tuple[int, int, Tuple[Pos, ...]]
    open_list: List[Node] = []
    start_joint = tuple(starts[a] for a in group)
    goal_joint = tuple(goals[a] for a in group)
    heappush(open_list, (joint_heuristic(start_joint), 0, start_joint))
    came_from: Dict[Tuple[Tuple[Pos, ...], int], Tuple[Pos, ...]] = {(start_joint, 0): start_joint}
    g_cost: Dict[Tuple[Tuple[Pos, ...], int], int]          = {(start_joint, 0): 0}
    start_time = time.time()
    while open_list:
        if time.time() - start_time > 120:  # timeout after X seconds
            raise ValueError("Joint A* timed out")
        f, g, joint = heappop(open_list)
        if joint == goal_joint:
            # Reconstruct per-agent paths
            t = g
            joint_path: List[Tuple[Pos, ...]] = [joint]
            while (joint, t) != (start_joint, 0):
                joint = came_from[(joint, t)]
                t -= 1
                joint_path.append(joint)
            joint_path = list(reversed(joint_path))

            # Split back into individual paths and pad them equally
            paths: Dict[int, Path] = {a: [] for a in group}
            for jp in joint_path:
                for a, p in zip(group, jp):
                    paths[a].append(p)
            return paths

        if g >= time_horizon:
            continue

        t_next = g + 1
        # Cartesian product of moves for *all* group members
        for moves in product(MOVES, repeat=len(group)):
            nxt_positions = []
            valid = True

            # Apply moves one by one to allow cheap pruning
            for idx, (a, (d_row, d_col)) in enumerate(zip(group, moves)):
                cur = joint[idx]
                nxt = (cur[0] + d_row, cur[1] + d_col)
                if (not in_bounds(nxt, R, C) or
                        nxt in static_obstacles or
                        nxt in dyn.get(t_next, set())):
                    valid = False
                    break
                nxt_positions.append(nxt)

            if not valid:
                continue

            # Internal vertex conflicts
            if len(set(nxt_positions)) != len(nxt_positions):
                continue
            # Internal edge conflicts (swap)
            swapped = False
            for (i, pi), (j, pj) in combinations(enumerate(joint), 2):
                if pi == nxt_positions[j] and pj == nxt_positions[i]:
                    swapped = True
                    break
            if swapped:
                continue

            nxt_joint = tuple(nxt_positions)
            key = (nxt_joint, t_next)
            new_g = g + 1
            if key not in g_cost or new_g < g_cost[key]:
                g_cost[key] = new_g
                h = joint_heuristic(nxt_joint)
                heappush(open_list, (new_g + h, new_g, nxt_joint))
                came_from[key] = joint
    return None  # no joint solution

# --------------------------------------------------------------------------- #
# Conflict detection utilities                                                #
# --------------------------------------------------------------------------- #

class Conflict(NamedTuple):
    a1: int
    a2: int
    time: int
    kind: str  # 'vertex' or 'edge'

def first_conflict(paths: PathsDict) -> Conflict | None:
    """Return the earliest conflict between any two agents, or None."""
    if not paths:
        return None
    horizon = max(len(p) for p in paths.values())
    for t in range(horizon):
        positions: Dict[Pos, int] = {}
        # --- Vertex conflicts ------------------------------------------------
        for a, path in paths.items():
            pos = path[t] if t < len(path) else path[-1]
            if pos in positions:
                return Conflict(positions[pos], a, t, 'vertex')
            positions[pos] = a

        # --- Edge conflicts --------------------------------------------------
        for a1, a2 in combinations(paths.keys(), 2):
            pos1_t     = paths[a1][t] if t < len(paths[a1]) else paths[a1][-1]
            pos2_t     = paths[a2][t] if t < len(paths[a2]) else paths[a2][-1]
            pos1_t1    = paths[a1][t + 1] if t + 1 < len(paths[a1]) else paths[a1][-1]
            pos2_t1    = paths[a2][t + 1] if t + 1 < len(paths[a2]) else paths[a2][-1]
            if pos1_t == pos2_t1 and pos2_t == pos1_t1:
                return Conflict(a1, a2, t + 1, 'edge')
    return None

# --------------------------------------------------------------------------- #
# Top-level M* planner                                                        #
# --------------------------------------------------------------------------- #

def mstar(obstacles: Set[Pos],
          agent_positions: Dict[int, Pos],
          agent_goals: Dict[int, Pos],
          R: int, C: int,
          time_horizon: int = 512) -> PathsDict:
    """
    Compute collision-free paths for *all* agents with the M* algorithm.

    Returns
    -------
    dict : agent_id → Path (list of (row, col) including both start and goal)
    """
    groups: Dict[int, Set[int]] = {i: {i} for i in agent_positions}
    belongs_to: Dict[int, int] = {i: i for i in agent_positions}
    paths: PathsDict = {}

    def compute_group_path(gid: int) -> Tuple[int, Dict[int, Path]]:
        group = tuple(sorted(groups[gid]))
        others_paths = {a: p for a, p in paths.items() if a not in group}
        if len(group) == 1:
            a = group[0]
            path = astar_single(start=agent_positions[a],
                                goal=agent_goals[a],
                                static_obstacles=obstacles,
                                dynamic_obstacles=paths_to_dyn(others_paths),
                                R=R, C=C, time_horizon=time_horizon)
            if path is None:
                raise RuntimeError(f"No path for agent {a}")
            return (a, {a: path})
        else:
            res = astar_joint(group=group,
                              starts=agent_positions,
                              goals=agent_goals,
                              static_obstacles=obstacles,
                              other_paths=others_paths,
                              R=R, C=C, time_horizon=time_horizon)
            if res is not None:
                return (None, res)
            raise RuntimeError(f"No joint path for group {group}.")

    def paths_to_dyn(pths: PathsDict) -> ObstacleMap:
        dyn: ObstacleMap = defaultdict(set)
        if not pths:
            return dyn
        Lmax = max(len(p) for p in pths.values())
        for t in range(Lmax):
            for pos in (p[t] if t < len(p) else p[-1] for p in pths.values()):
                dyn[t].add(pos)
        return dyn

    # -------- Initial individual planning (multi-threaded) ------------------
    with ThreadPoolExecutor() as executor:
        future_to_gid = {executor.submit(compute_group_path, gid): gid for gid in list(groups)}
        for future in as_completed(future_to_gid):
            gid = future_to_gid[future]
            try:
                result = future.result()
                if result[0] is not None:
                    # Single agent
                    paths.update(result[1])
                else:
                    # Multi-agent group
                    paths.update(result[1])
            except Exception as exc:
                raise RuntimeError(f"Initial planning failed for group {gid}: {exc}")

    # -------- Resolve conflicts iteratively (multi-threaded for all groups) ----
    while True:
        conflict = first_conflict(paths)
        if conflict is None:
            break  # success!
        g1 = belongs_to[conflict.a1]
        g2 = belongs_to[conflict.a2]
        if g1 == g2:
            raise RuntimeError(f"Unable to resolve internal conflict in group {groups[g1]}")
        merged_agents = groups[g1] | groups[g2]
        new_gid = min(g1, g2)
        groups[new_gid] = merged_agents
        for a in merged_agents:
            belongs_to[a] = new_gid
        if g1 != new_gid:
            groups.pop(g1, None)
        if g2 != new_gid:
            groups.pop(g2, None)
        # Re-plan for all groups in parallel after a merge
        with ThreadPoolExecutor() as executor:
            future_to_gid = {executor.submit(compute_group_path, gid): gid for gid in list(groups)}
            new_paths: Dict[int, Path] = {}
            for future in as_completed(future_to_gid):
                gid = future_to_gid[future]
                try:
                    result = future.result()
                    if result[0] is not None:
                        new_paths.update(result[1])
                    else:
                        new_paths.update(result[1])
                except Exception as exc:
                    raise RuntimeError(f"Planning failed for group {gid}: {exc}")
            paths.update(new_paths)

    # -------- Pad all paths to the same length ------------------------------
    horizon = max(len(p) for p in paths.values())
    for a, p in paths.items():
        paths[a] = pad_path(p, horizon)

    # -------- Print some statistics ----------------------------------------
    # print("M* paths computed successfully.")
    
    # paths_for_len = paths.copy()
    # # Remove duplicates at the end of each path
    # for a, p in paths_for_len.items():
    #     if len(p) > 1:
    #         # Find the last unique position
    #         last_unique = p[-1]
    #         for i in range(len(p) - 2, -1, -1):
    #             if p[i] != last_unique:
    #                 break
    #             last_unique = p[i]
    #         paths_for_len[a] = p[:i + 1]
    # path_lengths = {a: len(p) for a, p in paths_for_len.items()}
    # print(f"Path lengths: {path_lengths}")

    # Return the final paths
    return paths

# ─────────────────────────────────────────────────────────────────────────────
#  ASCII animation utilities
# ─────────────────────────────────────────────────────────────────────────────

FPS = 4  # frames per second

def _render_frame(grid: List[str], positions: Dict[int, Tuple[int, int]]):
    buf = [list(row) for row in grid]
    for idx, (agent, (r, c)) in enumerate(positions.items()):
        buf[r][c] = chr(ord('A') + agent)
    return "\n".join("".join(row) for row in buf)

def animate(grid: List[str], paths: Dict[int, Path]):
    horizon = max(len(p) for p in paths.values())
    agent_ids = sorted(paths.keys())
    for t in range(horizon):
        positions = {agent: paths[agent][min(t, len(paths[agent]) - 1)] for agent in agent_ids}
        sys.stdout.write("\x1b[H\x1b[2J" + _render_frame(grid, positions) + f"\n\nStep {t}\n")
        sys.stdout.flush()
        time.sleep(1 / FPS)

# ─────────────────────────────────────────────────────────────────────────────
#  Quick demo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    obstacles = {(2, 2), (3, 2), (4, 2), (5, 4)}
    agent_positions = {0: (0, 0), 1: (7, 7), 2: (2, 3), 3: (5, 6), 4: (1, 1)}
    agent_goals = {0: (7, 0), 1: (0, 7), 2: (2, 6), 3: (5, 1), 4: (1, 7)}
    R, C = 8, 8

    try:
        plan = mstar(obstacles, agent_positions, agent_goals, R, C)
    except ValueError as e:
        print("Path planning failed:", e)
        sys.exit(1)

    print("Computed plan – animating... (Ctrl+C to quit)")
    print("Paths:")
    for agent, path in plan.items():
        print(f"Agent {agent}: {path}")
    # time.sleep(1)
    # Convert back to grid for animation
    grid = [['.' for _ in range(C)] for _ in range(R)]
    for r, c in obstacles:
        grid[r][c] = '#'
    grid_strings = [''.join(row) for row in grid]
    animate(grid_strings, plan)
    # time.sleep(1)
    # Convert back to grid for animation
    grid = [['.' for _ in range(C)] for _ in range(R)]
    for r, c in obstacles:
        grid[r][c] = '#'
    grid_strings = [''.join(row) for row in grid]
    animate(grid_strings, plan)
