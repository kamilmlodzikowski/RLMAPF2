from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Set, Iterable, Optional
import heapq
import math

# ------------------------------------------------------------
#  Basic grid utilities
# ------------------------------------------------------------
Pos = Tuple[int, int]                  # (row, col)
State = Tuple[Pos, ...]                # joint state = positions of all agents (ordered)

MOVES4: Tuple[Pos, ...] = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))     # stay, N, S, W, E


def manhattan(a: Pos, b: Pos) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def inside(p: Pos, nrows: int, ncols: int) -> bool:
    return 0 <= p[0] < nrows and 0 <= p[1] < ncols


# ------------------------------------------------------------
#  Single-agent A* (used by M* for its low-dim heuristic paths)
# ------------------------------------------------------------
def astar_single(start: Pos, goal: Pos,
                 nrows: int, ncols: int, obstacles: Set[Pos]) -> List[Pos]:
    """Return shortest path (inclusive) for one agent ignoring others."""
    h = lambda p: manhattan(p, goal)
    open_: List[Tuple[int, int, Pos]] = []
    heapq.heappush(open_, (h(start), 0, start))
    came: Dict[Pos, Pos] = {}
    g_cost = {start: 0}

    while open_:
        f, g, cur = heapq.heappop(open_)
        if cur == goal:
            # reconstruct
            path = [cur]
            while cur in came:
                cur = came[cur]
                path.append(cur)
            return path[::-1]

        for d in MOVES4[1:]:                          # cannot stay in single-agent path
            nxt = (cur[0] + d[0], cur[1] + d[1])
            if not inside(nxt, nrows, ncols) or nxt in obstacles:
                continue
            tentative = g + 1
            if tentative < g_cost.get(nxt, math.inf):
                g_cost[nxt] = tentative
                came[nxt] = cur
                heapq.heappush(open_, (tentative + h(nxt), tentative, nxt))
    raise ValueError("No path")


# ------------------------------------------------------------
#  M* node and collision checking
# ------------------------------------------------------------
@dataclass(order=True)
class Node:
    f: int
    g: int = field(compare=False)
    state: State = field(compare=False)
    timestep: int = field(compare=False)
    collision_set: Set[int] = field(compare=False, default_factory=set)
    parent: Optional['Node'] = field(compare=False, default=None)


def edge_conflict(a1: Pos, a2: Pos, b1: Pos, b2: Pos) -> bool:
    """Agents cross each other: a moves a1→a2 while b moves b1→b2."""
    return a1 == b2 and b1 == a2


def detect_collisions(prev: State, nxt: State) -> Set[int]:
    """
    Indices of agents that would collide when moving prev → nxt.
    – Vertex conflict: same cell at the same time-step.
    – Edge swap:  a→b while b→a.
    """
    involved: Set[int] = set()
    n = len(prev)
    for i in range(n):
        for j in range(i + 1, n):
            # vertex
            if nxt[i] == nxt[j]:
                involved.update((i, j))
            # edge swap
            elif prev[i] == nxt[j] and prev[j] == nxt[i]:
                involved.update((i, j))
    return involved
# ------------------------------------------------------------
#  Core M* search
# ------------------------------------------------------------
def mstar(starts: List[Pos], goals: List[Pos],
          nrows: int, ncols: int, obstacles: Set[Pos]) -> List[List[Pos]]:
    nags = len(starts)
    # Pre-compute individual shortest paths & admissible heuristics
    paths_single = [astar_single(s, g, nrows, ncols, obstacles) for s, g in zip(starts, goals)]
    h_values = [len(p) - 1 for p in paths_single]               # distance-to-go at timestep 0
    h0 = sum(h_values)

    start_state: State = tuple(starts)
    open_: List[Node] = []
    heapq.heappush(open_, Node(f=h0, g=0, state=start_state, timestep=0))

    closed_g: Dict[Tuple[State, int], int] = { (start_state, 0): 0 }

    # helper to compute heuristic for joint state (sum of remaining single-agent distances)
    def heuristic(state: State, _t: int) -> int:
        """Admissible for makespan: longest remaining individual distance."""
        return max(manhattan(pos, goals[idx]) for idx, pos in enumerate(state))

    while open_:
        # print("Current open set size:", len(open_))
        # print("Current closed set size:", len(closed_g))
        # print("Current timestep:", open_[0].timestep)
        # print('--' * 20)
        cur = heapq.heappop(open_)

        if cur.state == tuple(goals):
            # ----- Reconstruct per-agent paths -----
            joint_seq: List[State] = []
            node = cur
            while node is not None:
                joint_seq.append(node.state)
                node = node.parent
            joint_seq.reverse()

            # split into individual paths (with waits inserted)
            paths: List[List[Pos]] = [[] for _ in range(nags)]
            for state in joint_seq:
                for i, p in enumerate(state):
                    paths[i].append(p)
            return paths

        # Expand neighbours
        successors = expand_node(cur, obstacles, nrows, ncols)
        for succ_state in successors:
            t_next = cur.timestep + 1
            g_next = cur.g + 1
            key = (succ_state, t_next)
            if g_next >= closed_g.get(key, math.inf):
                continue

            h_next = heuristic(succ_state, t_next)
            f_next = g_next + h_next
            closed_g[key] = g_next
            heapq.heappush(open_, Node(f_next, g_next, succ_state,
                                       t_next, parent=cur))

    raise RuntimeError("Search failed – no solution?!")


# ------------------------------------------------------------
#  Sub-dimensional expansion helper
# ------------------------------------------------------------
def expand_node(node: Node, obstacles: Set[Pos],
                nrows: int, ncols: int) -> List[State]:
    """
    Joint successor generator for M*.

    * Every returned state is guaranteed collision-free.
    * If a NEW collision is detected, enlarge node.collision_set **and**
      immediately retry the expansion in the higher dimension, without
      re-inserting the node into OPEN.  (That avoids the sentinel trick.)
    """
    parents = node.state

    # print("Expanding node:", node.state)

    # Pre-compute allowable moves for each agent
    moves_per_agent: List[List[Pos]] = []
    for p in parents:
        choices = []
        for d in MOVES4:   # stay + N,S,W,E
            nxt = (p[0] + d[0], p[1] + d[1])
            if inside(nxt, nrows, ncols) and nxt not in obstacles:
                choices.append(nxt)
        moves_per_agent.append(choices)

    # Recursive Cartesian product with on-the-fly vertex pruning
    joint: List[Tuple[Pos, ...]] = []

    def dfs(k: int, partial: List[Pos]):
        if k == len(parents):
            joint.append(tuple(partial))
            return
        for nxt in moves_per_agent[k]:
            if nxt in partial:              # vertex clash with fixed agent
                continue
            partial.append(nxt)
            dfs(k + 1, partial)
            partial.pop()

    while True:                       # repeat until **no new** collisions appear
        dfs(0, [])
        children: List[State] = []
        grew = False

        for cand in joint:
            coll = detect_collisions(parents, cand)
            if not coll:                          # ✓ safe successor
                children.append(tuple(cand))
            else:                                 # collision found
                if not coll.issubset(node.collision_set):
                    node.collision_set.update(coll)
                    grew = True                   # need another round
                # discard the colliding successor

        if not grew:                              # no new collisions → done
            return children

        # otherwise:  reset and enumerate again in the higher dimension
        joint.clear()

# ------------------------------------------------------------
# Example usage: 4 robots on a 20×20 grid with obstacles
# ------------------------------------------------------------
if __name__ == "__main__":
    # Grid size
    ROWS, COLS = 20, 20

    # Obstacles: a vertical wall in column 10 with a 1-cell doorway at row 10
    obstacles = {(r, 10) for r in range(ROWS) if r != 10 and r != 9}
    print("Obstacles:", obstacles)

    # Robot start positions (NW, NE, SW, SE “corners” – offset inward by 1)
    # starts = [(1, 1), (18, 18), (18, 1), (1, 18)]
    # goals  = [(18, 18), (1, 1), (1, 18), (18, 1)]
    starts = [(1, 1), (18, 1), (1, 18), (18, 18)]
    goals = [(18, 1), (1, 1), (18, 18), (1, 18)]

    # Print grid with obstacles
    print("Grid with obstacles:")
    for r in range(ROWS):
        for c in range(COLS):
            if (r, c) in obstacles:
                print("X", end=" ")
            elif (r, c) in starts:
                print("S", end=" ")
            else:
                print(".", end=" ")
        print()

    # Run M*
    print("M* pathfinding for 4 robots on a 20×20 grid with obstacles:")
    paths = mstar(starts, goals, ROWS, COLS, obstacles)

    # Pretty-print the results
    for i, p in enumerate(paths):
        print(f"Robot {i}: {p}")
    print(f"\nMakespan (timesteps until all finished): {max(len(p) for p in paths) - 1}")

    robot_grids = {i: [['.' for _ in range(COLS)] for _ in range(ROWS)] for i in range(len(paths))}
    
    print("\nFinal grid with paths for robots:")
    for i, p in enumerate(paths):
        print(f"Robot {i}: {p}")
        for pos in p:
            robot_grids[i][pos[0]][pos[1]] = str(i)
        for r in range(ROWS):
            for c in range(COLS):
                if robot_grids[i][r][c] == str(i):
                    print(f"\033[1;3{i+1}m{i}\033[0m", end=" ")
                elif (r, c) in obstacles:
                    print("X", end=" ")
                else:
                    print(robot_grids[i][r][c], end=" ")
            print()

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
            joint_paths = mstar(self.all_starts, self.all_goals, self.nrows, self.ncols, self.obstacles)
            self.path = joint_paths[self.agent_id]
            self.path_idx = 0
        except Exception as e:
            self.path = [obs['current_position']]
            self.path_idx = 0

    def act(self, obs):
        """
        Returns the next action (int) for the agent.
        """
        if self.path is None or self.path_idx >= len(self.path) - 1:
            return 4  # Wait
        cur = self.path[self.path_idx]
        nxt = self.path[self.path_idx + 1]
        move = (nxt[0] - cur[0], nxt[1] - cur[1])
        action = self.ACTIONS.get(move, 4)
        self.path_idx += 1
        return action