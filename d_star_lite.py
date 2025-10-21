# FROM https://github.com/Sollimann/Dstar-lite-pathplanner

import numpy as np
from typing import Dict, List, Tuple

OBSTACLE = 255
UNOCCUPIED = 0

class Vertex:
    def __init__(self, pos: (int, int)):
        self.pos = pos
        self.edges_and_costs = {}

    def add_edge_with_cost(self, succ: (int, int), cost: float):
        if succ != self.pos:
            self.edges_and_costs[succ] = cost

    @property
    def edges_and_c_old(self):
        return self.edges_and_costs

class Vertices:
    def __init__(self):
        self.list = []

    def add_vertex(self, v: Vertex):
        self.list.append(v)

    @property
    def vertices(self):
        return self.list


class OccupancyGridMap:
    def __init__(self, x_dim, y_dim, exploration_setting='4N'):
        """
        set initial values for the map occupancy grid
        |----------> y, column
        |           (x=0,y=2)
        |
        V (x=2, y=0)
        x, row
        :param x_dim: dimension in the x direction
        :param y_dim: dimension in the y direction
        """
        self.x_dim = x_dim
        self.y_dim = y_dim

        # the map extents in units [m]
        self.map_extents = (x_dim, y_dim)

        # the obstacle map
        self.occupancy_grid_map = np.zeros(self.map_extents, dtype=np.uint8)

        # traversal cost multiplier per cell (1.0 = baseline cost)
        self.traversal_costs = np.ones(self.map_extents, dtype=np.float32)

        # obstacles
        self.visited = {}
        self.exploration_setting = exploration_setting

    def get_map(self):
        """
        :return: return the current occupancy grid map
        """
        return self.occupancy_grid_map

    def set_map(self, new_ogrid):
        """
        :param new_ogrid:
        :return: None
        """
        if new_ogrid.shape != self.map_extents:
            raise ValueError("New occupancy grid shape {} does not match map extents {}".format(new_ogrid.shape, self.map_extents))
        self.occupancy_grid_map = new_ogrid

    def set_traversal_costs(self, new_costs: np.ndarray):
        """
        :param new_costs: array of same shape as map with traversal multipliers
        """
        if new_costs.shape != self.map_extents:
            raise ValueError("Traversal cost shape {} does not match map extents {}".format(new_costs.shape, self.map_extents))
        self.traversal_costs = new_costs.astype(np.float32)

    def reset_traversal_costs(self):
        """Reset traversal multipliers back to 1.0 everywhere."""
        self.traversal_costs.fill(1.0)

    def get_traversal_cost(self, pos: (int, int)) -> float:
        """Return traversal multiplier for cell `pos`."""
        (x, y) = (round(pos[0]), round(pos[1]))
        return float(self.traversal_costs[x, y])

    def is_unoccupied(self, pos: (int, int)) -> bool:
        """
        :param pos: cell position we wish to check
        :return: True if cell is occupied with obstacle, False else
        """
        (x, y) = (round(pos[0]), round(pos[1]))  # make sure pos is int
        (row, col) = (x, y)

        # if not self.in_bounds(cell=(x, y)):
        #    raise IndexError("Map index out of bounds")

        return self.occupancy_grid_map[row][col] == UNOCCUPIED

    def in_bounds(self, cell: (int, int)) -> bool:
        """
        Checks if the provided coordinates are within
        the bounds of the grid map
        :param cell: cell position (x,y)
        :return: True if within bounds, False else
        """
        (x, y) = cell
        return 0 <= x < self.x_dim and 0 <= y < self.y_dim

    def filter(self, neighbors: List, avoid_obstacles: bool):
        """
        :param neighbors: list of potential neighbors before filtering
        :param avoid_obstacles: if True, filter out obstacle cells in the list
        :return:
        """
        if avoid_obstacles:
            return [node for node in neighbors if self.in_bounds(node) and self.is_unoccupied(node)]
        return [node for node in neighbors if self.in_bounds(node)]

    def succ(self, vertex: (int, int), avoid_obstacles: bool = False) -> list:
        """
        :param avoid_obstacles:
        :param vertex: vertex you want to find direct successors from
        :return:
        """
        (x, y) = vertex

        if self.exploration_setting == '4N':  # change this
            movements = get_movements_4n(x=x, y=y)
        else:
            movements = get_movements_8n(x=x, y=y)

        # not needed. Just makes aesthetics to the path
        if (x + y) % 2 == 0: movements.reverse()

        filtered_movements = self.filter(neighbors=movements, avoid_obstacles=avoid_obstacles)
        return list(filtered_movements)

    def set_obstacle(self, pos: (int, int)):
        """
        :param pos: cell position we wish to set obstacle
        :return: None
        """
        (x, y) = (round(pos[0]), round(pos[1]))  # make sure pos is int
        (row, col) = (x, y)
        self.occupancy_grid_map[row, col] = OBSTACLE

    def remove_obstacle(self, pos: (int, int)):
        """
        :param pos: position of obstacle
        :return: None
        """
        (x, y) = (round(pos[0]), round(pos[1]))  # make sure pos is int
        (row, col) = (x, y)
        self.occupancy_grid_map[row, col] = UNOCCUPIED

    def local_observation(self, global_position: (int, int), view_range: int = 2) -> Dict:
        """
        :param global_position: position of robot in the global map frame
        :param view_range: how far ahead we should look
        :return: dictionary of new observations
        """
        (px, py) = global_position
        nodes = [(x, y) for x in range(px - view_range, px + view_range + 1)
                 for y in range(py - view_range, py + view_range + 1)
                 if self.in_bounds((x, y))]
        return {node: UNOCCUPIED if self.is_unoccupied(pos=node) else OBSTACLE for node in nodes}


class DStarLite:
    def __init__(self, map: OccupancyGridMap, s_start: (int, int), s_goal: (int, int)):
        """
        :param map: the ground truth map of the environment provided by gui
        :param s_start: start location
        :param s_goal: end location
        """
        self.new_edges_and_old_costs = None

        # algorithm start
        self.s_start = s_start
        self.s_goal = s_goal
        self.s_last = s_start
        self.k_m = 0  # accumulation
        self.U = PriorityQueue()
        self.rhs = np.ones((map.x_dim, map.y_dim)) * np.inf
        self.g = self.rhs.copy()

        self.sensed_map = map

        self.rhs[self.s_goal] = 0
        self.U.insert(self.s_goal, Priority(heuristic(self.s_start, self.s_goal), 0))

    def calculate_key(self, s: (int, int)):
        """
        :param s: the vertex we want to calculate key
        :return: Priority class of the two keys
        """
        k1 = min(self.g[s], self.rhs[s]) + heuristic(self.s_start, s) + self.k_m
        k2 = min(self.g[s], self.rhs[s])
        return Priority(k1, k2)

    def c(self, u: (int, int), v: (int, int)) -> float:
        """
        calcuclate the cost between nodes
        :param u: from vertex
        :param v: to vertex
        :return: euclidean distance to traverse. inf if obstacle in path
        """
        if not self.sensed_map.is_unoccupied(u) or not self.sensed_map.is_unoccupied(v):
            return float('inf')
        else:
            traversal_multiplier = self.sensed_map.get_traversal_cost(v)
            return heuristic(u, v) * traversal_multiplier

    def contain(self, u: (int, int)) -> (int, int):
        return u in self.U.vertices_in_heap

    def update_vertex(self, u: (int, int)):
        if self.g[u] != self.rhs[u] and self.contain(u):
            self.U.update(u, self.calculate_key(u))
        elif self.g[u] != self.rhs[u] and not self.contain(u):
            self.U.insert(u, self.calculate_key(u))
        elif self.g[u] == self.rhs[u] and self.contain(u):
            self.U.remove(u)

    def compute_shortest_path(self):
        while self.U.top_key() < self.calculate_key(self.s_start) or self.rhs[self.s_start] != self.g[self.s_start]:
            u = self.U.top()
            k_old = self.U.top_key()
            k_new = self.calculate_key(u)

            if k_old < k_new:
                self.U.update(u, k_new)
            elif self.g[u] > self.rhs[u]:
                self.g[u] = self.rhs[u]
                self.U.remove(u)
                pred = self.sensed_map.succ(vertex=u)
                for s in pred:
                    if s != self.s_goal:
                        self.rhs[s] = min(self.rhs[s], self.c(s, u) + self.g[u])
                    self.update_vertex(s)
            else:
                self.g_old = self.g[u]
                self.g[u] = float('inf')
                pred = self.sensed_map.succ(vertex=u)
                pred.append(u)
                for s in pred:
                    if self.rhs[s] == self.c(s, u) + self.g_old:
                        if s != self.s_goal:
                            min_s = float('inf')
                            succ = self.sensed_map.succ(vertex=s)
                            for s_ in succ:
                                temp = self.c(s, s_) + self.g[s_]
                                if min_s > temp:
                                    min_s = temp
                            self.rhs[s] = min_s
                    self.update_vertex(s)

    def get_next_step(self) -> (int, int):
        min_s = float('inf')
        succ = self.sensed_map.succ(vertex=self.s_start)
        for s in succ:
            temp = self.c(self.s_start, s) + self.g[s]
            if min_s > temp:
                min_s = temp
                s_min = s
        return s_min

    def get_last_step_in_window(self, win_size: int) -> (int, int):
        # returns the last point in the window of size win_size around the robot

        # path as numpy array
        path = np.zeros(self.sensed_map.map_extents, dtype=np.uint8)
        path[self.s_start] = 1

        # print("self.g: {}".format(self.g))
        # print("self.rhs: {}".format(self.rhs))

        # fill path
        for i in range(self.s_start[0], self.s_goal[0]):
            for j in range(self.s_start[1], self.s_goal[1]):
                if self.rhs[(i, j)] != float('inf'):
                    path[(i, j)] = 1
        
        # path[self.s_goal] = 6
        # path[self.s_start] = 5
        # print_path = path.copy()
        # print_path[self.s_goal] = 6
        # print_path[self.s_start] = 5
        # print("path: {}".format(print_path))

        # cut to window
        x_min = self.s_start[0] - win_size//2 
        x_max = self.s_start[0] + win_size//2 + 1
        y_min = self.s_start[1] - win_size//2
        y_max = self.s_start[1] + win_size//2 + 1

        pad_value = 0

        if x_min < 0:
            path = np.pad(path, ((abs(x_min), 0), (0, 0)), 'constant', constant_values=pad_value)
            x_min = 0
            x_max = win_size
        elif x_max > self.sensed_map.x_dim:
            path = np.pad(path, ((0, x_max - self.sensed_map.x_dim), (0, 0)), 'constant', constant_values=pad_value)
            x_max = path.shape[0]
            x_min = x_max - win_size
        if y_min < 0:
            path = np.pad(path, ((0, 0), (abs(y_min), 0)), 'constant', constant_values=pad_value)
            y_min = 0
            y_max = win_size
        elif y_max > self.sensed_map.y_dim:
            path = np.pad(path, ((0, 0), (0, y_max - self.sensed_map.y_dim)), 'constant', constant_values=pad_value)
            y_max = path.shape[1]
            y_min = y_max - win_size
        
        # check for highest value in window
        max_val = 0
        max_pos = (0, 0)
        for i in range(x_min, x_max):
            for j in range(y_min, y_max):
                if path[i, j] > max_val:
                    max_val = path[i, j]
                    max_pos = (i, j)
        
        return max_pos


    def rescan(self) -> Vertices:

        new_edges_and_old_costs = self.new_edges_and_old_costs
        self.new_edges_and_old_costs = None
        return new_edges_and_old_costs

    def move_and_replan(self, robot_position: (int, int)):
        path = [robot_position]
        self.s_start = robot_position
        self.s_last = self.s_start
        self.compute_shortest_path()

        while self.s_start != self.s_goal:
            assert (self.rhs[self.s_start] != float('inf')), "There is no known path!"

            succ = self.sensed_map.succ(self.s_start, avoid_obstacles=False)
            min_s = float('inf')
            arg_min = None
            for s_ in succ:
                temp = self.c(self.s_start, s_) + self.g[s_]
                if temp < min_s:
                    min_s = temp
                    arg_min = s_

            ### algorithm sometimes gets stuck here for some reason !!! FIX
            if arg_min is None:
                raise RuntimeError(f"No feasible successor from {self.s_start}; planner stuck.")
            self.s_start = arg_min
            path.append(self.s_start)
            # scan graph for changed costs
            changed_edges_with_old_cost = self.rescan()
            #print("len path: {}".format(len(path)))
            # if any edge costs changed
            if changed_edges_with_old_cost:
                self.k_m += heuristic(self.s_last, self.s_start)
                self.s_last = self.s_start

                # for all directed edges (u,v) with changed edge costs
                vertices = changed_edges_with_old_cost.vertices
                for vertex in vertices:
                    v = vertex.pos
                    succ_v = vertex.edges_and_c_old
                    for u, c_old in succ_v.items():
                        c_new = self.c(u, v)
                        if c_old > c_new:
                            if u != self.s_goal:
                                self.rhs[u] = min(self.rhs[u], self.c(u, v) + self.g[v])
                        elif self.rhs[u] == c_old + self.g[v]:
                            if u != self.s_goal:
                                min_s = float('inf')
                                succ_u = self.sensed_map.succ(vertex=u)
                                for s_ in succ_u:
                                    temp = self.c(u, s_) + self.g[s_]
                                    if min_s > temp:
                                        min_s = temp
                                self.rhs[u] = min_s
                            self.update_vertex(u)
            self.compute_shortest_path()
        # print("path found!")
        return path, self.g, self.rhs


def iterative_congestion_d_star(
    x_dim: int,
    y_dim: int,
    obstacle_grid: np.ndarray,
    agent_starts: Dict[str, Tuple[int, int]],
    agent_goals: Dict[str, Tuple[int, int]],
    iterations: int = 1,
    congestion_weight: float = 1.0,
) -> (Dict[str, List[Tuple[int, int]]], np.ndarray):
    """
    Run multiple rounds of D* Lite planning, cumulatively inflating traversal costs between rounds based on congestion.

    Args:
        x_dim: Grid width.
        y_dim: Grid height.
        obstacle_grid: Array matching grid size where OBSTACLE cells are blocked.
        agent_starts: Mapping of agent id to start position.
        agent_goals: Mapping of agent id to goal position.
        iterations: Number of planning iterations to perform (>= 1).
        congestion_weight: Multiplier applied to congestion counts when updating traversal costs. Penalties accumulate over iterations, but an agent’s own prior penalties are removed before replanning it.

    Returns:
        Tuple of:
            - dict mapping agent id to path (as produced during the final iteration)
            - np.ndarray of traversal cost multipliers to reuse for subsequent planning.
    """
    if iterations < 1:
        raise ValueError("iterations must be at least 1")
    if obstacle_grid.shape != (x_dim, y_dim):
        raise ValueError("obstacle_grid shape {} does not match provided dimensions ({}, {})".format(obstacle_grid.shape, x_dim, y_dim))

    base_map = OccupancyGridMap(x_dim, y_dim)
    base_map.set_map(obstacle_grid)

    traversal_costs = np.ones((x_dim, y_dim), dtype=np.float32)
    last_iteration_paths: Dict[str, List[Tuple[int, int]]] = {agent: [] for agent in agent_starts}
    agent_penalties: Dict[str, np.ndarray] = {
        agent: np.zeros((x_dim, y_dim), dtype=np.float32) for agent in agent_starts
    }

    for iteration_idx in range(iterations):
        base_map.set_traversal_costs(traversal_costs)
        current_paths: Dict[str, List[Tuple[int, int]]] = {}

        for agent_id, start in agent_starts.items():
            if congestion_weight != 0.0:
                prior_penalty = agent_penalties[agent_id]
                if prior_penalty.any():
                    traversal_costs -= prior_penalty
                    prior_penalty.fill(0.0)
                    base_map.set_traversal_costs(traversal_costs)

            goal = agent_goals[agent_id]
            planner = DStarLite(base_map, start, goal)
            path, _, _ = planner.move_and_replan(start)
            current_paths[agent_id] = path

            if congestion_weight != 0.0:
                penalty = agent_penalties[agent_id]
                penalty.fill(0.0)
                for cell in path:
                    penalty[cell] += congestion_weight
                traversal_costs += penalty
                base_map.set_traversal_costs(traversal_costs)

        last_iteration_paths = current_paths

    return last_iteration_paths, traversal_costs

class Priority:
    """
    handle lexicographic order of keys
    """

    def __init__(self, k1, k2):
        """
        :param k1: key value
        :param k2: key value
        """
        self.k1 = k1
        self.k2 = k2

    def __lt__(self, other):
        """
        lexicographic 'lower than'
        :param other: comparable keys
        :return: lexicographic order
        """
        return self.k1 < other.k1 or (self.k1 == other.k1 and self.k2 < other.k2)

    def __le__(self, other):
        """
        lexicographic 'lower than or equal'
        :param other: comparable keys
        :return: lexicographic order
        """
        return self.k1 < other.k1 or (self.k1 == other.k1 and self.k2 <= other.k2)


class PriorityNode:
    """
    handle lexicographic order of vertices
    """

    def __init__(self, priority, vertex):
        """
        :param priority: the priority of a
        :param vertex:
        """
        self.priority = priority
        self.vertex = vertex

    def __le__(self, other):
        """
        :param other: comparable node
        :return: lexicographic order
        """
        return self.priority <= other.priority

    def __lt__(self, other):
        """
        :param other: comparable node
        :return: lexicographic order
        """
        return self.priority < other.priority


class PriorityQueue:
    def __init__(self):
        self.heap = []
        self.vertices_in_heap = []

    def top(self):
        return self.heap[0].vertex

    def top_key(self):
        if len(self.heap) == 0: return Priority(float('inf'), float('inf'))
        return self.heap[0].priority

    def pop(self):
        """!!!THIS CODE WAS COPIED AND MODIFIED!!! Source: Lib/heapq.py"""
        """Pop the smallest item off the heap, maintaining the heap invariant."""
        lastelt = self.heap.pop()  # raises appropriate IndexError if heap is empty
        self.vertices_in_heap.remove(lastelt)
        if self.heap:
            returnitem = self.heap[0]
            self.heap[0] = lastelt
            self._siftup(0)
        else:
            returnitem = lastelt
        return returnitem

    def insert(self, vertex, priority):
        item = PriorityNode(priority, vertex)
        self.vertices_in_heap.append(vertex)
        """!!!THIS CODE WAS COPIED AND MODIFIED!!! Source: Lib/heapq.py"""
        """Push item onto heap, maintaining the heap invariant."""
        self.heap.append(item)
        self._siftdown(0, len(self.heap) - 1)

    def remove(self, vertex):
        self.vertices_in_heap.remove(vertex)
        for index, priority_node in enumerate(self.heap):
            if priority_node.vertex == vertex:
                self.heap[index] = self.heap[len(self.heap) - 1]
                self.heap.remove(self.heap[len(self.heap) - 1])
                break
        self.build_heap()

    def update(self, vertex, priority):
        for index, priority_node in enumerate(self.heap):
            if priority_node.vertex == vertex:
                self.heap[index].priority = priority
                break
        self.build_heap()

    # !!!THIS FUNCTION WAS COPIED AND MODIFIED!!! Source: Lib/heapq.py
    def build_heap(self):
        """Transform list into a heap, in-place, in O(len(x)) time."""
        n = len(self.heap)
        # Transform bottom-up.  The largest index there's any point to looking at
        # is the largest with a child index in-range, so must have 2*i + 1 < n,
        # or i < (n-1)/2.  If n is even = 2*j, this is (2*j-1)/2 = j-1/2 so
        # j-1 is the largest, which is n//2 - 1.  If n is odd = 2*j+1, this is
        # (2*j+1-1)/2 = j so j-1 is the largest, and that's again n//2-1.
        for i in reversed(range(n // 2)):
            self._siftup(i)

    # !!!THIS FUNCTION WAS COPIED AND MODIFIED!!! Source: Lib/heapq.py
    # 'heap' is a heap at all indices >= startpos, except possibly for pos.  pos
    # is the index of a leaf with a possibly out-of-order value.  Restore the
    # heap invariant.
    def _siftdown(self, startpos, pos):
        newitem = self.heap[pos]
        # Follow the path to the root, moving parents down until finding a place
        # newitem fits.
        while pos > startpos:
            parentpos = (pos - 1) >> 1
            parent = self.heap[parentpos]
            if newitem < parent:
                self.heap[pos] = parent
                pos = parentpos
                continue
            break
        self.heap[pos] = newitem

    def _siftup(self, pos):
        endpos = len(self.heap)
        startpos = pos
        newitem = self.heap[pos]
        # Bubble up the smaller child until hitting a leaf.
        childpos = 2 * pos + 1  # leftmost child position
        while childpos < endpos:
            # Set childpos to index of smaller child.
            rightpos = childpos + 1
            if rightpos < endpos and not self.heap[childpos] < self.heap[rightpos]:
                childpos = rightpos
            # Move the smaller child up.
            self.heap[pos] = self.heap[childpos]
            pos = childpos
            childpos = 2 * pos + 1
        # The leaf at pos is empty now.  Put newitem there, and bubble it up
        # to its final resting place (by sifting its parents down).
        self.heap[pos] = newitem
        self._siftdown(startpos, pos)

import math
from typing import List


def heuristic(p: (int, int), q: (int, int)) -> float:
    """
    Helper function to compute distance between two points.
    :param p: (x,y)
    :param q: (x,y)
    :return: manhattan distance
    """
    return math.sqrt((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2)


def get_movements_4n(x: int, y: int) -> List:
    """
    get all possible 4-connectivity movements.
    :return: list of movements with cost [(dx, dy, movement_cost)]
    """
    return [(x + 1, y + 0),
            (x + 0, y + 1),
            (x - 1, y + 0),
            (x + 0, y - 1)]


def get_movements_8n(x: int, y: int) -> List:
    """
    get all possible 8-connectivity movements.
    :return: list of movements with cost [(dx, dy, movement_cost)]
    """
    return [(x + 1, y + 0),
            (x + 0, y + 1),
            (x - 1, y + 0),
            (x + 0, y - 1),
            (x + 1, y + 1),
            (x - 1, y + 1),
            (x - 1, y - 1),
            (x + 1, y - 1)]
