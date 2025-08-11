import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import m_star


def test_avoids_edge_swap():
    start = (1, 0)
    goal = (1, 1)
    static_obstacles = set()
    # Other agent moves from (1,1) to (1,0) causing potential edge swap.
    dynamic_obstacles = {0: {(1, 1)}, 1: {(1, 0)}}

    path = m_star.astar_single(
        start=start,
        goal=goal,
        static_obstacles=static_obstacles,
        dynamic_obstacles=dynamic_obstacles,
        R=2,
        C=2,
        time_horizon=10,
    )

    assert path == [(1, 0), (0, 0), (0, 1), (1, 1)]
