from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import gymnasium as gym
import ray
from ray.rllib.env.env_context import EnvContext
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import numpy as np
import time
from gymnasium import spaces
from heapq import heappush, heappop
import os
import json
from d_star_lite import DStarLite, OccupancyGridMap, iterative_congestion_d_star

import matplotlib.pyplot as plt

from matplotlib.patches import Rectangle, Circle
from matplotlib import colors as mcolors
from matplotlib.animation import FFMpegWriter

from tqdm import tqdm

class RLMAPF(MultiAgentEnv):
    """
    RLMAPF environment for multi-agent pathfinding using D* Lite algorithm.
    """
    _MAP_CACHE: Dict[str, Dict[str, Any]] = {}

    def __init__(self, env_config):
        # Load environment configuration
        self.config = {**self.get_default_config(), **env_config}
        self.max_steps = self.config["max_steps"]
        self.goal_reward = self.config["goal_reward"]
        self.collision_penalty = self.config["collision_penalty"]
        self.step_cost = self.config["step_cost"]
        self.wait_cost_multiplier = self.config["wait_cost_multiplier"]
        self.movement_probability = self.config["movement_probability"]
        self.render_mode = self.config["render_mode"]
        self.render_config = self.config["render_config"]
        self.observation_type = self.config["observation_type"]
        if self.config["observation_size"] % 2 == 0:
            raise ValueError("Observation size must be odd")
        self.observation_size = self.config["observation_size"]
        self.use_d_star_lite = self.config["use_d_star_lite"]
        self.start_goal_on_periphery = self.config["start_goal_on_periphery"]
        self.cycle_maps_without_replacement = self.config.get("cycle_maps_without_replacement", False)
        self.shuffle_map_cycle = self.config.get("shuffle_map_cycle", True)
        self.initial_agents_num: int = 0
        self.successful_agents: int = 0
        self._map_cycle_queue: List[str] = []
        self._current_map_name: Optional[str] = None
        self._map_cycle_seed: Optional[int] = None
        maps_config = self.config.get("maps_names_with_variants") or {}
        self._map_usage_counter: Dict[str, int] = {str(name): 0 for name in maps_config}
        self.print_map_usage = self.config.get("print_map_usage", True)

        # Set seed
        if self.config["seed"] is not None:
            self._seed = self.config["seed"]
        else:
            self._seed = np.random.randint(0, 1e9)

        # Check if all maps have the same size
        # self.verify_map_sizes() # No longer needed

        # Debug visualization accumulators
        self._dstar_gif_writer = None
        self._dstar_debug_frame_idx = 0

        # Initialize environment variables
        self.reset(self._seed)
        
        # Define agent observation and action spaces
        self.define_observation_spaces()
        self.action_space = gym.spaces.Discrete(5)
        
        # Initialize last_actions to track agent movement directions
        self.last_actions = {}
        self._path_styles = {}

        # Initialize D* Lite-related variables only if enabled
        if self.use_d_star_lite:
            self.d_star_maps = {}
            self.d_star_paths = {}
            self.d_star_planned_paths = {}
            self.start_d_star_paths = {}
            self.d_stars = {}

        super().__init__()

    def get_default_config(self):
        default_config = {
            "agents_num": 2, 
            "max_steps": 100, 
            "goal_reward": 10, 
            "observation_type": 'array',  # 'array' 
            "observation_size": 9, # Rectangle around agent, must be odd
            # If True, observations will include a single HxWxC tensor suitable
            # for CNNs (stacking relevant grids as channels) plus a small vector
            # for distance to goal. If False, returns the legacy dict of 2D grids.
            "use_cnn_observation": False,
            "collision_penalty": 1, 
            "movement_probability": 0, 
            "render_mode": "none", # 'human' or 'none' 
            "render_config": {
                "title": "RLMAPF",
                "show_render": False,
                "save_video": False,
                "include_legend": True,
                "legend_position": (0, 0),
                "video_path": "render.mp4",
                "video_fps": 10,
                "video_dpi": 300,
                "render_delay": 0.2,
                "save_frames": False,
                "frames_path": "frames/",
                "smooth_motion": False,
                "motion_frames": 5,
                "frame_format": "png",
                "frame_dpi": 150,
            },
            "d_star_debug": {
                "save_gif": False,
                "gif_path": "dstar_debug.gif",
                "gif_fps": 5,
                "save_pngs": False,
                "png_dir": "dstar_debug_frames",
            },
            "seed": None, # int or None
            "map_path": os.getcwd() + "/maps/",
            "maps_names_with_variants": { # map_name: variants, if variants is None, then all variants are loaded
            },
            "step_cost": 0.0,
            "wait_cost_multiplier": 2,
            # Removed predict_distance support
            "penalize_collision": True,
            "penalize_waiting": True,
            "penalize_steps": True,
            "reward_closer_to_goal_each_step": False,
            "reward_closer_to_goal_final": False,
            "reward_final_d_star": True,
            "use_d_star_lite": True,  # Enable or disable D* Lite
            "d_star_iterations": 5,  # Number of congestion-based replanning rounds
            "d_star_congestion_weight": 5.0,  # Cost multiplier per unit of congestion
            "d_star_path_progress_weight": 0.0,  # Reward scaling for reducing D* path length step-to-step
            "penalize_left_side_bottom_passing": False,  # Penalize left-side/bottom passing
            "start_goal_on_periphery": False,  # Force spawn on border with mirrored goals
            "cycle_maps_without_replacement": False,
            "shuffle_map_cycle": True,
            "print_map_usage": True,
            # Collision priority multipliers (relative: left > bottom > right > top)
            "use_collision_priority_multiplier": True,
            "collision_priority_weights": {
                "left": 1.0,
                "bottom": 0.75,
                "right": 0.5,
                "top": 0.25,
            },
            "collision_priority_multiplier_range": (0.5, 1.5),
        }
        return default_config

    @classmethod
    def _resolve_map_path(cls, base_path: str, map_name: str) -> str:
        path = os.path.join(base_path, map_name)
        if not path.endswith(".json"):
            path = f"{path}.json"
        return path

    @classmethod
    def _load_map_data(cls, map_path: str) -> Dict[str, Any]:
        if map_path not in cls._MAP_CACHE:
            with open(map_path, "r") as handle:
                cls._MAP_CACHE[map_path] = json.load(handle)
        return cls._MAP_CACHE[map_path]

    def define_observation_spaces(self):
        if self.observation_type == 'array':
            if self.config.get("use_cnn_observation", False):
                # Stack grid-based features into channels for CNN consumption.
                # Channels: obstacles, agents_positions, and optionally
                # d_star_path, d_star_path_others when D* Lite is enabled.
                # channels = 2 + (2 if self.use_d_star_lite else 0) 
                channels = 3 if self.use_d_star_lite else 2 
                self.observation_space = gym.spaces.Dict({
                    'grid': gym.spaces.Box(
                        low=0.0, high=1.0,
                        shape=(self.observation_size, self.observation_size, channels),
                        dtype=np.float32,
                    ),
                    'distance_to_goal': gym.spaces.Box(
                        low=-1.0, high=1.0, shape=(2,), dtype=np.float32
                    ),
                })
            else:
                # Legacy dict of separate 2D grids (best for MLP).
                if self.use_d_star_lite:
                    self.observation_space = gym.spaces.Dict({
                        'd_star_path': gym.spaces.Box(low=0.0, high=1.0, shape=(self.observation_size, self.observation_size), dtype=np.float32),
                        'd_star_path_others': gym.spaces.Box(low=0.0, high=1.0, shape=(self.observation_size, self.observation_size), dtype=np.float32),
                        'distance_to_goal': gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
                        'agents_positions': gym.spaces.Box(low=0.0, high=1.0, shape=(self.observation_size, self.observation_size), dtype=np.float32),
                        'obstacles': gym.spaces.Box(low=0.0, high=1.0, shape=(self.observation_size, self.observation_size), dtype=np.float32),
                    })
                else:
                    self.observation_space = gym.spaces.Dict({
                        'distance_to_goal': gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
                        'agents_positions': gym.spaces.Box(low=0.0, high=1.0, shape=(self.observation_size, self.observation_size), dtype=np.float32),
                        'obstacles': gym.spaces.Box(low=0.0, high=1.0, shape=(self.observation_size, self.observation_size), dtype=np.float32),
                    })
        else:
            raise ValueError("Invalid observation type: {}".format(self.observation_type))

    def _random_pos(self, tries=None):
        free_positions = set()
        for x in range(self.grid_size[0]):
            for y in range(self.grid_size[1]):
                pos = (x, y)
                if pos not in self.obstacles and pos not in self.agent_positions.values() and pos not in self.goal_positions.values():
                    free_positions.add(pos)
        if len(free_positions) == 0:
            raise ValueError("No free positions available.")

        pos_id = np.random.choice(range(len(free_positions)))
        return list(free_positions)[pos_id]

    def _is_periphery(self, pos):
        x, y = pos
        max_x = self.grid_size[0] - 1
        max_y = self.grid_size[1] - 1
        return x == 0 or y == 0 or x == max_x or y == max_y

    def _assign_periphery_positions(self):
        max_x = self.grid_size[0] - 1
        max_y = self.grid_size[1] - 1
        candidates = []
        for x in range(self.grid_size[0]):
            for y in range(self.grid_size[1]):
                pos = (x, y)
                if not self._is_periphery(pos):
                    continue
                if pos in self.obstacles:
                    continue
                goal = (max_x - x, max_y - y)
                if goal in self.obstacles:
                    continue
                candidates.append(pos)

        required_agents = len(self._agent_ids)
        if len(candidates) < required_agents:
            raise ValueError(
                "Not enough obstacle-free periphery cells ({}) to place {} agents with mirrored goals.".format(
                    len(candidates), required_agents
                )
            )

        candidates = np.array(candidates, dtype=int)
        np.random.shuffle(candidates)

        self.agent_positions = {}
        self.goal_positions = {}
        for agent, start in zip(sorted(self._agent_ids), candidates):
            start = tuple(int(coord) for coord in start)
            goal = (max_x - start[0], max_y - start[1])
            self.agent_positions[agent] = start
            self.goal_positions[agent] = goal

    def _init_path_styles(self):
        """Assign random colors and linestyles per agent for D* Lite path rendering."""
        self._path_styles = {}
        if not self.use_d_star_lite:
            return
        line_styles = ["-", "--", "-.", ":"]
        style_labels = {
            "-": "solid",
            "--": "dashed",
            "-.": "dash-dot",
            ":": "dotted",
        }
        rng_seed = self._seed if self._seed is not None else None
        rng = np.random.default_rng(rng_seed)
        for agent in sorted(self._agent_ids):
            hue = float(rng.random())
            rgb = mcolors.hsv_to_rgb((hue, 0.7, 0.95))
            linestyle = rng.choice(line_styles)
            self._path_styles[agent] = {
                "color": rgb,
                "linestyle": linestyle,
                "style_label": style_labels.get(linestyle, "pattern"),
                "color_hex": mcolors.to_hex(rgb),
            }
    
    def _collision_priority_multiplier(
        self,
        self_pos: Optional[Tuple[int, int]],
        other_pos: Optional[Tuple[int, int]] = None,
    ) -> float:
        """
        Scale collision penalties based on relative position priority (left > bottom > right > top).
        """
        if not self.config.get("use_collision_priority_multiplier", False):
            return 1.0
        if self_pos is None:
            return 1.0

        if other_pos is None:
            return 1.0

        default_weights = {"left": 1.0, "bottom": 0.75, "right": 0.5, "top": 0.25}
        user_weights = self.config.get("collision_priority_weights", {}) or {}
        weights = {**default_weights}
        for key in default_weights:
            if key in user_weights:
                try:
                    weights[key] = float(user_weights[key])
                except (TypeError, ValueError):
                    pass

        range_cfg = self.config.get("collision_priority_multiplier_range", (0.5, 1.5))
        try:
            min_mult, max_mult = float(range_cfg[0]), float(range_cfg[1])
        except Exception:
            min_mult, max_mult = 0.5, 1.5
        if max_mult < min_mult:
            min_mult, max_mult = max_mult, min_mult

        dx = self_pos[0] - other_pos[0]
        dy = self_pos[1] - other_pos[1]
        if dx == 0 and dy == 0:
            return 1.0

        if abs(dx) >= abs(dy):
            direction = "left" if dx < 0 else "right"
        else:
            direction = "bottom" if dy > 0 else "top"

        weight = weights.get(direction, 1.0)
        min_weight = min(weights.values())
        max_weight = max(weights.values())
        if max_weight - min_weight <= 0:
            return 1.0

        normalized = np.clip((weight - min_weight) / (max_weight - min_weight), 0.0, 1.0)
        return min_mult + normalized * (max_mult - min_mult)

    def reset(self, seed=None, options=None):
        """
        Resets the environment to its initial state.

        Returns:
            dict: Observations for each agent.
        """
        if seed is not None:
            self.set_seed(seed)

        # Reset debug GIF state
        if getattr(self, "_dstar_gif_writer", None) is not None:
            try:
                self._dstar_gif_writer.close()
            except Exception:
                pass
            self._dstar_gif_writer = None
        self._dstar_debug_frame_idx = 0

        # # Reset environment variables
        # # TODO: REMOVE
        # self.config["agents_num"] = 8
        # # TODO: REMOVE

        self.steps = 0
        self._agent_ids = set({"agent_" + str(i) for i in range(self.config["agents_num"])})
        self.rewards = {agent: 0 for agent in self._agent_ids}
        self.dones = {agent: False for agent in self._agent_ids}
        self.dones['__all__'] = False
        self.initial_agents_num = len(self._agent_ids)
        self.successful_agents = 0
        self.number_of_collisions = {agent: 0 for agent in self._agent_ids}
        self.number_of_steps = {agent: 0 for agent in self._agent_ids}
        self._init_path_styles()
        self._prev_agent_positions = {}

        # Initialize empty agents, agent positions and goal positions
        self.agent_positions = dict()
        self.goal_positions = dict()
        self.obstacles = set()

        # Load map
        if self.config["maps_names_with_variants"] is not None and len(self.config["maps_names_with_variants"]) > 0:
            map_name = self._next_map_name()
            self.load_map(map_name)
        else:
            raise ValueError("No maps to load")
        
        if self.start_goal_on_periphery:
            self._assign_periphery_positions()
        else:
            # Generate random agent positions if empty
            if len(self.agent_positions) == 0:
                for agent in self._agent_ids:
                    self.agent_positions[agent] = self._random_pos()

            # Generate random goal positions if empty
            if len(self.goal_positions) == 0:
                for agent in self._agent_ids:
                    self.goal_positions[agent] = self._random_pos()

        # Precompute static obstacle grid for faster cropping
        obstacles_array = np.zeros(self.grid_size, dtype=np.float32)
        for pos in self.obstacles:
            obstacles_array[pos] = 1.0
        self._obstacles_grid = obstacles_array

        # Init D* Lite only if enabled
        if self.use_d_star_lite:
            d_star_obstacles = np.zeros(self.grid_size, dtype=np.uint8)
            for pos in self.obstacles:
                d_star_obstacles[pos] = 255

            iterations = max(1, int(self.config.get("d_star_iterations", 1)))
            congestion_weight = float(self.config.get("d_star_congestion_weight", 1.0))
            planned_paths, traversal_costs = iterative_congestion_d_star(
                self.grid_size[0],
                self.grid_size[1],
                d_star_obstacles,
                {agent: self.agent_positions[agent] for agent in self._agent_ids},
                {agent: self.goal_positions[agent] for agent in self._agent_ids},
                iterations=iterations,
                congestion_weight=congestion_weight,
            )

            self.d_star_maps = {}
            self.d_stars = {}
            self.d_star_paths = {}
            self.d_star_planned_paths = deepcopy(planned_paths)

            shared_occupancy_map = OccupancyGridMap(self.grid_size[0], self.grid_size[1])
            shared_occupancy_map.set_map(d_star_obstacles)
            shared_occupancy_map.set_traversal_costs(traversal_costs)
            for agent in self._agent_ids:
                planner = DStarLite(
                    shared_occupancy_map,
                    self.agent_positions[agent],
                    self.goal_positions[agent],
                )

                self.d_star_maps[agent] = shared_occupancy_map
                self.d_stars[agent] = planner

                path = planned_paths.get(agent)
                if path is None:
                    path, _, _ = planner.move_and_replan(self.agent_positions[agent])
                self.d_star_paths[agent] = path

            self.start_d_star_paths = deepcopy(self.d_star_planned_paths)

        # Render
        self._maybe_capture_d_star_step_frame(is_reset=True)

        if self.render_mode == "human":
            self.render(clear=False, 
                        title=self.render_config["title"],
                        save_frames=self.render_config["save_frames"], 
                        frames_path=self.render_config["frames_path"], 
                        save_video=self.render_config["save_video"], 
                        video_path=self.render_config["video_path"],
                        show_render=self.render_config["show_render"],
                        render_delay=self.render_config["render_delay"],
                        include_legend=self.render_config["include_legend"],
                        legend_position=self.render_config["legend_position"],
                        smooth_motion=self.render_config.get("smooth_motion", False),
                        motion_frames=self.render_config.get("motion_frames", 5))

        return self._get_observations(), self._get_info()

    def lookup_action(self, action):
        if action == 0:
            return "Up"
        elif action == 1:
            return "Down"
        elif action == 2:
            return "Left"
        elif action == 3:
            return "Right"
        elif action == 4:
            return "Wait"
        else:
            return "Invalid"
        
    def set_seed(self, seed):
        """
        Seed the environment.

        Arguments:
            seed (int): Seed value.
        """
        self._seed = seed
        np.random.seed(seed)
        if self.cycle_maps_without_replacement and (self._map_cycle_seed is None or self._map_cycle_seed != seed):
            # Rebuild map cycle only when the effective seed actually changes.
            self._map_cycle_seed = seed
            self._map_cycle_queue = []
            self._current_map_name = None
        self._map_cycle_queue = []

    def get_seed(self):
        """
        Returns the current seed value.

        Returns:
            int: Seed value.
        """
        return self._seed

    def step(self, input):
        # Move robots based on actions and handle collisions
        actions = input
        # Store last actions as a copy
        self.last_actions = {agent: action for agent, action in actions.items()}
        self.rewards = {agent: 0 for agent in self._agent_ids}
        terminateds = {agent: False for agent in self._agent_ids}
        current_state = {agent: self.agent_positions[agent] for agent in self._agent_ids}
        new_state = {agent: self.agent_positions[agent] for agent in self._agent_ids}

        def apply_penalty(agent, penalty):
            self.rewards[agent] -= penalty

        def apply_collision_penalty(agent, penalty=None, position=None, other_positions=None):
            base_penalty = self.collision_penalty if penalty is None else penalty
            pos = position
            if pos is None:
                pos = current_state.get(agent, self.agent_positions.get(agent))
            others = other_positions or []
            multipliers = []
            for other_pos in others:
                mult = self._collision_priority_multiplier(pos, other_pos)
                multipliers.append(mult)
            if multipliers:
                multiplier = float(np.mean(multipliers))
            else:
                multiplier = 1.0
            apply_penalty(agent, base_penalty * multiplier)

        def reward(agent, reward=self.goal_reward):
            self.rewards[agent] += reward

        # Move agents
        for agent, action in actions.items():
            new_pos = current_state[agent]
            if self.dones[agent]:
                continue
            # Define movement deltas for each action
            action_deltas = {
                0: (0, -1),  # Up
                1: (0, 1),   # Down
                2: (-1, 0),  # Left
                3: (1, 0),   # Right
                4: (0, 0),   # Wait
            }

            # Get the delta for the current action
            delta = action_deltas.get(action, None)
            if delta is None:
                raise ValueError("Invalid action: {}".format(action))

            # Calculate the new position
            new_pos = (new_pos[0] + delta[0], new_pos[1] + delta[1])

            # Check if the new position is out of bounds
            if not (0 <= new_pos[0] < self.grid_size[0] and 0 <= new_pos[1] < self.grid_size[1]):
                if self.config["penalize_collision"]:
                    apply_collision_penalty(agent, position=current_state[agent])
                new_pos = current_state[agent]  # Revert to the current position
            
            # Penalize for moving
            if action != 4:
                if self.config["penalize_steps"]:
                    apply_penalty(agent, self.step_cost)
            else:
                if self.config["penalize_waiting"]:
                    apply_penalty(agent, self.step_cost * self.wait_cost_multiplier)

            new_state[agent] = new_pos

        # Check for collisions
        for agent, new_pos in new_state.items():
            # Check for collisions with obstacles
            if new_pos in self.obstacles:
                if self.config["penalize_collision"]:
                    apply_collision_penalty(agent, position=new_pos)
                self.number_of_collisions[agent] += 1
                new_state[agent] = current_state[agent]

            for agent2, new_pos2 in new_state.items():
                if agent2 != agent and new_pos2 == current_state[agent] and new_pos == current_state[agent2]:
                    if self.config["penalize_collision"]:
                        apply_collision_penalty(
                            agent,
                            position=current_state[agent],
                            other_positions=[current_state[agent2]],
                        )
                        apply_collision_penalty(
                            agent2,
                            position=current_state[agent2],
                            other_positions=[current_state[agent]],
                        )
                    self.number_of_collisions[agent] += 1
                    self.number_of_collisions[agent2] += 1
                    new_state[agent] = current_state[agent]
                    new_state[agent2] = current_state[agent2]
                    # break

        # Check if any agents end up in the same position
        if len(set(new_state.values())) < len(new_state):
            for agent, new_pos in new_state.items():
                # Check if agent is in a position that is occupied by another agent
                if list(new_state.values()).count(new_pos) > 1:
                    agents_to_penalize = [a for a, p in new_state.items() if p == new_pos]
                    if self.config["penalize_collision"]:
                        for agent_to_penalize in agents_to_penalize:
                            apply_collision_penalty(
                                agent_to_penalize,
                                position=current_state.get(agent_to_penalize, new_pos),
                                other_positions=[
                                    current_state.get(a, new_pos)
                                    for a in agents_to_penalize
                                    if a != agent_to_penalize
                                ],
                            )
                    for agent_to_penalize in agents_to_penalize:
                        self.number_of_collisions[agent_to_penalize] += 1
                    for agent_to_penalize in agents_to_penalize:
                        new_state[agent_to_penalize] = current_state[agent_to_penalize]

        # Update agent positions
        self.agent_positions = new_state

        # Increment per-agent step counters
        for agent in self._agent_ids:
            if not self.dones[agent]:
                self.number_of_steps[agent] += 1

        # Penalize left-side/bottom passing (if enabled)
        if self.config.get("penalize_left_side_bottom_passing", False):
            # For each pair of agents, check if they are passing each other
            for agent1 in self._agent_ids:
                for agent2 in self._agent_ids:
                    if agent1 >= agent2:
                        continue
                    pos1_prev = current_state[agent1]
                    pos2_prev = current_state[agent2]
                    pos1_new = self.agent_positions[agent1]
                    pos2_new = self.agent_positions[agent2]
                    action1 = self.last_actions[agent1]
                    action2 = self.last_actions[agent2]

                    # Up/Down passing
                    # 0: up, 1: down, 2: left, 3: right, 4: wait
                    if (action1 == 0 and action2 == 1) or (action1 == 1 and action2 == 0):
                        # Check if they are on the same column and adjacent rows
                        if pos1_prev[0] == pos2_prev[0] and abs(pos1_prev[1] - pos2_prev[1]) == 1:
                            # Penalize when the agent going up is to the left (lower x)
                            if action1 == 0 and pos1_prev[0] < pos2_prev[0]:
                                apply_penalty(agent1, 0.1)
                                apply_penalty(agent2, 0.1)
                            elif action2 == 0 and pos2_prev[0] < pos1_prev[0]:
                                apply_penalty(agent1, 0.1)
                                apply_penalty(agent2, 0.1)

                    # Left/Right passing
                    if (action1 == 2 and action2 == 3) or (action1 == 3 and action2 == 2):
                        # Check if they are on the same row and adjacent columns
                        if pos1_prev[1] == pos2_prev[1] and abs(pos1_prev[0] - pos2_prev[0]) == 1:
                            # Penalize when the agent going left is below (higher y)
                            if action1 == 2 and pos1_prev[1] > pos2_prev[1]:
                                apply_penalty(agent1, 0.1)
                                apply_penalty(agent2, 0.1)
                            elif action2 == 2 and pos2_prev[1] > pos1_prev[1]:
                                apply_penalty(agent1, 0.1)
                                apply_penalty(agent2, 0.1)
        # Update D* Lite paths only if enabled
        d_star_progress_weight = float(self.config.get("d_star_path_progress_weight", 0.0))
        prev_d_star_lengths = {}
        if self.use_d_star_lite:
            if d_star_progress_weight != 0.0:
                prev_d_star_lengths = {
                    agent: len(self.d_star_paths.get(agent, [])) for agent in self._agent_ids
                }

            def _project_to_planned_path(full_path, current_pos):
                """
                Reuse the congestion-weighted path from reset and, if the agent
                drifted, snap back to the nearest point on that path instead of
                replanning a brand-new route.
                """
                if not full_path:
                    return []
                if current_pos in full_path:
                    idx = full_path.index(current_pos)
                    return full_path[idx:]

                # Find closest waypoint on the original path (Manhattan distance).
                best_idx = None
                best_dist = None
                for idx, cell in enumerate(full_path):
                    dist = abs(cell[0] - current_pos[0]) + abs(cell[1] - current_pos[1])
                    if best_dist is None or dist < best_dist:
                        best_dist = dist
                        best_idx = idx

                if best_idx is None:
                    return []

                snapped_suffix = full_path[best_idx:]
                if snapped_suffix and snapped_suffix[0] != current_pos:
                    return [current_pos] + snapped_suffix
                return snapped_suffix

            planned_paths_static = getattr(self, "d_star_planned_paths", {}) or {}
            for agent in self._agent_ids:
                full_path = planned_paths_static.get(agent, [])
                if not full_path:
                    # Fall back to a fresh plan only if we somehow missed the initial path.
                    path, _, _ = self.d_stars[agent].move_and_replan(self.agent_positions[agent])
                    self.d_star_paths[agent] = path
                    if agent not in planned_paths_static:
                        planned_paths_static[agent] = path
                    continue

                projected_path = _project_to_planned_path(full_path, self.agent_positions[agent])
                if projected_path:
                    self.d_star_paths[agent] = projected_path
                else:
                    # Absolute fallback: should rarely trigger, keeps agents from getting stuck.
                    path, _, _ = self.d_stars[agent].move_and_replan(self.agent_positions[agent])
                    self.d_star_paths[agent] = path

            if d_star_progress_weight != 0.0:
                for agent in self._agent_ids:
                    prev_len = prev_d_star_lengths.get(agent)
                    new_len = len(self.d_star_paths.get(agent, []))
                    if prev_len is None or not np.isfinite(prev_len) or prev_len <= 0:
                        continue
                    if not np.isfinite(new_len):
                        continue
                    delta = prev_len - new_len  # Positive when the planned path to goal shrinks
                    if delta != 0:
                        reward(agent, reward=d_star_progress_weight * delta)

        self._maybe_capture_d_star_step_frame(is_reset=False)

        for agent, new_pos in self.agent_positions.items():
            # Check for reaching goal positions and update rewards
            if new_pos == self.goal_positions[agent] and not self.dones[agent]:
                reward(agent) 
                terminateds[agent] = True
                self.dones[agent] = True
                self.successful_agents += 1

                # remove agent
                self._agent_ids.remove(agent)

        # Increment steps and check for episode termination
        self.steps += 1
        terminateds['__all__'] = all([self.dones[agent] for agent in self._agent_ids])
        truncateds = {agent: self.steps >= self.max_steps for agent in self._agent_ids}
        truncateds['__all__'] = self.steps >= self.max_steps

        # Reward based on D* Lite distance only if enabled
        if self.use_d_star_lite and self.config["reward_final_d_star"]:
            for agent in self._agent_ids:
                if truncateds[agent]:
                    start_path = self.start_d_star_paths.get(agent, [])
                    current_path = self.d_star_paths.get(agent, [])
                    d_star_distance_start = len(start_path)
                    d_star_distance_current = len(current_path)

                    if d_star_distance_start <= 1:
                        continue

                    d_star_reward = 5.0 - d_star_distance_current / d_star_distance_start * 5.0
                    if not np.isfinite(d_star_reward):
                        continue
                    reward(agent, reward=d_star_reward)

        # Reward for being closer to goal final
        if self.config["reward_closer_to_goal_final"]:
            for agent in self._agent_ids:
                if truncateds[agent]:
                    distance_to_goal = np.linalg.norm(np.array(self.goal_positions[agent]) - np.array(self.agent_positions[agent]))
                    reward(agent, reward=1.0 - distance_to_goal / (self.grid_size[0] + self.grid_size[1]))

        # Render
        if self.render_mode == "human":
            self.render(clear=False, 
                        title=self.render_config["title"],
                        save_frames=self.render_config["save_frames"], 
                        frames_path=self.render_config["frames_path"], 
                        save_video=self.render_config["save_video"], 
                        video_path=self.render_config["video_path"],
                        show_render=self.render_config["show_render"],
                        render_delay=self.render_config["render_delay"],
                        include_legend=self.render_config["include_legend"],
                        legend_position=self.render_config["legend_position"],
                        smooth_motion=self.render_config.get("smooth_motion", False),
                        motion_frames=self.render_config.get("motion_frames", 5))
            

        return self._get_observations(), self.rewards, terminateds, truncateds, self._get_info()
    
    def _get_info(self):
        """
        Returns info for 

        Returns:
            dict: Info
        """
        info_dict: Dict[str, Dict[str, Any]] = dict()
        for agent in self._agent_ids:
            obstacles_array = np.zeros(self.grid_size, dtype=int)
            for pos in self.obstacles:
                obstacles_array[pos] = 1


            info_dict[agent] = {
                "current_position": self.agent_positions[agent],
                "number_of_collisions": self.number_of_collisions[agent],
                "number_of_steps": self.number_of_steps[agent],
            }
        # Provide episode-level progress for callbacks/metrics.
        # RLlib allows a special "__common__" entry for shared info.
        info_dict["__common__"] = {
            "episode_successful_agents": self.successful_agents,
            "episode_success_rate": (
                self.successful_agents / self.initial_agents_num
                if self.initial_agents_num > 0
                else 0.0
            ),
        }
        return info_dict

    def crop_array(self, array, x, y, size, pad_value=0):
        x_min = x - size // 2
        x_max = x + size // 2 + 1
        y_min = y - size // 2
        y_max = y + size // 2 + 1

        # Padding
        if x_min < 0:
            array = np.pad(array, ((abs(x_min), 0), (0, 0)), 'constant', constant_values=pad_value)
            x_min = 0
            x_max = size
        elif x_max > self.grid_size[0]:
            array = np.pad(array, ((0, x_max - self.grid_size[0]), (0, 0)), 'constant', constant_values=pad_value)
            x_max = array.shape[0]
            x_min = x_max - size 
        if y_min < 0:
            array = np.pad(array, ((0, 0), (abs(y_min), 0)), 'constant', constant_values=pad_value)
            y_min = 0
            y_max = size
        elif y_max > self.grid_size[1]:
            array = np.pad(array, ((0, 0), (0, y_max - self.grid_size[1])), 'constant', constant_values=pad_value)
            y_max = array.shape[1]
            y_min = y_max - size

        return array[x_min:x_max, y_min:y_max]

    def _get_observations(self) -> gym.spaces.Dict:
        """
        Returns observations for each agent.

        Returns:
            dict: Observations for each agent.
        """
        if self.observation_type == 'array':
            observations = self._get_array_observations()
        else:
            raise ValueError("Invalid observation type: {}".format(self.observation_type))

        return observations
    
    def _get_array_observations(self):
        # Distance in x and y to goal (normalized and saturated at 32)
        distance_to_goal = {agent: np.zeros(2, dtype=np.float32) for agent in self._agent_ids}
        for agent, pos in self.agent_positions.items():
            raw = np.array(self.goal_positions[agent]) - np.array(pos)
            clipped = np.clip(raw, -32, 32)
            distance_to_goal[agent] = clipped.astype(np.float32) / 32.0

        # Position of other agents arrays
        agents_positions_arrays = {agent: np.zeros(self.grid_size, dtype=np.float32) for agent in self._agent_ids}
        for agent, array in agents_positions_arrays.items():
            for agent2, pos in self.agent_positions.items():
                if agent2 != agent:
                    agents_positions_arrays[agent][pos] = 1.0
        # Crop agent_positionsArrays to observation_size around agent
        for agent, array in agents_positions_arrays.items():
            x, y = self.agent_positions[agent]
            agents_positions_arrays[agent] = self.crop_array(array, x, y, self.observation_size, pad_value=0)
    
        # Crop precomputed obstacles grid for each agent
        obstacles_array = {}
        for agent in self._agent_ids:
            x, y = self.agent_positions[agent]
            obstacles_array[agent] = self.crop_array(self._obstacles_grid, x, y, self.observation_size, pad_value=0).astype(np.float32)

        # Include D* Lite paths only if enabled
        if self.use_d_star_lite:
            paths_for_obs = getattr(self, "d_star_paths", None)
            if not paths_for_obs:
                paths_for_obs = getattr(self, "d_star_planned_paths", {}) or {}
            d_star_path_arrays = {agent: np.zeros((self.observation_size, self.observation_size), dtype=np.float32) for agent in self._agent_ids}
            for agent, path in paths_for_obs.items():
                if agent not in self._agent_ids:
                    continue
                
                array = np.zeros(self.grid_size, dtype=float)
                for i, pos in enumerate(path):
                    array[pos] = i

                # Crop d_star_path_arrays to observation_size around agent
                x, y = self.agent_positions[agent]
                array = self.crop_array(array, x, y, self.observation_size, pad_value=0)

                # Limit and normalize values vectorially
                # array[array > 10] = 10
                # mask = array != 0
                # array = array.astype(np.float32)
                # array[mask] = 1.0 - (array[mask] - 1.0) / 10.0
                array[array>1] = 1.0

                d_star_path_arrays[agent] = array

            others_d_star_path_arrays = {agent: np.zeros((self.observation_size, self.observation_size), dtype=float) for agent in self._agent_ids}
            for agent in self._agent_ids:
                if agent not in self._agent_ids:
                    continue
                
                array = np.zeros(self.grid_size, dtype=float)
                for agent2, path in paths_for_obs.items():
                    if agent2 != agent:
                        for i, pos in enumerate(path):
                            if array[pos] == 0:
                                array[pos] = i
                            elif array[pos] > i:
                                array[pos] = i
                
                # Crop d_star_path_arrays to observation_size around agent
                x, y = self.agent_positions[agent]
                array = self.crop_array(array, x, y, self.observation_size, pad_value=0)
                
                # Limit and normalize values vectorially
                array[array > 5] = 5
                array = array.astype(np.float32)
                mask = array != 0
                array[mask] = 1.0 - (array[mask] - 1.0) / 5.0
                
                
                others_d_star_path_arrays[agent] = array
        else:
            d_star_path_arrays = None

        observations = dict()
        for agent in self._agent_ids:
            if self.config.get("use_cnn_observation", False):
                # Build a stacked HxWxC grid for CNNs.
                if self.use_d_star_lite:
                    grid = np.stack([
                        obstacles_array[agent].astype(np.float32),
                        # agents_positions_arrays[agent].astype(np.float32),
                        d_star_path_arrays[agent].astype(np.float32),
                        others_d_star_path_arrays[agent].astype(np.float32),
                    ], axis=-1)
                else:
                    grid = np.stack([
                        obstacles_array[agent].astype(np.float32),
                        agents_positions_arrays[agent].astype(np.float32),
                    ], axis=-1)

                # Ensure values within [0, 1]
                grid = np.clip(grid, 0.0, 1.0).astype(np.float32)

                observations[agent] = {
                    'grid': grid,
                    'distance_to_goal': distance_to_goal[agent],
                }
            else:
                if self.use_d_star_lite:
                    observations[agent] = {
                        # Include D* Lite paths only if enabled
                        **({'d_star_path': d_star_path_arrays[agent]} if self.use_d_star_lite else {}),
                        **({'d_star_path_others': others_d_star_path_arrays[agent]} if self.use_d_star_lite else {}), # Others' D* paths
                        'distance_to_goal': distance_to_goal[agent],
                        # 'agents_positions': agents_positions_arrays[agent],
                        'obstacles': obstacles_array[agent]
                    }
                else:
                    observations[agent] = {
                        'distance_to_goal': distance_to_goal[agent],
                        'agents_positions': agents_positions_arrays[agent],
                        'obstacles': obstacles_array[agent]
                    }
        return observations
    
    def get_agent_ids(self) -> set:
        """Returns a set of agent ids in the environment.

        Returns:
            Set of agent ids.
        """
        if not isinstance(self._agent_ids, set):
            self._agent_ids = set(self._agent_ids)
        return self._agent_ids

    def get_current_map_name(self) -> Optional[str]:
        """Return the map name used for the most recent reset."""
        return self._current_map_name

    def render(self, clear=True,
               title="RLMAPF Environment", 
               save_frames=False, 
               frames_path="frames/", 
               save_video=False, 
               video_path="render.mp4", 
               show_render=False, 
               render_delay=0.2, 
               include_legend=True, 
               legend_position=(0, 0),
               smooth_motion=False,
               motion_frames=5):
        """
        Renders the environment state using matplotlib.
        If save_video is True, saves the rendered frames as a video.
        """
        if not hasattr(self, "_fig"):
            self._fig, self._ax = plt.subplots(figsize=(8, 8))
            self._ax.set_xlim(0, self.grid_size[0])
            self._ax.set_ylim(0, self.grid_size[1])
            self._ax.set_aspect('equal')
            self._ax.set_xticks(range(self.grid_size[0] + 1))
            self._ax.set_yticks(range(self.grid_size[1] + 1))
            self._ax.grid(True)
            self._ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            self._patches = []
            self._texts = []
            self._prev_agent_positions = {}

        if not hasattr(self, "_prev_agent_positions"):
            self._prev_agent_positions = {}
        if not self._prev_agent_positions:
            self._prev_agent_positions = self.agent_positions.copy()

        frames_to_render = motion_frames if smooth_motion else 1

        def _agent_sort_key(agent_id: str) -> int:
            try:
                return int(agent_id.split("_")[1])
            except (IndexError, ValueError):
                return float("inf")

        for frame_idx in range(frames_to_render):
            rendered_paths_info = []
            if smooth_motion and motion_frames > 1:
                t = frame_idx / (motion_frames - 1)
            else:
                t = 1.0

            self._ax.set_title(title, fontsize=14)

            for patch in self._patches:
                patch.remove()
            for text in self._texts:
                text.remove()
            self._patches.clear()
            self._texts.clear()

            for obst in self.obstacles:
                rect = Rectangle((obst[0], obst[1]), 1, 1, color="gray")
                self._ax.add_patch(rect)
                self._patches.append(rect)

            paths_for_render = getattr(self, "d_star_paths", None) or getattr(self, "d_star_planned_paths", None)
            if self.use_d_star_lite and paths_for_render:
                path_styles = getattr(self, "_path_styles", {})
                for agent in sorted(paths_for_render.keys(), key=_agent_sort_key):
                    path = paths_for_render.get(agent, [])
                    if not path:
                        continue
                    style = path_styles.get(agent, {}) or {}
                    color = style.get("color", "red")
                    linestyle = style.get("linestyle", "--")
                    xs = [pos[0] + 0.5 for pos in path]
                    ys = [pos[1] + 0.5 for pos in path]
                    line, = self._ax.plot(
                        xs,
                        ys,
                        linestyle=linestyle,
                        linewidth=2,
                        color=color,
                        alpha=0.85,
                        zorder=2,
                    )
                    self._patches.append(line)
                    agent_nr = agent[len("agent_"):] if agent.startswith("agent_") else agent
                    color_hex = style.get("color_hex")
                    if color_hex is None:
                        try:
                            color_hex = mcolors.to_hex(color)
                        except ValueError:
                            color_hex = "#ff0000"
                    rendered_paths_info.append({
                        "agent_nr": agent_nr,
                        "style_label": style.get("style_label", linestyle),
                        "color_hex": color_hex,
                    })

            for agent, pos in self.goal_positions.items():
                agent_nr = agent[len("agent_"):]
                if not self.dones[agent]:
                    rect = Rectangle((pos[0], pos[1]), 1, 1, color="blue")
                else:
                    rect = Rectangle((pos[0], pos[1]), 1, 1, color="green")
                self._ax.add_patch(rect)
                self._patches.append(rect)
                text_size = 6 if len(agent_nr) > 1 else 8
                text = self._ax.text(pos[0] + 0.5, pos[1] + 0.45, agent_nr, color="white", 
                                ha="center", va="center", zorder=1, fontsize=text_size)
                self._texts.append(text)
            
            for agent, new_pos in self.agent_positions.items():
                agent_nr = agent[len("agent_"):]
                if not self.dones[agent]:
                    old_pos = self._prev_agent_positions.get(agent, new_pos)
                    interp_x = old_pos[0] + (new_pos[0] - old_pos[0]) * t
                    interp_y = old_pos[1] + (new_pos[1] - old_pos[1]) * t
                    rect = Circle((interp_x + 0.5, interp_y + 0.5), 0.4, facecolor="yellow", edgecolor="black")
                    self._ax.add_patch(rect)
                    self._patches.append(rect)
                    text_size = 6 if len(agent_nr) > 1 else 8
                    text = self._ax.text(interp_x + 0.5, interp_y + 0.45, agent_nr, color="black", 
                                    ha="center", va="center", fontsize=text_size)
                    self._texts.append(text)

            legend_lines = [
                "Legend:",
                "Yellow: Robot",
                "Blue: Goal",
                "Green: Reached Goal",
                "Gray: Obstacle",
            ]
            if rendered_paths_info:
                legend_lines.append("D* paths (color/style):")
                for info in rendered_paths_info:
                    legend_lines.append(f"Agent {info['agent_nr']}: {info['color_hex']} {info['style_label']}")
            legend_text = "\n".join(legend_lines)
            if include_legend:
                if hasattr(self, "_legend_text_obj"):
                    self._legend_text_obj.set_text(legend_text)
                else:
                    self._legend_text_obj = self._ax.text(
                        legend_position[0],
                        self.grid_size[1] + legend_position[1],
                        legend_text,
                        fontsize=10,
                        va="top",
                        ha="left",
                        bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'),
                    )
            elif hasattr(self, "_legend_text_obj"):
                self._legend_text_obj.remove()
                del self._legend_text_obj

            step_text = f"Step: {self.steps}"
            text_offset = len(step_text) * 0.3
            if hasattr(self, "_step_text_obj"):
                self._step_text_obj.set_text(step_text)
                self._step_text_obj.set_position((self.grid_size[0] + 3 - text_offset, self.grid_size[1] + 3))
            else:
                self._step_text_obj = self._ax.text(self.grid_size[0] + 3 - text_offset, self.grid_size[1] + 3, 
                                                step_text, fontsize=10, va="top")

            if show_render:
                plt.draw()
                plt.pause(render_delay / frames_to_render)

            if save_frames:
                if not os.path.exists(frames_path):
                    os.makedirs(frames_path)
                if smooth_motion:
                    frame_path = os.path.join(frames_path, f"step_{self.steps:04d}_frame_{frame_idx:02d}")
                else:
                    frame_path = os.path.join(frames_path, f"step_{self.steps:04d}")
                frame_format = self.render_config.get("frame_format", "png")
                frame_dpi = self.render_config.get("frame_dpi", 150)
                frame_path_with_ext = f"{frame_path}.{frame_format}"
                plt.savefig(frame_path_with_ext, format=frame_format, bbox_inches="tight", dpi=frame_dpi)

            if save_video:
                if not hasattr(self, "_video_writer"):
                    fps = int(self.render_config.get("video_fps", 10))
                    if smooth_motion:
                        fps = fps * motion_frames
                    self._video_writer = FFMpegWriter(fps=fps, metadata=dict(artist='RLMAPF2 Sim'), bitrate=1800)
                    self._video_writer.setup(self._fig, video_path, dpi=self.render_config["video_dpi"])
                self._video_writer.grab_frame()

            if clear and frame_idx == frames_to_render - 1:
                self._ax.cla()

        self._prev_agent_positions = self.agent_positions.copy()

    def finalize_video(self):
        """
        Finalizes and closes the video file.
        Call this at the end of your episode.
        """
        if hasattr(self, "_video_writer"):
            self._video_writer.finish()
            del self._video_writer

    def _maybe_capture_d_star_step_frame(self, is_reset: bool) -> None:
        """
        Capture the current D* Lite planner state into a GIF/PNG (debug only).
        """
        debug_cfg = self.config.get("d_star_debug", {})
        if not debug_cfg.get("save_gif", False):
            return
        if not self.use_d_star_lite:
            return
        try:
            import imageio
            import matplotlib.pyplot as plt
            from matplotlib.patches import Rectangle
            from d_star_lite import OBSTACLE
        except ImportError as exc:
            print(f"[D* Lite debug] Skipping GIF capture (missing dependency): {exc}")
            return

        if not hasattr(self, "grid_size") or self.grid_size is None:
            return

        obstacle_grid = np.zeros(self.grid_size, dtype=np.uint8)
        for pos in getattr(self, "obstacles", []):
            obstacle_grid[pos] = OBSTACLE

        traversal_costs = None
        if hasattr(self, "d_star_maps") and self.d_star_maps:
            some_map = next(iter(self.d_star_maps.values()))
            traversal_costs = getattr(some_map, "traversal_costs", None)
        if traversal_costs is None:
            traversal_costs = np.ones(self.grid_size, dtype=np.float32)

        agent_ids = sorted(self.agent_positions.keys())
        paths_source = getattr(self, "d_star_paths", None) or getattr(self, "d_star_planned_paths", {})
        paths = {agent: list(paths_source.get(agent, [])) for agent in agent_ids}

        fig, ax = plt.subplots(figsize=(6, 6))
        costs = traversal_costs.T
        im = ax.imshow(costs, origin="lower", cmap="viridis", interpolation="none")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Traversal multiplier")

        obstacle_positions = np.argwhere(obstacle_grid == OBSTACLE)
        for ox, oy in obstacle_positions:
            rect = Rectangle((ox - 0.5, oy - 0.5), 1.0, 1.0, facecolor="dimgray", edgecolor="black")
            ax.add_patch(rect)

        color_map = plt.colormaps.get_cmap("tab10")
        for idx, agent in enumerate(agent_ids):
            path = paths.get(agent, [])
            color = color_map(idx % color_map.N)
            if path:
                xs = [cell[0] for cell in path]
                ys = [cell[1] for cell in path]
                ax.plot(xs, ys, "-o", color=color, linewidth=2, markersize=4, label=agent)

            start = self.agent_positions.get(agent)
            goal = self.goal_positions.get(agent)
            if start is not None:
                ax.scatter(start[0], start[1], marker="s", s=80, color=color, edgecolor="black")
            if goal is not None:
                ax.scatter(goal[0], goal[1], marker="*", s=120, color=color, edgecolor="black")

        ax.set_title(f"Step {self.steps}{' (reset)' if is_reset else ''}")
        ax.set_xlim(-0.5, self.grid_size[0] - 0.5)
        ax.set_ylim(-0.5, self.grid_size[1] - 0.5)
        ax.set_aspect("equal")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.legend(loc="upper right")
        ax.grid(False)

        fig.canvas.draw()
        if hasattr(fig.canvas, "tostring_rgb"):
            buf = fig.canvas.tostring_rgb()
            w, h = fig.canvas.get_width_height()
            frame = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 3))
        else:
            rgba = np.asarray(fig.canvas.buffer_rgba())
            frame = rgba[..., :3].copy()
            h, w, _ = frame.shape

        if self._dstar_gif_writer is None:
            fps = max(1, int(debug_cfg.get("gif_fps", 5)))
            gif_path = debug_cfg.get("gif_path", "dstar_debug.gif")
            self._dstar_gif_writer = imageio.get_writer(gif_path, mode="I", duration=1.0 / fps)

        self._dstar_gif_writer.append_data(frame)

        if debug_cfg.get("save_pngs", False):
            png_dir = Path(debug_cfg.get("png_dir", "dstar_debug_frames"))
            png_dir.mkdir(parents=True, exist_ok=True)
            png_path = png_dir / f"frame_{self._dstar_debug_frame_idx:05d}.png"
            fig.savefig(png_path, dpi=200, bbox_inches="tight")

        self._dstar_debug_frame_idx += 1
        plt.close(fig)

    def _next_map_name(self) -> str:
        if not self.config["maps_names_with_variants"]:
            raise ValueError("No maps to load")
        map_names = list(self.config["maps_names_with_variants"].keys())
        if not self.cycle_maps_without_replacement:
            return str(np.random.choice(map_names))
        if not self._map_cycle_queue:
            self._map_cycle_queue = map_names[:]
            if self.shuffle_map_cycle:
                np.random.shuffle(self._map_cycle_queue)
        if self.shuffle_map_cycle:
            return str(self._map_cycle_queue.pop())
        return str(self._map_cycle_queue.pop(0))

    def load_map(self, map_name: Optional[str] = None):
        # Choose map and load map data (cached in memory to avoid repeated disk I/O)
        if map_name is None:
            map_name = self._next_map_name()

        if map_name not in self.config["maps_names_with_variants"]:
            raise ValueError("Map {} not configured".format(map_name))

        random_map_path = self._resolve_map_path(self.config["map_path"], map_name)
        map_data = deepcopy(self._load_map_data(random_map_path))
        self._current_map_name = map_name
        self._map_usage_counter.setdefault(map_name, 0)
        self._map_usage_counter[map_name] += 1

        # Choose random variant
        variants = self.config["maps_names_with_variants"][map_name]
        if variants is not None and len(variants) == 0:
            raise ValueError("Configured variants list for map {} is empty".format(map_name))
        if variants is not None:
            random_variant = np.random.choice(variants)
        else:
            # Choose random variant from all variants
            available_variants = list(map_data["map_variant"].keys())
            if not available_variants:
                raise ValueError("No variants found for map {}".format(map_name))
            random_variant = np.random.choice(available_variants)

        if random_variant is None:
            raise ValueError("No variants found for map {}".format(map_name))
        
        # Check number of agents:
        if self.config["agents_num"] < map_data["metadata"]["min_num_of_agents"]:
            raise ValueError("Number of agents is less than minimum number of agents for map ({})".format(map_data["metadata"]["min_num_of_agents"]))
        if self.config["agents_num"] > map_data["metadata"]["max_num_of_agents"]:
            raise ValueError("Number of agents is greater than maximum number of agents for map ({})".format(map_data["metadata"]["max_num_of_agents"]))

        self.obstacles = map_data["map_variant"][str(random_variant)]["obstacles"]
        self.agent_positions = map_data["map_variant"][str(random_variant)]["starting_positions"]
        self.goal_positions = map_data["map_variant"][str(random_variant)]["goal_positions"]

        if self.print_map_usage:
            usage_summary = ", ".join(
                f"{name}: {count}" for name, count in sorted(self._map_usage_counter.items())
            )
            print(
                "[RLMAPF] Loaded map '{}' (variant {}). Usage totals -> {}".format(
                    map_name,
                    random_variant,
                    usage_summary if usage_summary else "no data",
                )
            )

        # Convert from number to agent_number
        self.agent_positions = {"agent_" + str(agent): tuple(pos) for agent, pos in self.agent_positions.items()}
        self.goal_positions = {"agent_" + str(agent): tuple(pos) for agent, pos in self.goal_positions.items()}

        # Convert to set
        self.obstacles = {(pos[0], pos[1]) for pos in self.obstacles}

        self.grid_size = (map_data["metadata"]["width"], map_data["metadata"]["height"])

    def get_map_usage_counts(self) -> Dict[str, int]:
        """Return a copy of cumulative map usage counts for the current env instance."""
        return dict(self._map_usage_counter)
        
    def verify_map_sizes(self):
        self.grid_size = None
        # Check if all maps have the same size
        for map_name in self.config["maps_names_with_variants"]:
            map_path = self._resolve_map_path(self.config["map_path"], map_name)
            map_data = self._load_map_data(map_path)
            grid_size = (map_data["metadata"]["width"], map_data["metadata"]["height"])
            if self.grid_size is None:
                self.grid_size = grid_size
            elif self.grid_size != grid_size:
                raise ValueError("Map sizes do not match. Expected: {}, Found: {}, when processing {}".format(self.grid_size, grid_size, map_name))

    def close(self):
        if getattr(self, "_dstar_gif_writer", None) is not None:
            try:
                self._dstar_gif_writer.close()
            except Exception:
                pass
            self._dstar_gif_writer = None
        try:
            super().close()
        except Exception:
            pass


def _calculate_overlap_percentages(
    paths: Dict[str, List[Tuple[int, int]]]
) -> Tuple[Dict[str, float], float, int, float]:
    """
    Compute per-agent and overall overlap percentages for the supplied paths.
    Mirrors the helper from run_dlite_on_map.py for quick debugging output.
    """
    if not paths:
        return {}, 0.0, 0, 0.0

    path_sets: Dict[str, set[Tuple[int, int]]] = {agent: set(path) for agent, path in paths.items()}

    per_agent: Dict[str, float] = {}
    for agent, cells in path_sets.items():
        if not cells:
            per_agent[agent] = 0.0
            continue

        other_union = set().union(*(path_sets[a] for a in path_sets if a != agent))
        overlap_count = len(cells & other_union)
        per_agent[agent] = 100.0 * overlap_count / len(cells)

    cell_counts: Dict[Tuple[int, int], int] = {}
    for cells in path_sets.values():
        for cell in cells:
            cell_counts[cell] = cell_counts.get(cell, 0) + 1

    overlapped_cells = sum(1 for count in cell_counts.values() if count > 1)
    total_unique_cells = len(cell_counts)
    overall = 100.0 * overlapped_cells / total_unique_cells if total_unique_cells else 0.0
    average = float(np.mean(list(per_agent.values()))) if per_agent else 0.0

    return per_agent, overall, overlapped_cells, average


def _debug_print_d_star_paths(env: RLMAPF, label: str) -> None:
    """
    Print D* Lite path details for debugging, similar to run_dlite_on_map.py output.
    """
    if not getattr(env, "use_d_star_lite", False):
        print(f"[D* Lite debug] {label}: D* Lite disabled; skipping path print.")
        return

    paths = getattr(env, "d_star_paths", None) or getattr(env, "d_star_planned_paths", None) or {}
    if not paths:
        print(f"[D* Lite debug] {label}: no D* Lite paths available.")
        return

    map_name = env.get_current_map_name() or "unknown"
    grid = getattr(env, "grid_size", None)
    grid_label = f"{grid[0]}x{grid[1]}" if grid else "unknown"
    print(f"[D* Lite debug] {label} | map: {map_name} | grid: {grid_label}")

    sample_map = None
    if hasattr(env, "d_star_maps") and env.d_star_maps:
        sample_map = next(iter(env.d_star_maps.values()))
    if sample_map is not None and hasattr(sample_map, "traversal_costs"):
        traversal_costs = sample_map.traversal_costs
        min_cost = float(np.min(traversal_costs))
        max_cost = float(np.max(traversal_costs))
        print(f"  Traversal multiplier range [{min_cost:.2f}, {max_cost:.2f}]")
    else:
        print("  Traversal multiplier stats unavailable.")

    def _agent_sort_key(agent_id: str) -> int:
        try:
            return int(agent_id.split("_")[1])
        except (IndexError, ValueError):
            return float("inf")

    for agent_id in sorted(paths.keys(), key=_agent_sort_key):
        path = paths.get(agent_id, [])
        length = len(path)
        if length > 0:
            print(f"  {agent_id}: path length {length}, start {path[0]} -> goal {path[-1]}")
            print(f"    path: {path}")
        else:
            print(f"  {agent_id}: path length {length}, empty path")

    per_agent_overlap, overall_overlap, total_overlap_cells, avg_overlap = _calculate_overlap_percentages(paths)
    formatted_agents = ", ".join(
        f"{agent}: {pct:.1f}%"
        for agent, pct in sorted(per_agent_overlap.items(), key=lambda item: _agent_sort_key(item[0]))
    )
    print(
        f"  Overlap: {total_overlap_cells} cells ({overall_overlap:.1f}%) | avg {avg_overlap:.1f}%"
        + (f" | {formatted_agents}" if formatted_agents else "")
    )


def _visualize_d_star_iterations(
    env: RLMAPF,
    save_dir: str = "dstar",
    pause_seconds: float = 0.0,
    display: bool = False,
) -> None:
    """
    Render D* Lite congestion iterations as PNGs, matching run_dlite_on_map output.
    """
    if not getattr(env, "use_d_star_lite", False):
        print("[D* Lite debug] Visualization skipped: D* Lite disabled.")
        return

    try:
        from pathlib import Path
        from d_star_lite import OBSTACLE
        from run_dlite_on_map import run_iterative_planning, visualize_iterations
    except ImportError as exc:
        print(f"[D* Lite debug] Visualization skipped: imports failed ({exc}).")
        return

    if not hasattr(env, "grid_size") or env.grid_size is None:
        print("[D* Lite debug] Visualization skipped: grid_size missing.")
        return

    obstacle_grid = np.zeros(env.grid_size, dtype=np.uint8)
    for pos in getattr(env, "obstacles", []):
        obstacle_grid[pos] = OBSTACLE

    starts = {agent: tuple(pos) for agent, pos in env.agent_positions.items()}
    goals = {agent: tuple(pos) for agent, pos in env.goal_positions.items()}

    iterations = max(1, int(env.config.get("d_star_iterations", 1)))
    congestion_weight = float(env.config.get("d_star_congestion_weight", 1.0))

    history = run_iterative_planning(
        env.grid_size[0],
        env.grid_size[1],
        obstacle_grid,
        starts,
        goals,
        iterations=iterations,
        congestion_weight=congestion_weight,
    )

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    visualize_iterations(
        env.grid_size[0],
        env.grid_size[1],
        obstacle_grid,
        starts,
        goals,
        history,
        display=display,
        pause_seconds=pause_seconds,
        save_dir=save_path,
        block=False,
    )
    print(f"[D* Lite debug] Saved iteration frames to {save_path}/iteration_*.png")


def _visualize_env_planner_state(
    env: RLMAPF,
    save_dir: str = "dstar",
    display: bool = False,
) -> None:
    """
    Render the current env-held D* Lite paths/traversal costs (no recomputation).
    """
    if not getattr(env, "use_d_star_lite", False):
        print("[D* Lite debug] Visualization skipped: D* Lite disabled.")
        return

    try:
        from pathlib import Path
        from d_star_lite import OBSTACLE
        from run_dlite_on_map import visualize_iterations
    except ImportError as exc:
        print(f"[D* Lite debug] Visualization skipped: imports failed ({exc}).")
        return

    if not hasattr(env, "grid_size") or env.grid_size is None:
        print("[D* Lite debug] Visualization skipped: grid_size missing.")
        return

    obstacle_grid = np.zeros(env.grid_size, dtype=np.uint8)
    for pos in getattr(env, "obstacles", []):
        obstacle_grid[pos] = OBSTACLE

    starts = {agent: tuple(pos) for agent, pos in env.agent_positions.items()}
    goals = {agent: tuple(pos) for agent, pos in env.goal_positions.items()}

    traversal_costs = None
    if hasattr(env, "d_star_maps") and env.d_star_maps:
        some_map = next(iter(env.d_star_maps.values()))
        traversal_costs = getattr(some_map, "traversal_costs", None)
    if traversal_costs is None:
        traversal_costs = np.ones(env.grid_size, dtype=np.float32)

    history = [
        {
            "iteration": 1,
            "paths": {agent: list(path) for agent, path in (getattr(env, "d_star_paths", None) or getattr(env, "d_star_planned_paths", {})).items()},
            "traversal_costs": traversal_costs.copy(),
        }
    ]

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    visualize_iterations(
        env.grid_size[0],
        env.grid_size[1],
        obstacle_grid,
        starts,
        goals,
        history,
        display=display,
        pause_seconds=0.0,
        save_dir=save_path,
        block=False,
    )
    print(f"[D* Lite debug] Saved current planner state to {save_path}/iteration_01.png")


if __name__ == "__main__":
    # Example usage
    env_config = {
        "agents_num": 10,
        "render_mode": "human",
        "max_steps": 250,
        "observation_type": "array",
        "maps_names_with_variants": {
            # "warehouse0_1-10a-21x21": None,
            # "maze2_1-100a-30x30": None,
            "slots_10-10a-39x29": None,
            # "congested_center_10-10a-49x23": None,
            # "corridors_10-10a-32x32": [0],
            # "circle_10-10a-17x17": None,
            # "empty_1-4a-5x4": None,
        },
        "render_config": {
            "title": "RLMAPF2",
            "show_render": False,
            "save_video": True,
            "include_legend": True,
            "legend_position": (0, 0),
            "video_path": "render.mp4",
            "video_fps": 10,
            "video_dpi": 300,
            "render_delay": 0.2,
            "save_frames": True,
            "frames_path": "frames/",
        },
        "d_star_debug": {
            "save_gif": True,
            "gif_path": "dstar_debug.gif",
            "gif_fps": 5,
            "save_pngs": False,
            "png_dir": "dstar_debug_frames",
        },
        # "seed": 42,
    }

    print("Initializing environment...")

    env = RLMAPF(env_config)

    print("Environment initialized.")

    # sample_obs= env.observation_space.sample()
    # actual_obs, _ = env.reset()
    # print("Sample observation:", sample_obs)
    # print("Actual observation:", actual_obs)
    # exit()

    for i in range(1):
        obs, _ = env.reset()
        # _debug_print_d_star_paths(env, label=f"Initial D* Lite plan (episode {i})")
        _visualize_env_planner_state(env, save_dir="dstar_env", display=False)
        _visualize_d_star_iterations(env, save_dir="dstar", pause_seconds=0.5, display=False)
        total_rewards = {agent: 0 for agent in env.get_agent_ids()}
        last_terminateds = dict()    
        terminateds = dict()
        truncateds = dict()
        terminateds['__all__'] = False
        truncateds['__all__'] = False
        print("Episode {} started!".format(i))
        pbar = tqdm(total=env.max_steps, desc="Episode {}".format(i), unit="step")
        while not terminateds['__all__'] and not truncateds['__all__']:
            pbar.update(1)
            pbar.set_postfix({"Mean rewards": np.mean(list(total_rewards.values()))})
            actions = {agent: env.action_space.sample() for agent in env.get_agent_ids()}
            obs, reward, terminateds, truncateds, info = env.step(actions)

            for agent in env.get_agent_ids():
                total_rewards[agent] += reward[agent]
            
            # env.render(clear=False)
            # print("Info:", info)
            # if (any(terminateds.values())):
            #     print("     terminateds {}".format(terminateds))
            #     print("Last terminateds {}".format(last_terminateds))
            last_terminateds = terminateds
        pbar.close()

        print("Episode {} finished! Rewards:".format(i), total_rewards)

    env.close()
