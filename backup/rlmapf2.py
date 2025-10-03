from copy import deepcopy
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
from d_star_lite import DStarLite, OccupancyGridMap

import matplotlib.pyplot as plt

from matplotlib.patches import Rectangle, Circle
from matplotlib.animation import FFMpegWriter

from tqdm import tqdm

class RLMAPF(MultiAgentEnv):
    """
    RLMAPF environment for multi-agent pathfinding using D* Lite algorithm.
    """
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

        # Set seed
        if self.config["seed"] is not None:
            self._seed = self.config["seed"]
        else:
            self._seed = np.random.randint(0, 1e9)

        # Check if all maps have the same size
        self.verify_map_sizes()

        # Initialize environment variables
        self.reset(self._seed)
        
        # Define agent observation and action spaces
        self.define_observation_spaces()
        self.action_space = gym.spaces.Discrete(5)
        
        # Initialize last_actions to track agent movement directions
        self.last_actions = {}

        # Initialize D* Lite-related variables only if enabled
        if self.use_d_star_lite:
            self.d_star_maps = {}
            self.d_star_paths = {}
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
            "penalize_left_side_bottom_passing": False,  # Penalize left-side/bottom passing
        }
        return default_config

    def define_observation_spaces(self):
        if self.observation_type == 'array':
            if self.config.get("use_cnn_observation", False):
                # Stack grid-based features into channels for CNN consumption.
                # Channels: obstacles, agents_positions, and optionally
                # d_star_path, d_star_path_others when D* Lite is enabled.
                # channels = 2 + (2 if self.use_d_star_lite else 0) 
                channels = 3 if self.use_d_star_lite else 2 # NO OTHER D* PATHS, CHANGE BACK TO 4
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
                
    def reset(self, seed=None, options=None):
        """
        Resets the environment to its initial state.

        Returns:
            dict: Observations for each agent.
        """
        if seed is not None:
            self.set_seed(seed)

        # # Reset environment variables
        # # TODO: REMOVE
        # self.config["agents_num"] = 8
        # # TODO: REMOVE

        self.steps = 0
        self._agent_ids = set({"agent_" + str(i) for i in range(self.config["agents_num"])})
        self.rewards = {agent: 0 for agent in self._agent_ids}
        self.dones = {agent: False for agent in self._agent_ids}
        self.dones['__all__'] = False
        self.number_of_collisions = {agent: 0 for agent in self._agent_ids}
        self.number_of_steps = {agent: 0 for agent in self._agent_ids}

        # Initialize empty agents, agent positions and goal positions
        self.agent_positions = dict()
        self.goal_positions = dict()
        self.obstacles = set()

        # Load map
        if self.config["maps_names_with_variants"] is not None and len(self.config["maps_names_with_variants"]) > 0:
            self.load_map()
        else:
            raise ValueError("No maps to load")
        
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
            self.d_star_maps = {agent: OccupancyGridMap(self.grid_size[0], self.grid_size[1]) for agent in self._agent_ids}
            self.d_star_paths = {agent: [] for agent in self._agent_ids}
            obstacles_array = np.zeros(self.grid_size, dtype=int)
            for pos in self.obstacles:
                obstacles_array[pos] = 255
            for agent in self._agent_ids:
                self.d_star_maps[agent].set_map(obstacles_array)
            self.d_stars = {agent: DStarLite(self.d_star_maps[agent], self.agent_positions[agent], self.goal_positions[agent]) for agent in self._agent_ids}
            for agent in self._agent_ids:
                path, _, _ = self.d_stars[agent].move_and_replan(self.agent_positions[agent])
                self.d_star_paths[agent] = path
            self.start_d_star_paths = deepcopy(self.d_star_paths)

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
                        legend_position=self.render_config["legend_position"])

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

        def penalize(agent, penalty=self.collision_penalty):
            self.rewards[agent] -= penalty

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
                    penalize(agent)
                new_pos = current_state[agent]  # Revert to the current position
            
            # Penalize for moving
            if action != 4:
                if self.config["penalize_waiting"]:
                    penalize(agent, penalty=self.step_cost)
            else:
                if self.config["penalize_steps"]:
                    penalize(agent, penalty=self.step_cost*self.wait_cost_multiplier)

            new_state[agent] = new_pos

        # Check for collisions
        for agent, new_pos in new_state.items():
            # Check for collisions with obstacles
            if new_pos in self.obstacles:
                if self.config["penalize_collision"]:
                    penalize(agent)
                self.number_of_collisions[agent] += 1
                new_state[agent] = current_state[agent]

            for agent2, new_pos2 in new_state.items():
                if agent2 != agent and new_pos2 == current_state[agent] and new_pos == current_state[agent2]:
                    if self.config["penalize_collision"]:
                        penalize(agent)
                        penalize(agent2)
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
                            penalize(agent_to_penalize)
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
                                penalize(agent1, penalty=0.1)
                                penalize(agent2, penalty=0.1)
                            elif action2 == 0 and pos2_prev[0] < pos1_prev[0]:
                                penalize(agent1, penalty=0.1)
                                penalize(agent2, penalty=0.1)

                    # Left/Right passing
                    if (action1 == 2 and action2 == 3) or (action1 == 3 and action2 == 2):
                        # Check if they are on the same row and adjacent columns
                        if pos1_prev[1] == pos2_prev[1] and abs(pos1_prev[0] - pos2_prev[0]) == 1:
                            # Penalize when the agent going left is below (higher y)
                            if action1 == 2 and pos1_prev[1] > pos2_prev[1]:
                                penalize(agent1, penalty=0.1)
                                penalize(agent2, penalty=0.1)
                            elif action2 == 2 and pos2_prev[1] > pos1_prev[1]:
                                penalize(agent1, penalty=0.1)
                                penalize(agent2, penalty=0.1)
        # Update D* Lite paths only if enabled
        if self.use_d_star_lite:
            for agent in self._agent_ids:
                path, _, _ = self.d_stars[agent].move_and_replan(self.agent_positions[agent])
                self.d_star_paths[agent] = path

        for agent, new_pos in self.agent_positions.items():
            # Check for reaching goal positions and update rewards
            if new_pos == self.goal_positions[agent] and not self.dones[agent]:
                reward(agent) 
                terminateds[agent] = True
                self.dones[agent] = True

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
                    d_star_distance_current = len(self.d_star_paths[agent])
                    d_star_distance_start = len(self.start_d_star_paths[agent])

                    d_star_reward = 1.0 - d_star_distance_current / d_star_distance_start
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
                        legend_position=self.render_config["legend_position"])
            

        return self._get_observations(), self.rewards, terminateds, truncateds, self._get_info()
    
    def _get_info(self):
        """
        Returns info for 

        Returns:
            dict: Info
        """
        info_dict = dict()
        for agent in self._agent_ids:
            obstacles_array = np.zeros(self.grid_size, dtype=int)
            for pos in self.obstacles:
                obstacles_array[pos] = 1


            info_dict[agent] = {
                "current_position": self.agent_positions[agent],
                "number_of_collisions": self.number_of_collisions[agent],
                "number_of_steps": self.number_of_steps[agent],
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
            d_star_path_arrays = {agent: np.zeros((self.observation_size, self.observation_size), dtype=np.float32) for agent in self._agent_ids}
            for agent, path in self.d_star_paths.items():
                if agent not in self._agent_ids:
                    continue
                
                array = np.zeros(self.grid_size, dtype=float)
                for i, pos in enumerate(path):
                    array[pos] = i

                # Crop d_star_path_arrays to observation_size around agent
                x, y = self.agent_positions[agent]
                array = self.crop_array(array, x, y, self.observation_size, pad_value=0)

                # Limit and normalize values vectorially
                array[array > 10] = 10
                mask = array != 0
                array = array.astype(np.float32)
                array[mask] = 1.0 - (array[mask] - 1.0) / 10.0
                
                d_star_path_arrays[agent] = array

            others_d_star_path_arrays = {agent: np.zeros((self.observation_size, self.observation_size), dtype=float) for agent in self._agent_ids}
            for agent in self._agent_ids:
                if agent not in self._agent_ids:
                    continue
                
                array = np.zeros(self.grid_size, dtype=float)
                for agent2, path in self.d_star_paths.items():
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
                array[array > 10] = 10
                array = array.astype(np.float32)
                mask = array != 0
                array[mask] = 1.0 - (array[mask] - 1.0) / 10.0
                
                
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
                        agents_positions_arrays[agent].astype(np.float32),
                        d_star_path_arrays[agent].astype(np.float32),
                        # others_d_star_path_arrays[agent].astype(np.float32),
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
                        'agents_positions': agents_positions_arrays[agent],
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

    def render(self, clear=True,
               title="RLMAPF Environment", 
               save_frames=False, 
               frames_path="frames/", 
               save_video=False, 
               video_path="render.mp4", 
               show_render=False, 
               render_delay=0.2, 
               include_legend=True, 
               legend_position=(0, 0)):
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

        # Set the title
        self._ax.set_title(title, fontsize=14)

        # Clear previous patches and texts
        for patch in self._patches:
            patch.remove()
        for text in self._texts:
            text.remove()
        self._patches.clear()
        self._texts.clear()

        # Draw obstacles
        for obst in self.obstacles:
            rect = Rectangle((obst[0], obst[1]), 1, 1, color="gray")
            self._ax.add_patch(rect)
            self._patches.append(rect)

        # Draw goals
        for agent, pos in self.goal_positions.items():
            agent_nr = agent[len("agent_"):]
            if not self.dones[agent]:
                rect = Rectangle((pos[0], pos[1]), 1, 1, color="blue")
            else:
                rect = Rectangle((pos[0], pos[1]), 1, 1, color="green")
            self._ax.add_patch(rect)
            self._patches.append(rect)
            text_size = 6 if len(agent_nr) > 1 else 8  # Adjust text size for double-digit numbers
            text = self._ax.text(pos[0] + 0.5, pos[1] + 0.45, agent_nr, color="white", ha="center", va="center", zorder=1, fontsize=text_size)
            self._texts.append(text)
        
        # Draw agents
        for agent, pos in self.agent_positions.items():
            agent_nr = agent[len("agent_"):]
            if not self.dones[agent]:
                rect = Circle((pos[0] + 0.5, pos[1] + 0.5), 0.4, facecolor="yellow", edgecolor="black")
                self._ax.add_patch(rect)
                self._patches.append(rect)
                text_size = 6 if len(agent_nr) > 1 else 8  # Adjust text size for double-digit numbers
                text = self._ax.text(pos[0] + 0.5, pos[1] + 0.45, agent_nr, color="black", ha="center", va="center", fontsize=text_size)
                self._texts.append(text)
                    

        # Add legend
        if not hasattr(self, "_legend_text_obj") and include_legend:
            legend_text = (
                "Legend:\n"
                "Yellow: Robot\n"
                "Blue: Goal\n"
                "Green: Reached Goal\n"
                "Gray: Obstacle\n"
            )
            # Put legend text in the top left corner
            self._legend_text_obj = self._ax.text(legend_position[0], self.grid_size[1] + legend_position[1], legend_text, fontsize=10, va="top", ha="left", bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))

        # Add step number
        step_text = f"Step: {self.steps}"
        text_offset = len(step_text) * 0.3  # Adjust offset based on text length
        if hasattr(self, "_step_text_obj"):
            self._step_text_obj.set_text(step_text)
            self._step_text_obj.set_position((self.grid_size[0] + 3 - text_offset, self.grid_size[1] + 3))
        else:
            self._step_text_obj = self._ax.text(self.grid_size[0] + 3 - text_offset, self.grid_size[1] + 3, step_text, fontsize=10, va="top")

        # Update the plot
        if show_render:
            plt.draw()
            plt.pause(render_delay)

        # Save frames if enabled
        if save_frames:
            if not os.path.exists(frames_path):
                os.makedirs(frames_path)
            frame_path = os.path.join(frames_path, f"frame_{self.steps}.svg")
            plt.savefig(frame_path, format="svg", bbox_inches="tight", dpi=300)

        # Save video if enabled
        if save_video:
            if not hasattr(self, "_video_writer"):
                fps = int(self.render_config.get("video_fps", 10))
                self._video_writer = FFMpegWriter(fps=fps, metadata=dict(artist='RLMAPF2 Sim'), bitrate=1800)
                self._video_writer.setup(self._fig, video_path, dpi=self.render_config["video_dpi"])
            self._video_writer.grab_frame()

        # Clear the output if needed
        if clear:
            self._ax.cla()

    def load_map(self):
        map_data = dict()

        # Choose random map
        random_map = np.random.choice(list(self.config["maps_names_with_variants"].keys()))

        # Check if map ends with .json
        random_map_path = os.path.join(self.config["map_path"], random_map)
        if not random_map_path.endswith(".json"):
            random_map_path = random_map_path + ".json"

        # Load map data
        with open(random_map_path, 'r') as f:
            # Load map data
            map_data = json.load(f)
        
        # Choose random variant
        if self.config["maps_names_with_variants"][random_map] is not None:
            random_variant = np.random.choice(self.config["maps_names_with_variants"][random_map])
        else:
            # Choose random variant from all variants
            random_variant = np.random.choice(list(map_data["map_variant"].keys()))

        if random_variant is None:
            raise ValueError("No variants found for map {}".format(random_map))
        
        # Check number of agents:
        if self.config["agents_num"] < map_data["metadata"]["min_num_of_agents"]:
            raise ValueError("Number of agents is less than minimum number of agents for map ({})".format(map_data["metadata"]["min_num_of_agents"]))
        if self.config["agents_num"] > map_data["metadata"]["max_num_of_agents"]:
            raise ValueError("Number of agents is greater than maximum number of agents for map ({})".format(map_data["metadata"]["max_num_of_agents"]))

        self.obstacles = map_data["map_variant"][str(random_variant)]["obstacles"]
        self.agent_positions = map_data["map_variant"][str(random_variant)]["starting_positions"]
        self.goal_positions = map_data["map_variant"][str(random_variant)]["goal_positions"]

        # Convert from number to agent_number
        self.agent_positions = {"agent_" + str(agent): tuple(pos) for agent, pos in self.agent_positions.items()}
        self.goal_positions = {"agent_" + str(agent): tuple(pos) for agent, pos in self.goal_positions.items()}

        # Convert to set
        self.obstacles = {(pos[0], pos[1]) for pos in self.obstacles}

        
    def verify_map_sizes(self):
        self.grid_size = None
        # Check if all maps have the same size
        for map_name in self.config["maps_names_with_variants"]:
            map_path = os.path.join(self.config["map_path"], map_name+".json")
            with open(map_path, 'r') as f:
                # Load map data
                map_data = json.load(f)
                grid_size = (map_data["metadata"]["width"], map_data["metadata"]["height"])
                if self.grid_size is None:
                    self.grid_size = grid_size
                elif self.grid_size != grid_size:
                    raise ValueError("Map sizes do not match. Expected: {}, Found: {}, when processing {}".format(self.grid_size, grid_size, map_name))


if __name__ == "__main__":
    # Example usage
    env_config = {
        "agents_num": 10,
        "render_mode": "human",
        "max_steps": 200,
        "observation_type": "array",
        "maps_names_with_variants": {
            # "warehouse0_1-10a-21x21": None,
            "pprai_1-20a-42x39": None,
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
            "save_frames": False,
            "frames_path": "frames/",
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
        obs = env.reset()
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
