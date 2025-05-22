from m_star import MStarAgent
from rlmapf2 import RLMAPF
import numpy as np
from tqdm import tqdm

def obs_to_mstar_obs(agent, obs, info):
    # Convert RLMAPF obs/info to MStarAgent obs format
    return {
        'current_position': info[agent]['current_position'],
        'goal_position': env.goal_positions[agent],
        'obstacles': env.obstacles,
    }

if __name__ == "__main__":
    env_config = {
        "agents_num": 6,
        "render_mode": "human",
        "max_steps": 100,
        "observation_type": "array",
        "maps_names_with_variants": {
            # Provide a valid map here
            # "empty_4-4a-10x8": None,
            # "box_4-4a-10x8": None,
            "chambers_6-6a-10x8": None,
            # "circle_10-10a-17x17": None,
        },
        "render_config": {
            "title": "M* Agent",
            "show_render": False,
            "save_video": True,
            "include_legend": True,
            "legend_position": (0, 0),
            "video_path": "render.mp4",
            "video_fps": 10,
            "video_dpi": 300,
            "render_delay": 0.5,
            "save_frames": False,
            "frames_path": "frames/",
        },
        # "seed": 42,
    }

    env = RLMAPF(env_config)
    obs, info = env.reset()
    agent_ids = list(env.get_agent_ids())
    n_agents = len(agent_ids)

    # Prepare joint start/goal for all agents
    all_starts = [env.agent_positions[agent] for agent in agent_ids]
    all_goals = [env.goal_positions[agent] for agent in agent_ids]
    obstacles = env.obstacles
    grid_size = env.grid_size

    # Create MStarAgent for each agent
    agents = {
        agent: MStarAgent(
            agent_id=i,
            grid_size=grid_size,
            all_starts=all_starts,
            all_goals=all_goals,
            obstacles=obstacles
        )
        for i, agent in enumerate(agent_ids)
    }

    # Reset each agent with its observation
    for i, agent in enumerate(agent_ids):
        mstar_obs = obs_to_mstar_obs(agent, obs, info)
        agents[agent].reset(mstar_obs)

    total_rewards = {agent: 0 for agent in agent_ids}
    terminateds = {agent: False for agent in agent_ids}
    terminateds['__all__'] = False
    truncateds = {agent: False for agent in agent_ids}
    truncateds['__all__'] = False

    print("Testing MStarAgent on RLMAPF2...")
    pbar = tqdm(total=env.max_steps, desc="Episode", unit="step")
    while not terminateds['__all__'] and not truncateds['__all__']:
        actions = {}
        for agent in agent_ids:
            mstar_obs = obs_to_mstar_obs(agent, obs, info)
            action = agents[agent].act(mstar_obs)
            actions[agent] = action
        # print("Actions:", actions)
        obs, reward, terminateds, truncateds, info = env.step(actions)
        for agent in agent_ids:
            total_rewards[agent] += reward[agent]
        pbar.update(1)
        pbar.set_postfix({"Mean rewards": np.mean(list(total_rewards.values()))})
    pbar.close()
    print("Test finished! Rewards:", total_rewards)
