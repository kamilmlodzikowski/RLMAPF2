import time
from m_star import MStarAgent
from rlmapf2 import RLMAPF
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

NUM_THREADS = 5

def obs_to_mstar_obs(agent, obs, info, env):
    # Convert RLMAPF obs/info to MStarAgent obs format
    return {
        'current_position': info[agent]['current_position'],
        'goal_position': env.goal_positions[agent],
        'obstacles': env.obstacles,
    }

if __name__ == "__main__":
    time_results = {}
    deadlocks = {}

    def run_repeat(agents_num, repeat):
        env_config = {
            "agents_num": agents_num,
            "render_mode": "none",
            "max_steps": 500,
            "seed": 42+repeat,
            "observation_type": "array",
            "maps_names_with_variants": {
                "pprai_1-20a-42x39": None,
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
        }
        print(f"Running test with {agents_num} agents, seed {42+repeat}...")

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

        # Reset each agent with its observation (multi-threaded)
        start_time = time.time()
        try:
            print("Starting test with ", n_agents, "agents.")
            with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
                futures = {
                    executor.submit(agents[agent].reset, obs_to_mstar_obs(agent, obs, info, env)): agent
                    for agent in agent_ids
                }
                for future in as_completed(futures):
                    agent = futures[future]
                    try:
                        future.result()
                    except Exception as e:
                        raise RuntimeError(f"Agent {agent} reset failed: {e}")
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"M* paths calculated in {elapsed_time:.5f} seconds.")
            print(f'Length of M* paths: {[len(agents[agent].path) for agent in agent_ids]}')
        except Exception as e:
            print(f"Error during M* path calculation: {e}")
            env.close()
            return (agents_num, repeat, None, 1)  # deadlock

        total_rewards = {agent: 0 for agent in agent_ids}
        terminateds = {agent: False for agent in agent_ids}
        terminateds['__all__'] = False
        truncateds = {agent: False for agent in agent_ids}
        truncateds['__all__'] = False

        while not terminateds['__all__'] and not truncateds['__all__']:
            actions = {}
            with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
                action_futures = {
                    executor.submit(
                        agents[agent].act, obs_to_mstar_obs(agent, obs, info, env), env.agent_positions[agent]
                    ): agent
                    for agent in agent_ids
                    if (agent in obs and agent in info) and not terminateds[agent] and not truncateds[agent]
                }
                for future in as_completed(action_futures):
                    agent = action_futures[future]
                    try:
                        actions[agent] = future.result()
                    except Exception as e:
                        print(f"Agent {agent} act failed: {e}")
                        actions[agent] = 4  # fallback to wait

            obs, reward, terminateds, truncateds, info = env.step(actions)
            for agent in reward:
                if agent in total_rewards:
                    total_rewards[agent] += reward[agent]

        print("Episode ", repeat + 1, " finished with ", n_agents, " agents.")
        print(f"Test finished! Rewards: {total_rewards}")
        print('-' * 30)
        env.close()
        return (agents_num, repeat, elapsed_time, 0)  # no deadlock

    # Create a folder to save results with readable timestamp
    import time
    results_folder_name = 'mstar_test_results_' + time.strftime("%Y%m%d_%H%M%S")
    import os
    if not os.path.exists(results_folder_name):
        os.makedirs(results_folder_name)
    print(f"Results will be saved in folder: {results_folder_name}")

    for agents_num in range(4, 21):
        deadlocks[agents_num] = 0
        results = []
        with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
            futures = [executor.submit(run_repeat, agents_num, repeat) for repeat in range(10)]
            for future in as_completed(futures):
                agents_num_r, repeat_r, elapsed_time, deadlock = future.result()
                if elapsed_time is not None:
                    time_results[(agents_num_r, repeat_r)] = elapsed_time
                deadlocks[agents_num_r] += deadlock

        # Print and save intermediate results
        avg_time = {key[0]: 0 for key in time_results.keys() if key[0] == agents_num}
        for key, total_time in time_results.items():
            if key[0] == agents_num:
                avg_time[key[0]] += total_time / len([k for k in time_results if k[0] == key[0]])
        print(f"Intermediate results for {agents_num} agents:")
        print(f"Average time: {avg_time[agents_num]:.5f} seconds, Deadlocks: {deadlocks[agents_num]}")

        # Save intermediate results to file by appending
        intermediate_name = results_folder_name + '/intermediate_results_' + str(agents_num) + '_agents'
        print(f"Saving intermediate results to {intermediate_name}")
        import os
        if not os.path.exists(results_folder_name):
            os.makedirs(results_folder_name)
        with open(intermediate_name + '.txt', 'w') as f:
            f.write(f"Intermediate results for {agents_num} agents:\n")
            f.write(f"Average time: {avg_time[agents_num]:.5f} seconds, Deadlocks: {deadlocks[agents_num]}\n")
            f.write("\nDetailed time results:\n")
            for (agents_num_r, repeat), elapsed_time in time_results.items():
                if agents_num_r == agents_num:
                    f.write(f"Agents: {agents_num_r}, Repeat: {repeat}, Time: {elapsed_time:.5f} seconds\n")

        # Save intermediate results to csv by appending
        df = pd.DataFrame.from_dict(
            {key: elapsed_time for key, elapsed_time in time_results.items() if key[0] == agents_num},
            orient='index',
            columns=['elapsed_time']
        )
        df.index.names = ['agents_num']
        df.reset_index(inplace=True)
        with open(intermediate_name + '.csv', 'w') as f:
            df.to_csv(f, index=False, header=f.tell() == 0)  # Write header only if file is empty

    avg_time = {key[0]: 0 for key in time_results.keys()}
    print('==' * 30)
    for key, total_time in time_results.items():
        avg_time[key[0]] += total_time / len([k for k in time_results if k[0] == key[0]])
    print("Average time results:")
    for agents_num, total_time in avg_time.items():
        print(f"Agents: {agents_num}, Average time: {total_time:.5f} seconds, Deadlocks: {deadlocks[agents_num]}")
    


    # Save results to file
    timed_name = results_folder_name + '/final_results'
    print(f"Saving results to {timed_name}")
    with open(timed_name + '.txt', 'w') as f:
        f.write("Average time results:\n")
        for agents_num, total_time in avg_time.items():
            f.write(f"Agents: {agents_num}, Average time: {total_time:.5f} seconds, Deadlocks: {deadlocks[agents_num]}\n")
        f.write("\nDetailed time results:\n")
        for (agents_num, repeat), elapsed_time in time_results.items():
            f.write(f"Agents: {agents_num}, Repeat: {repeat}, Time: {elapsed_time:.5f} seconds\n")

    # Save to csv
    import pandas as pd
    df = pd.DataFrame.from_dict(time_results, orient='index', columns=['elapsed_time'])
    df.index.names = ['agents_num']
    df.reset_index(inplace=True)
    df.to_csv(timed_name + '.csv', index=False)