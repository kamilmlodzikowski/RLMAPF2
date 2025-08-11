import ray
import torch
from ray import tune, train
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print
from rlmapf2 import RLMAPF
from ray.tune.stopper import TrialPlateauStopper, MaximumIterationStopper, CombinedStopper
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.tune.schedulers import PopulationBasedTraining
import numpy as np
import logging
import sys
import wandb
import time
from ray.rllib.algorithms.algorithm import Algorithm
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

# Ignore deprecation warnings
os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning"

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Config
NUM_CPUS = 2  # Number of CPU cores available
NUM_GPUS = 0  # Number of GPUs available
CHECKPOINT_PATH = "/home/kamil/Documents/RLMAPF/saved_models/model_run_687_30-06-2025_11-53-05_final"
USE_D_STAR_LITE = False  # Use D* Lite algorithm

# Environment setup
AGENTS_NUM = 20
OBSERVATION_TYPE = 'array'
EVAL_EPISODES = 1
REPEATS = 1

EVAL_SEEDS = [a for a in range(42, 42 + EVAL_EPISODES)]

def configure_env(agents_num=20, maps_names_with_variants=None, max_steps=250, 
                      reward_closer_to_goal_final=True, reward_final_d_star=False, 
                      use_d_star_lite=False, render_mode="none", render_delay=0.01, seed=42):
        """Configure the default environment with passed parameters."""
        return {
            "agents_num": agents_num,
            "render_mode": render_mode,
            "render_delay": render_delay,
            "seed": seed,
            "map_path": "/home/kamil/Documents/RLMAPF/maps",
            "maps_names_with_variants": maps_names_with_variants or {
                "pprai_1-20a-42x39": None,
            },
            "max_steps": max_steps,
            "collision_penalty": 0.1,
            "step_cost": 0.02,
            "wait_cost_multiplier": 2,
            "goal_reward": 10,
            "observation_type": OBSERVATION_TYPE,
            "predict_distance": False,
            "penalize_collision": True,
            "penalize_waiting": True,
            "penalize_steps": True,
            "reward_closer_to_goal_each_step": False,
            "reward_closer_to_goal_final": reward_closer_to_goal_final,
            "reward_final_d_star": reward_final_d_star,
            "reward_low_density": False,
            "use_d_star_lite": use_d_star_lite,
            "render_config": {
                "show_render": False,
                "save_video": True,
                "include_legend": True,
                "legend_position": (0, 0),
                "video_path": "render.mp4",
                "video_fps": 10,
                "video_dpi": 300,
                "render_delay": 0.2,
                "title": "RLMAPF Evaluation" + (" (D*)" if use_d_star_lite else " (no D*)"),
                "save_frames": False,
                "frames_path": "frames/",
            },
        }

try:

    # Create the algorithm
    # algorithm = Algorithm.from_checkpoint(CHECKPOINT_PATH)

    new_algorithm_config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False
        )
        .framework("torch")
        .training(
            model={
                "fcnet_hiddens": [1024, 512, 512],
                "fcnet_activation": "relu", 
            },
        )
        .environment(RLMAPF, env_config=configure_env(
            agents_num=AGENTS_NUM,
            maps_names_with_variants={
                "pprai_1-20a-42x39": None,
            },
            max_steps=200,
            reward_closer_to_goal_final=True,
            reward_final_d_star=False,
            use_d_star_lite=USE_D_STAR_LITE,
        ))
        .evaluation(
            evaluation_interval=1,
            evaluation_duration=1,
            # evaluation_force_reset_envs_before_iteration=True,
            evaluation_config={"env_config": 
                configure_env(
                    agents_num=AGENTS_NUM,
                    maps_names_with_variants={
                        "pprai_1-20a-42x39": None,
                    },
                    max_steps=200,
                    reward_closer_to_goal_final=True,
                    reward_final_d_star=False,
                    use_d_star_lite=USE_D_STAR_LITE,
                    render_mode="none",
                )
            }
        )
        # .training(
        #     # clip_param=0.1,
        #     grad_clip=0.2,
        # )
        .resources(
            num_gpus=NUM_GPUS,
            # num_cpus=NUM_CPUS,

        )
        # .rollouts(sample_timeout_s=1000)

    )

    new_algorithm = new_algorithm_config.build()

    # Load weights from the checkpoint
    # new_algorithm.set_weights(algorithm.get_weights())
    new_algorithm.restore(CHECKPOINT_PATH)

    rewards = {}
    lengths = {}
    # env = algorithm.env_creator(algorithm.config['env_config'])
    # env = new_algorithm.env_creator(n.config['evaluation_config']['env_config'])
    # env._seed = 42
    # obs, _ = env.reset(seed=42)


    # for repeat in range(REPEATS):
    #     print("Repeat {0} out of {1}".format(repeat+1, REPEATS))
    #     rewards[repeat] = {}
    #     lengths[repeat] = {}
    #     print('='*30)
    #     print("Repeat {0} out of {1}".format(repeat+1, REPEATS))
    #     print('='*30)
    #     for i, seed in enumerate(EVAL_SEEDS):
    #         rewards[repeat][seed] = []
    #         lengths[repeat][seed] = []
    #         print('-' * 30)
    #         print("Running evaluation {0} out of {1} with seed: {2}".format(i+1, len(EVAL_SEEDS), seed))
    #         # algorithm = Algorithm.from_checkpoint(CHECKPOINT_PATH)
    #         # # Change evaluation to render mode
    #         # # print(algorithm.config['render_env'])
    #         # algorithm.config['evaluation_config']["env_config"]['agents_num'] = AGENTS_NUM
    #         # algorithm.config['evaluation_config']["env_config"]["render_mode"] = "human"
    #         # algorithm.config['evaluation_config']['render_env'] = True
    #         # algorithm.config['evaluation_config']["env_config"]["seed"] = 42
    #         # print(algorithm.config['evaluation_config'])

    #         for episode in range(EVAL_EPISODES):
    #             print("Episode {0} out of {1}".format(episode+1, EVAL_EPISODES))
    #             results = new_algorithm.evaluate()
    #             rewards[repeat][seed].append(results["env_runners"]["episode_reward_mean"])
    #             lengths[repeat][seed].append(results["env_runners"]["episode_len_mean"])

    #         # Print mid evaluation results
    #         print("Mean reward per agent: {0}".format(np.mean(rewards[repeat][seed])/AGENTS_NUM))
    #         print("Mean episode length: {0}".format(np.mean(lengths[repeat][seed])))
    #         # print("Time elapsed: {0}".format(time.time() - start_time))
    #         # print("Expected time remaining: {0}".format((time.time() - start_time) * (len(EVAL_SEEDS) - i - 1) / (i + 1)))

    # print('='*30)
    # print("Evaluation results:")

    # print("Mean reward per agent: {0}".format(np.mean([np.mean(rewards[repeat][seed]) for seed in EVAL_SEEDS for repeat in range(REPEATS)])/AGENTS_NUM))
    # print("Min reward per agent: {0}".format(np.min([np.min(rewards[repeat][seed]) for seed in EVAL_SEEDS for repeat in range(REPEATS)])/AGENTS_NUM))
    # print("Max reward per agent: {0}".format(np.max([np.max(rewards[repeat][seed]) for seed in EVAL_SEEDS for repeat in range(REPEATS)])/AGENTS_NUM))
    # print("Reward std per agent: {0}".format(np.std([np.std(rewards[repeat][seed]) for seed in EVAL_SEEDS for repeat in range(REPEATS)])/AGENTS_NUM))
    # print("Mean episode length: {0}".format(np.mean([np.mean(lengths[repeat][seed]) for seed in EVAL_SEEDS for repeat in range(REPEATS)])))
    # print("Episode length std: {0}".format(np.std([np.std(lengths[repeat][seed]) for seed in EVAL_SEEDS for repeat in range(REPEATS)])))



    NUM_THREADS = 1
    AGENTS_RANGE = range(4, 21)  # Range of agents_num
    REPEATS = 10  # Number of repeats for each agents_num

    time_results = {}
    lengths_results = {}
    deadlocks = {}

    def run_repeat(agents_num, repeat):
        print(f"Running evaluation with {agents_num} agents, repeat {repeat + 1}...")
        new_algorithm_config.environment(RLMAPF, env_config=configure_env(
            agents_num=agents_num,
            maps_names_with_variants={
                "pprai_1-20a-42x39": None,
            },
            max_steps=200,
            reward_closer_to_goal_final=True,
            reward_final_d_star=False,
            use_d_star_lite=USE_D_STAR_LITE,
            render_mode="none",
            seed=42 + repeat  # Use a different seed for each repeat
        ))
        new_algorithm = new_algorithm_config.build()
        new_algorithm.restore(CHECKPOINT_PATH)

        start_time = time.time()
        try:
            results = new_algorithm.evaluate()
            elapsed_time = time.time() - start_time
            episode_len = results['env_runners']['episode_len_mean']
            print(f"Evaluation completed in {elapsed_time:.5f} seconds.")
            print(f"Steps taken: {episode_len}")
            deadlock = 0 if episode_len < 200 else 1  # Assuming deadlock if episode_len >= 200
            return (agents_num, repeat, elapsed_time, episode_len, deadlock)
        except Exception as e:
            print(f"Error during evaluation: {e}")
            return (agents_num, repeat, None, None, 1)  # deadlock

    # Create a folder to save results with readable timestamp
    results_folder_name = 'eval_test_results_' + time.strftime("%Y%m%d_%H%M%S")
    if not os.path.exists(results_folder_name):
        os.makedirs(results_folder_name)
    print(f"Results will be saved in folder: {results_folder_name}")

    for agents_num in AGENTS_RANGE:
        deadlocks[agents_num] = 0
        with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
            futures = [executor.submit(run_repeat, agents_num, repeat) for repeat in range(REPEATS)]
            for future in as_completed(futures):
                agents_num_r, repeat_r, elapsed_time, episode_len, deadlock = future.result()
                if elapsed_time is not None and deadlock == 0:
                    time_results[(agents_num_r, repeat_r)] = elapsed_time
                if episode_len is not None and deadlock == 0:
                    lengths_results[(agents_num_r, repeat_r)] = episode_len
                deadlocks[agents_num_r] += deadlock

        # Print and save intermediate results
        avg_time = {key[0]: 0 for key in time_results.keys() if key[0] == agents_num}
        for key, total_time in time_results.items():
            if key[0] == agents_num:
                avg_time[key[0]] += total_time / len([k for k in time_results if k[0] == key[0]])
        avg_length = {key[0]: 0 for key in lengths_results.keys() if key[0] == agents_num}
        for key, total_length in lengths_results.items():
            if key[0] == agents_num:
                avg_length[key[0]] += total_length / len([k for k in lengths_results if k[0] == key[0]])
        print(f"Intermediate results for {agents_num} agents:")
        print(f"Average time: {avg_time[agents_num]:.5f} seconds, Length: {avg_length[agents_num]:.2f}, Deadlocks: {deadlocks[agents_num]}")

        # Save intermediate results to file
        intermediate_name = results_folder_name + f'/intermediate_results_{agents_num}_agents'
        with open(intermediate_name + '.txt', 'w') as f:
            f.write(f"Intermediate results for {agents_num} agents:\n")
            f.write(f"Average time: {avg_time[agents_num]:.5f} seconds, Length: {avg_length[agents_num]:.2f}, Deadlocks: {deadlocks[agents_num]}\n")
            f.write("\nDetailed time results:\n")
            for (agents_num_r, repeat), elapsed_time in time_results.items():
                if agents_num_r == agents_num:
                    f.write(f"Agents: {agents_num_r}, Repeat: {repeat}, Time: {elapsed_time:.5f} seconds\n")

        # Save intermediate results to CSV
        df = pd.DataFrame.from_dict(
            {
            key: {
                'elapsed_time': elapsed_time,
                'episode_length': lengths_results[key]
            } for key, elapsed_time in time_results.items() if key[0] == agents_num
            },
            orient='index'
        )
        df.index.names = ['agents_num', 'repeat']
        df.reset_index(inplace=True)
        df.to_csv(intermediate_name + '.csv', index=False)

    avg_time = {key[0]: 0 for key in time_results.keys()}
    avg_length = {key[0]: 0 for key in lengths_results.keys()}
    print('==' * 30)
    for key, total_time in time_results.items():
        avg_time[key[0]] += total_time / len([k for k in time_results if k[0] == key[0]])
    for key, total_length in lengths_results.items():
        avg_length[key[0]] += total_length / len([k for k in lengths_results if k[0] == key[0]])
    print("Average time results:")
    for agents_num, total_time in avg_time.items():
        print(f"Agents: {agents_num}, Average time: {total_time:.5f} seconds, Length: {avg_length[agents_num]:.2f}, Deadlocks: {deadlocks[agents_num]}")

    # Save final results to file
    final_name = results_folder_name + '/final_results'
    with open(final_name + '.txt', 'w') as f:
        f.write("Average time results:\n")
        for agents_num, total_time in avg_time.items():
            f.write(f"Agents: {agents_num}, Average time: {total_time:.5f} seconds, Deadlocks: {deadlocks[agents_num]}\n")
        f.write("\nDetailed time results:\n")
        for (agents_num, repeat), elapsed_time in time_results.items():
            f.write(f"Agents: {agents_num}, Repeat: {repeat}, Time: {elapsed_time:.5f} seconds\n")

    # Save final results to CSV
    df = pd.DataFrame.from_dict(
        {
            key: {
                'elapsed_time': elapsed_time,
                'episode_length': lengths_results[key]
            } for key, elapsed_time in time_results.items()
        },
        orient='index'
    )
    df.index.names = ['agents_num', 'repeat']
    df.reset_index(inplace=True)
    df.to_csv(final_name + '.csv', index=False)
except Exception as e:
    logger.error(f"An error occurred: {e}", exc_info=True)
finally:
    ray.shutdown()