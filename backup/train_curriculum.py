import ray
import torch
from ray.rllib.algorithms.ppo import PPOConfig
from rlmapf2 import RLMAPF
import numpy as np
import logging
import wandb
import time
from random import randint
import argparse

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TRAINING_EPISODES = 1200
NUM_CPUS = 4
NUM_GPUS = 1
SAVE_PATH = "/home/kamil/Documents/RLMAPF2/saved_models"
SAVE_INTERVAL = 500
EVAL_INTERVAL = 50
AGENTS_NUM = 20
OBSERVATION_TYPE = 'array'
USE_WANDB = True
TAGS = [f"agents_{AGENTS_NUM}", "basic_PPO", "reduced_observation", "no_d_star_in_obs"]
PARAMS_TO_LOG = [
    "env_runners/episode_reward_mean",
    "env_runners/episode_reward_min",
    "env_runners/episode_reward_max",
    "env_runners/episode_len_mean",
]

# Curriculum schedule: (step, maps_to_include)
curriculum = [
    (0, ["pprai_1-20a-42x39"]),
    (200, ["pprai_1-20a-42x39", "box_4-4a-10x8"]),
    (400, ["pprai_1-20a-42x39", "box_4-4a-10x8", "chambers_6-6a-10x8"]),
    (600, ["pprai_1-20a-42x39", "box_4-4a-10x8", "chambers_6-6a-10x8", "circle_10-10a-17x17"]),
]

def configure_env(agents_num=20, maps_names_with_variants=None, max_steps=250,
                  reward_closer_to_goal_final=True, reward_final_d_star=False,
                  use_d_star_lite=False):
    return {
        "agents_num": agents_num,
        "render_mode": "none",
        "render_delay": 0.01,
        "map_path": "/home/kamil/Documents/RLMAPF2/maps",
        "maps_names_with_variants": maps_names_with_variants or {"pprai_1-20a-42x39": None},
        "max_steps": max_steps,
        "collision_penalty": 0.1,
        "step_cost": 0.02,
        "wait_cost_multiplier": 2,
        "goal_reward": 10,
        "observation_type": OBSERVATION_TYPE,
        "penalize_collision": True,
        "penalize_waiting": True,
        "penalize_steps": True,
        "reward_closer_to_goal_each_step": False,
        "reward_closer_to_goal_final": reward_closer_to_goal_final,
        "reward_final_d_star": reward_final_d_star,
        "reward_low_density": False,
        "use_d_star_lite": use_d_star_lite,
    }

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train PPO with curriculum learning.")
    parser.add_argument("--agents_num", type=int, default=20)
    parser.add_argument("--max_steps", type=int, default=250)
    parser.add_argument("--reward_closer_to_goal_final", type=bool, default=True)
    parser.add_argument("--reward_final_d_star", type=bool, default=False)
    parser.add_argument("--use_d_star_lite", type=bool, default=False)
    return parser.parse_args()

args = parse_arguments()
WANDB_GROUP = "D-star" if args.use_d_star_lite else "No-D-star"
WANDB_GROUP += f"_{args.agents_num}"

context = ray.init(num_cpus=NUM_CPUS, num_gpus=NUM_GPUS, namespace=WANDB_GROUP if USE_WANDB else None)
run_name = f"run_{randint(0, 1000)}_{time.strftime('%d-%m-%Y_%H-%M-%S')}"
print(f"Run name: {run_name}")

# Initial curriculum stage
curriculum_idx = 0
maps_to_use = curriculum[curriculum_idx][1]

config = (
    PPOConfig()
    .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
    .framework("torch")
    .training(model={"fcnet_hiddens": [1024, 512, 512], "fcnet_activation": "relu"})
    .environment(RLMAPF, env_config=configure_env(
        agents_num=args.agents_num,
        maps_names_with_variants={name: None for name in maps_to_use},
        max_steps=args.max_steps,
        reward_closer_to_goal_final=args.reward_closer_to_goal_final,
        reward_final_d_star=args.reward_final_d_star,
        use_d_star_lite=args.use_d_star_lite
    ))
    .evaluation(
        evaluation_interval=EVAL_INTERVAL,
        evaluation_duration=10,
        evaluation_config={"env_config": configure_env(
            agents_num=args.agents_num,
            maps_names_with_variants={name: None for name in maps_to_use},
            max_steps=args.max_steps,
            reward_closer_to_goal_final=args.reward_closer_to_goal_final,
            reward_final_d_star=args.reward_final_d_star,
            use_d_star_lite=args.use_d_star_lite
        )}
    )
    .resources(num_gpus=NUM_GPUS)
)

run = wandb.init(
    project="rlmapf_limited_icra",
    group=WANDB_GROUP,
    tags=TAGS,
    notes="Curriculum learning run",
    config=config.to_dict(),
    name=run_name,
) if USE_WANDB else None

algorithm = config.build()

print("Starting curriculum training")
for episode in range(TRAINING_EPISODES):
    # Check if we need to update the curriculum
    if curriculum_idx + 1 < len(curriculum) and episode >= curriculum[curriculum_idx + 1][0]:
        curriculum_idx += 1
        maps_to_use = curriculum[curriculum_idx][1]
        print(f"Curriculum update at episode {episode}: maps = {maps_to_use}")
        # Update environment config
        algorithm.env_config["maps_names_with_variants"] = {name: None for name in maps_to_use}
        algorithm._env = None  # Force env re-creation
        algorithm._env_config = algorithm.env_config
        algorithm._initialize_env()

    results = algorithm.train()
    print(f"Episode: {episode}")
    print(f"\tMean reward: {results['env_runners']['episode_reward_mean']/AGENTS_NUM}")
    print(f"\tMin reward: {results['env_runners']['episode_reward_min']/AGENTS_NUM}")
    print(f"\tMax reward: {results['env_runners']['episode_reward_max']/AGENTS_NUM}")
    print(f"\tMean episode length: {results['env_runners']['episode_len_mean']}")
    if 'evaluation' in results.keys():
        print("Evaluation:")
        print(f"\tEvaluation mean reward: {results['evaluation']['env_runners']['episode_reward_mean']/AGENTS_NUM}")
        print(f"\tEvaluation min reward: {results['evaluation']['env_runners']['episode_reward_min']/AGENTS_NUM}")
        print(f"\tEvaluation max reward: {results['evaluation']['env_runners']['episode_reward_max']/AGENTS_NUM}")
        print(f"\tEvaluation mean episode length: {results['evaluation']['env_runners']['episode_len_mean']}")
    if episode % SAVE_INTERVAL == 0 and episode > 0:
        algorithm.save(f"{SAVE_PATH}/model_{run_name}_{episode}")
        print(f"Model saved at: {SAVE_PATH}/model_{run_name}_{episode}")
        if USE_WANDB:
            wandb.log({"model_path": f"{SAVE_PATH}/model_{run_name}_{episode}"})
    if USE_WANDB:
        wandb_results = {}
        for param in [p.split("/") for p in PARAMS_TO_LOG]:
            idx = 0
            new_results = results
            while idx < len(param):
                new_results = new_results[param[idx]]
                idx += 1
            param_name = "/".join(param)
            if "reward" in param_name:
                new_results = new_results/AGENTS_NUM
            wandb_results[param_name] = new_results
        if 'evaluation' in results.keys():
            for param in [p.split("/") for p in PARAMS_TO_LOG]:
                idx = 0
                new_results = results['evaluation']
                while idx < len(param):
                    new_results = new_results[param[idx]]
                    idx += 1
                param_name = "/".join(['evaluation']+param)
                if "reward" in param_name:
                    new_results = new_results/AGENTS_NUM
                wandb_results[param_name] = new_results
        wandb.log(wandb_results)

print("="*15, "Curriculum Training finished", "="*15)
algorithm.save(f"{SAVE_PATH}/model_{run_name}_final")
print(f"Final model saved at: {SAVE_PATH}/model_{run_name}_final")
if USE_WANDB:
    wandb.log({"model_path": f"{SAVE_PATH}/model_{run_name}_final"})
    wandb.finish()
ray.shutdown()
