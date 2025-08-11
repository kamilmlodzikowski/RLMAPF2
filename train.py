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
from random import randint
import argparse

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Config
TRAINING_EPISODES = 1200  # Number of episodes to train for
# TRAINING_EPISODES = 5000
COCURRENT_TRAININGS = 1  # Number of trainings that will be run in parallel
NUM_CPUS = 4  # Number of CPU cores available
NUM_GPUS = 1  # Number of GPUs available
SAVE_PATH = "/home/kamil/Documents/RLMAPF2/saved_models"
SAVE_INTERVAL = 500  # Save the model every SAVE_INTERVAL episodes
EVAL_INTERVAL = 50  # Evaluate the model every EVAL_INTERVAL episodes

# Environment setup
AGENTS_NUM = 20
# AGENTS_NUM = 10
OBSERVATION_TYPE = 'array' 

# WandB setup
USE_WANDB = True  # If True, will use Weights and Biases for logging
# WANDB_GROUP = "No-D-star-20"  # Group name for Weights and Biases

TAGS = ["agents_"+str(AGENTS_NUM), 
        "basic_PPO", 
        "reduced_observation",
        # "d_star_in_obs",
        "no_d_star_in_obs",
]

PARAMS_TO_LOG = [
    "env_runners/episode_reward_mean",
    "env_runners/episode_reward_min",
    "env_runners/episode_reward_max",
    "env_runners/episode_len_mean",
    # "env_runners/episode_len_min",
    # "env_runners/episode_len_max",
]

best_results = {
    "env_runners/episode_reward_mean": {"value": -np.inf, "episode": -1},
    "env_runners/episode_reward_min": {"value": -np.inf, "episode": -1},
    "env_runners/episode_reward_max": {"value": -np.inf, "episode": -1},
    "env_runners/episode_len_mean": {"value": np.inf, "episode": -1},
    "evaluation/env_runners/episode_reward_mean": {"value": -np.inf, "episode": -1},
    "evaluation/env_runners/episode_reward_min": {"value": -np.inf, "episode": -1},
    "evaluation/env_runners/episode_reward_max": {"value": -np.inf, "episode": -1},
    "evaluation/env_runners/episode_len_mean": {"value": np.inf, "episode": -1},
}

try:

    # Split params to log by '/'
    PARAMS_TO_LOG = [param.split("/") for param in PARAMS_TO_LOG]

    # Create run name from current time DD-MM-YYYY_HH-MM-SS
    run_name = "run_"+str(randint(0, 1000))+'_' + str(time.strftime("%d-%m-%Y_%H-%M-%S"))
    print("Run name: {0}".format(run_name))

    

    def configure_env(agents_num=20, maps_names_with_variants=None, max_steps=250, 
                      reward_closer_to_goal_final=True, reward_final_d_star=False, 
                      use_d_star_lite=False):
        """Configure the default environment with passed parameters."""
        return {
            "agents_num": agents_num,
            "render_mode": "none",
            "render_delay": 0.01,
            "map_path": "/home/kamil/Documents/RLMAPF2/maps",
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
        }

    def parse_arguments():
        """Parse command-line arguments for environment configuration."""
        parser = argparse.ArgumentParser(description="Train PPO with configurable environment.")
        parser.add_argument("--agents_num", type=int, default=20, help="Number of agents.")
        parser.add_argument("--maps_names_with_variants", type=str, nargs="*", default=["pprai_1-20a-42x39"], help="Map names with variants.")
        parser.add_argument("--max_steps", type=int, default=250, help="Maximum steps per episode.")
        parser.add_argument("--reward_closer_to_goal_final", type=bool, default=True, help="Reward closer to goal at final step.")
        parser.add_argument("--reward_final_d_star", type=bool, default=False, help="Reward based on D* Lite.")
        parser.add_argument("--use_d_star_lite", type=bool, default=False, help="Use D* Lite algorithm.")
        return parser.parse_args()

    # Parse arguments
    args = parse_arguments()

    # Dynamically set WANDB_GROUP based on use_d_star_lite
    WANDB_GROUP = "D-star" if args.use_d_star_lite else "No-D-star"
    WANDB_GROUP += "_" + str(args.agents_num)

    # Ray initialization
    context = ray.init(
        num_cpus=NUM_CPUS,
        num_gpus=NUM_GPUS,
        # _temp_dir="/media/storage2/ray_tmp",
        namespace=WANDB_GROUP if USE_WANDB else None,

    )
    print("-" * 20)
    print("Ray initialized")
    # print("Dashboard URL: http://" + context.dashboard_url)
    print("-" * 20)

    config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False
        )
        .framework("torch")
        .training(
            model={
                # "fcnet_hiddens": [2048, 1024, 256],  # Larger network
                "fcnet_hiddens": [1024, 512, 512],  # Smaller network
                "fcnet_activation": "relu",  # ReLU often performs better than tanh
                # "vf_share_layers": False,  # Separate value function
                # "use_lstm": True,  
            },
        )
        .environment(RLMAPF, env_config=configure_env(
            agents_num=args.agents_num,
            maps_names_with_variants={name: None for name in args.maps_names_with_variants},
            max_steps=args.max_steps,
            reward_closer_to_goal_final=args.reward_closer_to_goal_final,
            reward_final_d_star=args.reward_final_d_star,
            use_d_star_lite=args.use_d_star_lite
        ))
        .evaluation(
            evaluation_interval=EVAL_INTERVAL,
            evaluation_duration=10,
            # evaluation_force_reset_envs_before_iteration=True,
            evaluation_config={"env_config": {
                "agents_num": AGENTS_NUM,
                "render_mode": "none",
                "render_delay": 0.01,
                "seed": 42,
                "map_path": "/home/kamil/Documents/RLMAPF/maps",
                "maps_names_with_variants": {
                    "pprai_1-20a-42x39": None,
                },
                "max_steps": 200,
                "collision_penalty": 0.1,
                "step_cost": 0.01,
                "wait_cost_multiplier": 2,
                "goal_reward": 10,
                "observation_type": OBSERVATION_TYPE,
                "predict_distance": False,
                "penalize_collision": True,
                "penalize_waiting": True,
                "penalize_steps": True,
                "reward_closer_to_goal_each_step": False,
                "reward_closer_to_goal_final": True,
                "reward_final_d_star": False,
                "reward_low_density": False,
            }}
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
    # Config wandb with parameters
        # WandB setup
    run = wandb.init(
        # project="RLMAPF_new_test",
        project = "rlmapf_limited_icra",
        group=WANDB_GROUP,
        tags=TAGS,
        notes="",
        config=config.to_dict(),
        name=run_name,
    ) if USE_WANDB else None

    if ["env_runners", "episode_reward_mean"] in PARAMS_TO_LOG:
        wandb.define_metric("env_runners/episode_reward_mean", summary="max")
    else:
        # Print warrning
        print("!"*30)
        print("env_runners/episode_reward_mean not in PARAMS_TO_LOG")
        print("!"*30)

    if ["env_runners", "episode_reward_min"] in PARAMS_TO_LOG:
        wandb.define_metric("env_runners/episode_len_mean", summary="min")
    else:
        # Print warrning
        print("!"*30)
        print("env_runners/episode_len_mean not in PARAMS_TO_LOG")
        print("!"*30)

    # Create the algorithm
    algorithm = config.build()


    # Train loop
    print("Starting training")
    for episode in range(TRAINING_EPISODES):
        results = algorithm.train()

        # Print results
        print('-' * 30)
        print("Episode: {0}".format(episode))
        print("\tMean reward: {0}".format(results["env_runners"]["episode_reward_mean"]/AGENTS_NUM))
        print("\tMin reward: {0}".format(results["env_runners"]["episode_reward_min"]/AGENTS_NUM))
        print("\tMax reward: {0}".format(results["env_runners"]["episode_reward_max"]/AGENTS_NUM))
        print("\tMean episode length: {0}".format(results["env_runners"]["episode_len_mean"]))
        # print("\tMin episode length: {0}".format(results["env_runners"]["episode_len_min"]))
        # print("\tMax episode length: {0}".format(results["env_runners"]["episode_len_max"]))

        if 'evaluation' in results.keys():
            print("Evaluation:")
            print("\tEvaluation mean reward: {0}".format(results["evaluation"]["env_runners"]["episode_reward_mean"]/AGENTS_NUM))
            print("\tEvaluation min reward: {0}".format(results["evaluation"]["env_runners"]["episode_reward_min"]/AGENTS_NUM))
            print("\tEvaluation max reward: {0}".format(results["evaluation"]["env_runners"]["episode_reward_max"]/AGENTS_NUM))
            print("\tEvaluation mean episode length: {0}".format(results["evaluation"]["env_runners"]["episode_len_mean"]))
            # print("\tEvaluation min episode length: {0}".format(results["evaluation"]["env_runners"]["episode_len_min"]))
            # print("\tEvaluation max episode length: {0}".format(results["evaluation"]["env_runners"]["episode_len_max"]))

        if episode % SAVE_INTERVAL == 0 and episode > 0:
            algorithm.save(SAVE_PATH + "/model_" + run_name + "_" + str(episode))
            print("Model saved at: {0}".format(SAVE_PATH + "/model_" + str(episode)))
            if USE_WANDB:
                # Log model path to wandb
                wandb.log({"model_path": SAVE_PATH + "/model_" + run_name + "_" + str(episode)})

        # save results to wandb
        if USE_WANDB:
            wandb_results = {}
            for param in PARAMS_TO_LOG:
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
                for param in PARAMS_TO_LOG:
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
        
    # Training finished
    print("="*15, "Training finished", "="*15)
    
    # Save the final model
    algorithm.save(SAVE_PATH + "/model_" + run_name + "_final")
    print("Final model saved at: {0}".format(SAVE_PATH + "/model_" + run_name + "_final"))
    if USE_WANDB:
        # Log model path to wandb
        wandb.log({"model_path": SAVE_PATH + "/model_" + run_name + "_final"})
    

    # Print best results
    print("Best results:")
    for key, value in best_results.items():
        print("\t{0}: {1} at episode {2}".format(key, value["value"], value["episode"]))

    # Log best results to wandb
    if USE_WANDB:
        wandb.log(best_results)
    
    # Finish wandb run
    wandb.finish()

except Exception as e:
    logger.error(f"An error occurred: {e}", exc_info=True)
finally:
    ray.shutdown()
