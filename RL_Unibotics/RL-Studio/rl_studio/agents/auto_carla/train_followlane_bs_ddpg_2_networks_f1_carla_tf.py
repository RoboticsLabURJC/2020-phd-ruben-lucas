import logging
from datetime import datetime, timedelta
import glob
import time
import pynvml
import psutil
from stable_baselines3.common.monitor import Monitor

import mlflow
import mlflow.sklearn
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.policies import BasePolicy

from gym import spaces
from typing import Callable, Tuple

import torch as th
import torch.nn as nn

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import gymnasium as gym
import tensorflow as tf
from tqdm import tqdm
from rl_studio.agents.utilities.plot_npy_dataset import plot_rewards
from rl_studio.agents.utilities.push_git_repo import git_add_commit_push
from rl_studio.algorithms.utils import (
    save_actorcritic_baselines_model,
)
from stable_baselines3.common.callbacks import CallbackList

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import EvalCallback
# from wandb.integration.sb3 import WandbCallback
# import wandb


from rl_studio.algorithms.ddpg import (
    ModifiedTensorBoard,
)


from rl_studio.agents.f1.loaders import (
    LoadAlgorithmParams,
    LoadEnvParams,
    LoadEnvVariablesDDPGCarla,
    LoadGlobalParams,
)
from rl_studio.agents.utils import (
    print_messages,
    render_params,
    save_dataframe_episodes,
    LoggingHandler,
)

from rl_studio.algorithms.ddpg import (
    OUActionNoise
)

from rl_studio.envs.gazebo.gazebo_envs import *
from rl_studio.envs.carla.carla_env import CarlaEnv

from stable_baselines3 import DDPG
# from stable_baselines3 import DDPG
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.logger import configure

try:
    sys.path.append(
        glob.glob(
            "../carla/dist/carla-*%d.%d-%s.egg"
            % (
                sys.version_info.major,
                sys.version_info.minor,
                "win-amd64" if os.name == "nt" else "linux-x86_64",
            )
        )[0]
    )
except IndexError:
    pass


# Function to update scatter plot with new data
def update_scatter_plot(ax, x, y, z, xlabel, ylabel, zlabel):
    ax.clear()
    ax.scatter(x, y, z)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    plt.draw()
    plt.pause(0.001)


def collect_usage():
    cpu_usage = psutil.cpu_percent(interval=None)  # Get CPU usage percentage
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    gpu_info = pynvml.nvmlDeviceGetUtilizationRates(handle)
    gpu_usage = gpu_info.gpu
    return cpu_usage, gpu_usage


def combine_attributes(obj1, obj2, obj3):
    combined_dict = {}

    # Extract attributes from obj1
    obj1_dict = obj1.__dict__
    for key, value in obj1_dict.items():
        combined_dict[key] = value

    # Extract attributes from obj2
    obj2_dict = obj2.__dict__
    for key, value in obj2_dict.items():
        combined_dict[key] = value

    # Extract attributes from obj3
    obj3_dict = obj3.__dict__
    for key, value in obj3_dict.items():
        combined_dict[key] = value

    return combined_dict


class PeriodicSaveCallback(BaseCallback):
    def __init__(self, env, params, save_path, save_freq=10000, verbose=1):
        super(PeriodicSaveCallback, self).__init__(verbose)
        self.save_path = save_path
        self.save_freq = save_freq
        self.step_count = 0
        self.env = env
        self.params = params

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        self.step_count += 1
        if self.step_count % self.save_freq == 0:
            model_save_path = os.path.join(self.save_path, f"model_{self.step_count}_steps")
            self.model.save(model_save_path)
            date_time = time.strftime('%Y%m%d-%H%M%S')

            mlflow.set_experiment("followlane_carla")
            with mlflow.start_run(nested=True):
                mlflow.log_param("model_type", "ddpg_bs")
                mlflow.log_metric("avg_speed", self.env.last_avg_speed)
                mlflow.log_metric("max_speed", self.env.last_max_speed)
                mlflow.log_metric("deviation", self.env.episode_last_d_deviation)
                mlflow.log_metric("cum_reward", self.env.last_cum_reward)
                mlflow.set_tag("detection_mode", self.params["detection_mode"])
                mlflow.log_param("actions", self.params["actions"])
                mlflow.log_param("zig_zag_punish", self.params["zig_zag_punish"])
                mlflow.set_tag("running_mode", self.params["running_mode"])
                mlflow.log_param("datetime", date_time)
                mlflow.log_metric("last_episode_steps", self.env.last_steps)
                mlflow.log_metric("steps", self.step_count)
                mlflow.log_artifact(model_save_path + ".zip", artifact_path="saved_models")
            # mlflow.log_metric("gamma", self.env.episode_d_deviation)
            # mlflow.log_metric("tau", self.env.episode_d_deviation)
            # mlflow.log_metric("lr", self.env.episode_d_deviation)
            # mlflow.log_metric("d_importance", self.env.episode_d_deviation)
            # mlflow.log_metric("v_importance", self.env.episode_d_deviation)
            # mlflow.log_metric("high_vel_punish", self.env.episode_d_deviation)
            if self.verbose > 0:
                print(f"Saved model at step {self.step_count}")
        return True


class ExplorationRateCallback(BaseCallback):
    def __init__(self, tensorboard, stage=None, initial_exploration_rate=1.0,
                 exploration_min=0.03, decay_rate=0.005, decay_steps=1000, verbose=1):
        super(ExplorationRateCallback, self).__init__(verbose)
        self.min_exploration_rate = exploration_min
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.current_step = 0
        self.tensorboard = tensorboard
        # Configure noise rates based on stage
        if stage in (None, "w"):
            self.w_initial = 0.07
            self.v_initial = 0
        else:
            self.w_initial = 0
            self.v_initial = 0.2

        self.w_exploration_rate = self.w_initial
        self.v_exploration_rate = self.v_initial
        self.n_actions = None  # Will be initialized at training start

    def _on_training_start(self):
        # Initialize number of actions and set initial action noise
        self.n_actions = self.model.action_space.shape[0]

        self.model.action_noise = NormalActionNoise(
            mean=np.zeros(self.n_actions),
            sigma=np.array([self.v_exploration_rate] + [self.w_exploration_rate])
        )
        self.tensorboard.update_stats(std_dev_v=self.v_exploration_rate)
        self.tensorboard.update_stats(std_dev_w=self.w_exploration_rate)

        if self.verbose > 0:
            print(f"Training started: Initial exploration rates set to v={self.v_exploration_rate}, w={self.w_exploration_rate}")
        return True

    def _on_step(self) -> bool:
        self.current_step += 1

        if self.current_step % self.decay_steps == 0:
            # Decay exploration rates for first and subsequent actions
            self.v_exploration_rate = max(self.min_exploration_rate,
                                          self.v_exploration_rate - self.decay_rate)
            self.w_exploration_rate = max(self.min_exploration_rate,
                                          self.w_exploration_rate - self.decay_rate)

            # Update the action noise with the new exploration rates
            self.model.action_noise = NormalActionNoise(
                mean=np.zeros(self.n_actions),
                sigma=np.array([self.v_exploration_rate] + [self.w_exploration_rate])
            )
            self.tensorboard.update_stats(std_dev_v=self.v_exploration_rate)
            self.tensorboard.update_stats(std_dev_w=self.w_exploration_rate)

        return True

class TrainerFollowLaneDDPGCarla:
    """
    Mode: training
    Task: Follow Line
    Algorithm: DDPG
    Agent: F1
    Simulator: Gazebo
    Framework: TensorFlow
    """

    def __init__(self, config):

        pynvml.nvmlInit()

        self.actor_loss = 0
        self.critic_loss = 0
        self.algoritmhs_params = LoadAlgorithmParams(config)
        self.env_params = LoadEnvParams(config)
        self.global_params = LoadGlobalParams(config)
        self.environment = LoadEnvVariablesDDPGCarla(config)
        self.environment.environment["debug_waypoints"] = False
        logs_dir = f"{self.global_params.logs_tensorboard_dir}/{self.algoritmhs_params.model_name}-{time.strftime('%Y%m%d-%H%M%S')}"
        self.tensorboard = ModifiedTensorBoard(
            log_dir=logs_dir
        )
        self.environment.environment["tensorboard"] = self.tensorboard

        self.loss = 0

        os.makedirs(f"{self.global_params.models_dir}", exist_ok=True)
        os.makedirs(f"{self.global_params.logs_dir}", exist_ok=True)
        os.makedirs(f"{self.global_params.metrics_data_dir}", exist_ok=True)
        os.makedirs(f"{self.global_params.metrics_graphics_dir}", exist_ok=True)

        agent_log_file = f"{self.global_params.logs_dir}/{time.strftime('%Y%m%d-%H%M%S')}_baselines_{self.global_params.mode}_{self.global_params.task}_{self.global_params.algorithm}_{self.global_params.agent}_{self.global_params.framework}.log"
        # self.log = LoggingHandler(agent_log_file)

        ## Load Carla server
        # CarlaEnv.__init__(self)
        self.environment.environment["actions"] = self.global_params.actions_set

        self.environment.environment["entropy_factor"] = config["settings"]["entropy_factor"]
        self.environment.environment["use_curves_state"] = config["settings"]["use_curves_state"]
        self.w_net_dir = self.environment.environment['retrain_ddpg_tf_model_name_w']
        self.v_net_dir = self.environment.environment['retrain_ddpg_tf_model_name_v']
        self.stage = self.environment.environment.get("stage")
        if self.stage == "v":
            self.environment.environment["w_net"] = DDPG.load(self.w_net_dir)

        self.env = gym.make(self.env_params.env_name, **self.environment.environment)
        self.all_steps = 0
        self.current_max_reward = 0
        self.best_epoch = 0
        self.episodes_speed = []
        self.episodes_d_reward = []
        self.episodes_steer = []
        self.episodes_reward = []
        self.step_fps = []
        self.bad_perceptions = 0
        self.crash = 0

        self.all_steps_reward = []
        self.all_steps_velocity = []
        self.all_steps_steer = []
        self.all_steps_state0 = []
        self.all_steps_state11 = []

        self.cpu_usages = 0
        self.gpu_usages = 0

        self.exploration = self.algoritmhs_params.std_dev if self.global_params.mode != "inference" else 0
        self.n_actions = self.env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(self.n_actions), sigma=self.exploration * np.ones(self.n_actions))


        self.params = {
            "policy": "CustomPolicy",
            "learning_rate": 0.00035,
            "buffer_size": 100000,
            "batch_size": 256,
            "gamma": 0.90,
            "tau": 0.005,
            "total_timesteps": 5000000
        }

        # Init Agents
        if self.environment.environment["mode"] in ["inference", "retraining"]:
            if self.stage == "w":
                actor_retrained_model = self.environment.environment['retrain_ddpg_tf_model_name_w']
            else:
                actor_retrained_model = self.environment.environment['retrain_ddpg_tf_model_name_v']
            self.ddpg_agent = DDPG.load(actor_retrained_model)
            # Set the environment on the loaded model
            self.ddpg_agent.set_env(self.env)
        else:
            # Assuming `self.params` and `self.global_params` are defined properly
            self.ddpg_agent = DDPG(
                # CustomDDPGPolicy,
                "MlpPolicy",
                self.env,
                policy_kwargs=dict(net_arch=dict(pi=[32, 32, 32], qf=[32, 32, 32])),
                learning_rate=self.params["learning_rate"],
                buffer_size=self.params["buffer_size"],
                batch_size=self.params["batch_size"],
                tau=self.params["tau"],
                gamma=self.params["gamma"],
                verbose=1,
                # tensorboard_log=f"{self.global_params.logs_tensorboard_dir}/{self.algoritmhs_params.model_name}-{time.strftime('%Y%m%d-%H%M%S')}"
            )

        print(self.ddpg_agent.actor)
        # Set the action noise on the loaded model
        self.ddpg_agent.action_noise = action_noise

        agent_logger = configure(agent_log_file, ["stdout", "csv", "tensorboard"])

        self.ddpg_agent.set_logger(agent_logger)
        random.seed(1)
        np.random.seed(1)
        tf.compat.v1.random.set_random_seed(1)

    def main(self):
        # run = wandb.init(
        #     project="rl-follow-lane",
        #     config=self.params,
        #     sync_tensorboard=True,
        # )
        exploration_rate_callback = ExplorationRateCallback(self.tensorboard,
                                                            stage=self.environment.environment.get("stage"),
                                                            initial_exploration_rate=self.exploration,
                                                            decay_rate= self.global_params.decrease_substraction,
                                                            decay_steps=self.global_params.steps_to_decrease,
                                                            exploration_min=self.global_params.decrease_min,
                                                            verbose=1)

        # Assuming `self.env` is your original environment
        self.eval_env = Monitor(self.env)

        # TODO Note that evalCallback is useful, but it slow down training getting stucked. To refine
        # wandb_callback = WandbCallback(gradient_save_freq=100, verbose=2)
        eval_callback = EvalCallback(
            self.eval_env,
            best_model_save_path=f"{self.global_params.models_dir}/{time.strftime('%Y%m%d-%H%M%S')}",
            eval_freq=20000,
            deterministic=True,
            render=False
        )

        params = {
            "detection_mode": self.environment.environment["detection_mode"],
            "actions": self.global_params.actions_set,
            "zig_zag_punish": self.environment.environment["punish_zig_zag_value"],
            "running_mode": self.environment.environment["mode"],
        }
        periodic_save_callback = PeriodicSaveCallback(
            env = self.env,
            params = params,
            save_path=f"{self.global_params.models_dir}/{time.strftime('%Y%m%d-%H%M%S')}",
            verbose=1
        )

        callback_list = CallbackList([exploration_rate_callback, eval_callback, periodic_save_callback])
        #callback_list = CallbackList([exploration_rate_callback, periodic_save_callback])

        # TODO (OJO) Falta por adaptar esto a las 2 redes
        if self.environment.environment["mode"] == "inference":
            self.evaluate_ddpg_agent(self.env, self.ddpg_agent, 10000)

        self.ddpg_agent.learn(total_timesteps=self.params["total_timesteps"],
                              callback=callback_list)



    def evaluate_ddpg_agent(self, env, agent, num_episodes):
        for episode in range(num_episodes):
            obs, _ = env.reset()
            done = False
            episode_reward = 0

            while not done:
                # Use the agent to predict the action without learning (no training)
                action, _states = agent.predict(obs, deterministic=True)

                # Take the action in the environment
                obs, reward, done, done, info = env.step(action)
                episode_reward += reward

            print(f"Episode {episode + 1}: Total Reward = {episode_reward}")

        # self.env.close()

    def set_stats(self, info, prev_state):
        self.episodes_speed.append(info["velocity"])
        self.episodes_steer.append(info["steering_angle"])
        self.episodes_d_reward.append(info["d_reward"])
        self.episodes_reward.append(info["reward"])

        self.all_steps_reward.append(info["reward"])
        self.all_steps_velocity.append(info["velocity"])
        self.all_steps_steer.append(info["steering_angle"])
        self.all_steps_state0.append(prev_state[0])
        self.all_steps_state11.append(prev_state[6])

        pass