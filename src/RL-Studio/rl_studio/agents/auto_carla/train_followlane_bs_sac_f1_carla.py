import logging
from datetime import datetime, timedelta
import glob
import time
import random

import numpy as np
import pynvml
import psutil
from typing import Callable
from stable_baselines3.common.vec_env import DummyVecEnv
from rl_studio.envs.carla.utils.logger import logger

import mlflow
import mlflow.sklearn
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import MlpExtractor
from rl_studio.agents.utilities.extract_reward_function import extract_reward_function

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
import torch

# from wandb.integration.sb3 import WandbCallback
# import wandb


from rl_studio.algorithms.ddpg import (
    ModifiedTensorBoard,
)


from rl_studio.agents.f1.loaders import (
    LoadAlgorithmParams,
    LoadEnvParams,
    LoadEnvVariablesSACCarla,
    LoadGlobalParams,
)
from rl_studio.agents.utils import (
    print_messages,
    render_params,
    save_dataframe_episodes,
    LoggingHandler,
)

from rl_studio.envs.gazebo.gazebo_envs import *
from rl_studio.envs.carla.carla_env import CarlaEnv

from stable_baselines3 import SAC
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
    # handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    # gpu_info = pynvml.nvmlDeviceGetUtilizationRates(handle)
    # gpu_usage = gpu_info.gpu
    # return cpu_usage, gpu_usage


class CustomPolicyNetwork(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super(CustomPolicyNetwork, self).__init__(observation_space, features_dim)
        self.net = nn.Sequential(
            nn.Linear(observation_space.shape[0], 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, features_dim)
        )

    def forward(self, observations):
        return self.net(observations)

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
            model_save_path = os.path.join(self.save_path, f"model_{self.step_count}_steps_{time.strftime('%Y%m%d-%H%M%S')}")
            self.model.save(model_save_path)
            # date_time = time.strftime('%Y%m%d-%H%M%S')

            # mlflow.set_experiment("followlane_carla")
            # with mlflow.start_run(nested=True):
            #     mlflow.log_param("model_type", "sac_bs")
            #     mlflow.log_metric("avg_speed", self.env.last_avg_speed)
            #     mlflow.log_metric("max_speed", self.env.last_max_speed)
            #     mlflow.log_metric("deviation", self.env.episode_last_d_deviation)
            #     mlflow.log_metric("cum_reward", self.env.last_cum_reward)
            #     mlflow.set_tag("detection_mode", self.params["detection_mode"])
            #     mlflow.log_param("actions", self.params["actions"])
            #     mlflow.log_param("zig_zag_punish", self.params["zig_zag_punish"])
            #     mlflow.set_tag("running_mode", self.params["running_mode"])
            #     mlflow.log_param("datetime", date_time)
            #     mlflow.log_metric("last_episode_steps", self.env.last_steps)
            #     mlflow.log_metric("steps", self.step_count)
            #     mlflow.log_artifact(model_save_path + ".zip", artifact_path="saved_models")
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
                 exploration_min=0.03,
                 decay_rate=0.005,
                 initial_std_v=None,
                 initial_std_w=None,
                 decay_steps=1000, verbose=1):
        super(ExplorationRateCallback, self).__init__(verbose)
        self.min_exploration_rate = exploration_min
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.current_step = 0
        self.tensorboard = tensorboard
        # Configure noise rates based on stage
        if stage in (None, "w"):
            self.w_initial = 0.4 if initial_std_w is None else initial_std_w
            self.v_initial = 0 if initial_std_v is None else initial_std_v
        elif stage == "v":
            self.w_initial = 0.1 if initial_std_w is None else initial_std_w
            self.v_initial = 0.3 if initial_std_v is None else initial_std_v
        else:
            self.w_initial = 0.3 if initial_std_w is None else initial_std_w
            self.v_initial = 0.6 if initial_std_v is None else initial_std_v

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
            if self.verbose > 0:
                print(f"Step {self.current_step}: Updated exploration rates to v={self.v_exploration_rate}, w={self.w_exploration_rate}")

        cycle = 1000
        active_phase = 800

        if (self.current_step % cycle) < active_phase and np.random.rand() < 0.5:
            self.model.action_noise = NormalActionNoise(
                mean=np.zeros(self.n_actions),
                sigma=np.array([0, 0])
            )
        else:
            self.model.action_noise = NormalActionNoise(
                mean=np.zeros(self.n_actions),
                sigma=np.array([self.v_exploration_rate] + [self.w_exploration_rate])
            )

        return True



import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import FlattenExtractor


class CustomActorCriticPolicy(ActorCriticPolicy): # No se esta usando ahora mismo!!!!
    def __init__(self, *args, **kwargs):
        super(CustomActorCriticPolicy, self).__init__(*args, **kwargs)

        # Define a custom flatten extractor
        #self.features_extractor = FlattenExtractor(self.observation_space)

        # Ensure the feature dimension is correctly calculated
        # self.feature_dim = int(torch.prod(torch.tensor(self.observation_space.shape)))

        # Optional: If feature_dim is not matching, add a linear layer to match
        # self.feature_transform = nn.Sequential(
        #    nn.Linear(self.feature_dim, 256),  # Transform to the correct size
        #    nn.ReLU()
        # )

        # Policy network
        self.action_net = nn.Sequential(
            #nn.Linear(256, 256),  # Hidden layer
            #nn.ReLU(),  # Intermediate activation
            #nn.Linear(256, 256),  # Hidden layer
            #nn.ReLU(),  # Intermediate activation
            #nn.Linear(256, 256),  # Hidden layer
            #nn.ReLU(),  # Intermediate activation
            #nn.Linear(256, self.action_space.shape[0]),  # Output layer
            self.action_net,
            nn.Tanh()  # Tanh activation for output layer
        )

        # Value network
        # self.value_net = nn.Sequential(
        #     nn.Linear(256, 256),  # Hidden layer for value function
        #     nn.ReLU(),  # Intermediate activation
        #     nn.Linear(256, 256),  # Hidden layer
        #     nn.ReLU(),  # Intermediate activation
        #     nn.Linear(256, 256),  # Hidden layer
        #     nn.ReLU(),  # Intermediate activation
        #     nn.Linear(256, 1)  # Output for value function
        # )

    # def calculate_log_probs(self, action_logits):
    #     # Assuming you have a mean and stddev for your actions
    #     # Use a normal distribution to compute log probabilities
    #     mean = action_logits  # or some transformation based on your model
    #     stddev = torch.exp(self.log_std)  # log_std should be a learnable parameter
    #
    #     # Create a normal distribution
    #     dist = torch.distributions.Normal(mean, stddev)
    #
    #     # Calculate log probabilities
    #     log_probs = dist.log_prob(action_logits)
    #     return log_probs.sum(dim=-1)

    # def forward(self, obs):
    #     features = self.features_extractor(obs)  # Extract features
    #     features_transformed = self.feature_transform(features)  # Transform features to expected size
    #     action_logits = self.action_net(features_transformed)  # Use the modified action_net
    #     value = self.value_net(features_transformed)  # Get the value using the transformed features
    #     log_probs = self.calculate_log_probs(action_logits)  # Implement this method based on your action space
    #     return action_logits, value, log_probs


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func

class TrainerFollowLaneSACCarla:
    """
    Mode: training
    Task: Follow Line
    Algorithm: sac
    Agent: F1
    Simulator: Gazebo
    Framework: TensorFlow
    """

    def __init__(self, config):

        # pynvml.nvmlInit()
        log_path = f"./logs_episode/sac/{time.strftime('%Y%m%d-%H%M%S')}.log"
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(file_handler)

        self.actor_loss = 0
        self.critic_loss = 0
        self.algoritmhs_params = LoadAlgorithmParams(config)
        self.env_params = LoadEnvParams(config)
        self.global_params = LoadGlobalParams(config)
        self.environment = LoadEnvVariablesSACCarla(config)
        logs_dir = f"{self.global_params.logs_tensorboard_dir}/{time.strftime('%Y%m%d-%H%M%S')}"
        self.tensorboard = ModifiedTensorBoard(
            log_dir=f"{logs_dir}/sac/overall"
        )

        self.environment.environment["tensorboard"] = self.tensorboard
        self.environment.environment["tensorboard_logs_dir"] = logs_dir
        self.environment.environment["compensated_inits"] = self.global_params.compensated_inits
        self.environment.environment["random_direction"] = self.global_params.random_direction
        self.environment.environment["random_speeds"] = self.global_params.random_speeds
        self.environment.environment["seed"] = self.global_params.seed

        self.loss = 0

        os.makedirs(f"{self.global_params.models_dir}", exist_ok=True)
        os.makedirs(f"{self.global_params.logs_dir}", exist_ok=True)
        os.makedirs(f"{self.global_params.metrics_data_dir}", exist_ok=True)
        os.makedirs(f"{self.global_params.metrics_graphics_dir}", exist_ok=True)

        agent_log_file = f"{self.global_params.logs_dir}/{time.strftime('%Y%m%d-%H%M%S')}_baselines_{self.global_params.mode}_{self.global_params.task}_{self.global_params.algorithm}_{self.global_params.agent}_{self.global_params.framework}.log"
        # self.log = LoggingHandler(agent_log_file)

        ## Load Carla server
        # CarlaEnv.__init__(self)

        self.env = gym.make(self.env_params.env_name, **self.environment.environment)

        self.exploration = self.algoritmhs_params.std_dev if self.global_params.mode != "inference" else 0
        self.n_actions = self.env.action_space.shape[-1]
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

        # TODO This must come from config states in yaml
        state_size = len(self.environment.environment["x_row"]) + 2
        type(self.env.action_space)

        self.params = {
            "policy": "MlpPolicy",
            "learning_rate": self.environment.environment["critic_lr"],
            "gamma": self.algoritmhs_params.gamma,
            "epsilon": self.algoritmhs_params.epsilon,
            "total_timesteps": 5000000,
        }

        env = DummyVecEnv([lambda: self.env])
        # Init Agents
        if self.environment.environment["mode"] in ["inference", "retraining"]:
            actor_retrained_model = self.environment.environment['retrain_sac_tf_model_name']
            self.sac_agent = SAC.load(actor_retrained_model)
            # Set the environment on the loaded model
            self.sac_agent.set_env(env)
        else:
            # Assuming `self.params` and `self.global_params` are defined properly
            self.sac_agent = SAC(
                'MlpPolicy',
                env,
                policy_kwargs=dict(
                    net_arch=dict(
                        pi=self.global_params.net_arch,  # The architecture for the policy network
                        qf=self.global_params.net_arch
                    ),
                    # activation_fn=nn.ReLU,
                    # ortho_init=True,
                ),
                learning_rate=self.params["learning_rate"],
                buffer_size=500000,
                batch_size=256,
                tau=0.002,
                ent_coef='auto_0.1',
                target_entropy='auto',
                gamma=self.params["gamma"],
                train_freq=2,
                verbose=0,
                seed=self.global_params.seed
            )
            
        logger.info(self.sac_agent.policy)

        agent_logger = configure(agent_log_file, ["stdout", "csv", "tensorboard"])
        # agent_logger = configure(agent_log_file, ["tensorboard"])

        self.sac_agent.set_logger(agent_logger)
        random.seed(self.global_params.seed)
        np.random.seed(self.global_params.seed)
        tf.compat.v1.random.set_random_seed(self.global_params.seed)
        th.manual_seed(self.global_params.seed)
        if th.cuda.is_available():
            th.cuda.manual_seed_all(self.global_params.seed)
            th.backends.cudnn.deterministic = True
            th.backends.cudnn.benchmark = False

    def main(self):
        hyperparams = self.tensorboard.get_hparams(self.algoritmhs_params,
                                                   self.environment,
                                                   self.global_params)
        all_config = self.tensorboard.combine_attributes(self.algoritmhs_params,
                                                         self.environment,
                                                         self.global_params)
        reward_filename = f"{os.getcwd()}/envs/carla/followlane/followlane_carla_sb_3.py"
        reward_method = 'rewards_easy'
        reward_function = extract_reward_function(reward_filename, reward_method)
        all_config['reward_function'] = reward_function

        self.tensorboard.update_hpparams(hyperparams)
        self.tensorboard.update_hyperparams(all_config)
        # run = wandb.init(
        #     project="rl-follow-lane",
        #     config=self.params,
        #     sync_tensorboard=True,
        # )

        exploration_rate_callback = ExplorationRateCallback(self.tensorboard,
                                                            stage=self.environment.environment.get("stage"),
                                                            initial_exploration_rate=self.exploration,
                                                            initial_std_v=self.global_params.initial_std_v,
                                                            initial_std_w=self.global_params.initial_std_w,
                                                            decay_rate= self.global_params.decrease_substraction,
                                                            decay_steps=self.global_params.steps_to_decrease,
                                                            exploration_min=self.global_params.decrease_min,
                                                            verbose=1)
        # wandb_callback = WandbCallback(gradient_save_freq=100, verbose=2)
        eval_callback = EvalCallback(
            self.env,
            best_model_save_path=f"{self.global_params.models_dir}/{time.strftime('%Y%m%d-%H%M%S')}",
            eval_freq=100000,
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

        if self.environment.environment["mode"] in ["inference"]:
            self.evaluate_ddpg_agent(self.env, self.sac_agent, 10000, 2000)

        callbacks_to_add = [
            periodic_save_callback,
            eval_callback
        ]

        if self.global_params.steps_to_decrease < 100000:
            print("Added exploration callback!")
            callbacks_to_add.append(exploration_rate_callback)

        callback_list = CallbackList(
            callbacks_to_add
        )

        self.sac_agent.learn(total_timesteps=self.params["total_timesteps"],
                              callback=callback_list)

        # self.env.close()

    def evaluate_ddpg_agent(self, env, agent, num_episodes, num_steps):
        for episode in range(num_episodes):
            obs, _ = env.reset()
            done = False
            episode_reward = 0
            step = 0
            while not done and step < num_steps:
                step += 1
                # Use the agent to predict the action without learning (no training)
                action, _states = agent.predict(obs, deterministic=True)

                # Take the action in the environment
                obs, reward, done, done, info = env.step(action)
                episode_reward += reward

            print(f"Episode {episode + 1}: Total Reward = {episode_reward}")

