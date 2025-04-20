import logging
from datetime import datetime, timedelta
import glob
import time
import pynvml
import psutil
from typing import Callable

import mlflow
import mlflow.sklearn
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import MlpExtractor

import torch as th
import torch.nn as nn

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import gymnasium as gym
import tensorflow as tf
from torch.fx.experimental.unification.multipledispatch import restart_ordering
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
    LoadEnvVariablesPPOCarla,
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

from stable_baselines3 import PPO
# from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.logger import configure
import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy

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
            model_save_path = os.path.join(self.save_path, f"model_{self.step_count}_steps")
            self.model.save(model_save_path)
            date_time = time.strftime('%Y%m%d-%H%M%S')

            mlflow.set_experiment("followlane_carla")
            with mlflow.start_run(nested=True):
                mlflow.log_param("model_type", "ppo_bs")
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

class SetClipRangeCallback(BaseCallback):
    def __init__(self, clip_range, verbose=0):
        """
        Callback to set the clip_range once at the start of training.

        :param clip_range: (float or callable) The new clip range value or function.
        :param verbose: (int) Verbosity level.
        """
        super(SetClipRangeCallback, self).__init__(verbose)
        self.clip_range = clip_range

    def _on_training_start(self) -> None:
        self.model.clip_range = self.clip_range

    def _on_step(self):
        return True

class DynamicClipRange:
    def __init__(self, initial_clip_range, min_clip_range):
        self.initial_clip_range = initial_clip_range
        self.min_clip_range = min_clip_range

    def __call__(self, progress_remaining):
        return max(self.min_clip_range, self.initial_clip_range * progress_remaining)

class ExplorationRateCallback(BaseCallback):
    def __init__(self, stage=None, initial_log_std=-1.0,
                 min_log_std=-5.8, decay_rate=0.01, decay_steps=1000, verbose=1):
        super(ExplorationRateCallback, self).__init__(verbose)
        self.initial_log_std = initial_log_std
        self.min_log_std = min_log_std
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps

        self.no_exploration = torch.full_like(
            self.model.policy.log_std[0:2], min_log_std
        ).to(self.model.policy.log_std.device)

        if stage in (None, "w"):
            self.w_initial = initial_log_std
            self.v_initial = initial_log_std
        else:
            self.w_initial = -0.8
            self.v_initial = -0.8
        self.current_step = 0

    def _on_training_start(self):
        # Set initial log_std
        #self.model.policy.log_std.data = torch.full_like(
        #    self.model.policy.log_std, self.initial_log_std
        #).to(self.model.policy.log_std.device)
        self.model.policy.log_std.data[1:2] = torch.full_like(
            self.model.policy.log_std[1:2], self.w_initial
        ).to(self.model.policy.log_std.device)
        self.model.policy.log_std.data[:1] = torch.full_like(
           self.model.policy.log_std[:1], self.v_initial
        ).to(self.model.policy.log_std.device)

    def _on_step(self) -> bool:
        self.current_step += 1

        # Check if both components have reached the min_log_std
        # if (
        #     torch.all(self.model.policy.log_std[:1] <= self.min_log_std).item() and
        #     torch.all(self.model.policy.log_std[1:2] <= self.min_log_std).item()
        # ):
        #     # Reset to initial values
        #     self.model.policy.log_std.data[1:2] = torch.full_like(
        #         self.model.policy.log_std[1:2], self.w_initial
        #     ).to(self.model.policy.log_std.device)
        #     self.model.policy.log_std.data[:1] = torch.full_like(
        #         self.model.policy.log_std[:1], self.v_initial
        #     ).to(self.model.policy.log_std.device)
        #
        #     if self.verbose > 0:
        #         print(
        #             f"Step {self.current_step}: log_std reset to initial values "
        #             f"v_initial={self.v_initial}, w_initial={self.w_initial}"
        #         )
        #     return True

        if self.current_step % self.decay_steps == 0:
            new_log_std = self.model.policy.log_std.clone()
            new_log_std[1:2] =  torch.maximum(
                 torch.full_like(self.model.policy.log_std[1:2], self.min_log_std),
                 self.model.policy.log_std[1:2] - self.decay_rate
            )
            new_log_std[:1] = torch.maximum(
                torch.full_like(self.model.policy.log_std[:1], self.min_log_std),
                self.model.policy.log_std[:1] - self.decay_rate
            )
            self.last_log_std = new_log_std
            self.model.policy.log_std.data = new_log_std

            if self.verbose > 0:
                print(f"Step {self.current_step}: Updated log_std to {new_log_std.detach().cpu().numpy()}")

        if np.random.rand() < 0.5:
            self.model.policy.log_std.data = self.no_exploration
        else:
            self.model.policy.log_std.data = self.last_log_std

        return True

class EntropyCoefficientCallback(BaseCallback):
    def __init__(self, initial_ent_coef=0.01, min_ent_coef=0.001, decay_rate=0.0001, decay_steps=1000, verbose=0):
        """
        Callback to decay the entropy coefficient over training.

        :param initial_ent_coef: (float) Initial entropy coefficient.
        :param min_ent_coef: (float) Minimum entropy coefficient.
        :param decay_rate: (float) Amount to decrease the entropy coefficient by at each decay step.
        :param decay_steps: (int) Number of steps between each decay.
        :param verbose: (int) Verbosity level.
        """
        super(EntropyCoefficientCallback, self).__init__(verbose)
        self.initial_ent_coef = initial_ent_coef
        self.min_ent_coef = min_ent_coef
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.current_step = 0

    def _on_training_start(self):
        # Set the initial entropy coefficient
        self.model.ent_coef = self.initial_ent_coef
        if self.verbose > 0:
            print(f"Training started: Entropy coefficient set to {self.model.ent_coef}")

    def _on_step(self) -> bool:
        self.current_step += 1
        if self.current_step % self.decay_steps == 0:
            # Decay the entropy coefficient
            new_ent_coef = max(self.min_ent_coef, self.model.ent_coef - self.decay_rate)
            self.model.ent_coef = new_ent_coef

            if self.verbose > 0:
                print(f"Step {self.current_step}: Updated entropy coefficient to {self.model.ent_coef}")

        return True


class CustomEvalCallback(EvalCallback):
    def __init__(self, env, tensorboard_writer, *args, **kwargs):
        super(CustomEvalCallback, self).__init__(env, *args, **kwargs)
        self.env = env
        self.tensorboard_writer = tensorboard_writer

    def _on_step(self) -> bool:
        # Call the parent class's method to perform the regular evaluation
        result = super()._on_step()

        # Log the 2D histogram or scatter plot during evaluation
        if self.n_calls % self.eval_freq == 0:
            self.log_action_map()

        return result

    def log_action_map(self):
        # Extract relevant data from the environment
        distance_errors = []
        velocities = []
        throttle = []
        steer = []

        obs, _ = self.env.reset()
        done = False
        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, done, done, info = self.env.step(action)

            distance_errors.append(info["distance_error"][0])  # Adjust based on your environment
            velocities.append(info["velocity"])  # Adjust based on your environment
            throttle.append(action[0])  # Assuming action is scalar or index 0 for one component
            steer.append(action[1])  # Assuming action is scalar or index 0 for one component

        # Create a 2D histogram or scatter plot
        with self.tensorboard_writer.writer.as_default():
            # Log a custom color-coded map
            t_data = np.array([distance_errors, velocities, throttle]).T  # Shape (N, 3)
            tf.summary.image(
                "distance_velocity_throttle_map",
                plot_colormap(t_data, title="Distance vs Velocity (Color: Throttle)"),
                step=self.num_timesteps,
            )
            s_data = np.array([distance_errors, velocities, steer]).T  # Shape (N, 3)
            tf.summary.image(
                "distance_velocity_steer_map",
                plot_colormap(s_data, title="Distance vs Velocity (Color: steer)"),
                step=self.num_timesteps,
            )


import io

def plot_colormap(data, title="Colormap"):
    """
    Generate a scatter plot with color-coded actions.
    :param data: numpy array of shape (N, 3), where columns represent distance, velocity, and actions.
    :param title: Title of the plot.
    :return: TensorFlow image tensor.
    """
    distances, velocities, actions = data[:, 0], data[:, 1], data[:, 2]

    plt.figure(figsize=(6, 6))
    scatter = plt.scatter(distances, velocities, c=actions, cmap="viridis", s=10)
    plt.colorbar(scatter, label="Actions")
    plt.xlabel("Distance Error")
    plt.ylabel("Velocity")
    plt.title(title)

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close()

    # Convert plot to TensorFlow image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)  # Add batch dimension
    return image

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

class TrainerFollowLanePPOCarla:
    """
    Mode: training
    Task: Follow Line
    Algorithm: PPO
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
        self.environment = LoadEnvVariablesPPOCarla(config)
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

        self.rebuild_env()
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


        # Init Agents
        if self.environment.environment["mode"] in ["inference", "retraining"]:
            actor_retrained_model = self.environment.environment['retrain_ppo_tf_model_name']
            self.ppo_agent = PPO.load(actor_retrained_model)
            # Set the environment on the loaded model
            self.ppo_agent.set_env(self.env)
        else:
            # Assuming `self.params` and `self.global_params` are defined properly
            self.ppo_agent = PPO(
                CustomActorCriticPolicy,  # Use the custom policy class
                #"MlpPolicy",
                self.env,
                policy_kwargs=dict(
                    net_arch=dict(
                        pi=[64, 128, 128, 64],  # The architecture for the policy network
                        vf=[64, 128, 128, 64]  # The architecture for the value network
                    ),
                    activation_fn=nn.ReLU,
                    log_std_init=-1.5,
                ),
                max_grad_norm=1,
                learning_rate=linear_schedule(self.environment.environment["critic_lr"]),
                gamma=self.algoritmhs_params.gamma,
                # gae_lambda=0.9,
                ent_coef=0.05,
                clip_range=self.algoritmhs_params.epsilon,
                batch_size=1024,
                verbose=1,
                # Uncomment if you want to log to TensorBoard
                # tensorboard_log=f"{self.global_params.logs_tensorboard_dir}/{self.algoritmhs_params.model_name}-{time.strftime('%Y%m%d-%H%M%S')}"
            )

        print(self.ppo_agent.policy)

        agent_logger = configure(agent_log_file, ["stdout", "csv", "tensorboard"])

        self.ppo_agent.set_logger(agent_logger)
        random.seed(1)
        np.random.seed(1)
        tf.compat.v1.random.set_random_seed(1)

    def main(self):
        hyperparams = combine_attributes(self.algoritmhs_params,
                                         self.environment,
                                         self.global_params)
        self.tensorboard.update_hyperparams(hyperparams)
        # run = wandb.init(
        #     project="rl-follow-lane",
        #     config=self.params,
        #     sync_tensorboard=True,
        # )

        # log_std = -0.223 <= 0.8;  -1 <= 0.36
        exploration_rate_callback = ExplorationRateCallback(
            stage=self.environment.environment.get("stage"), initial_log_std=-0.5,
            min_log_std=self.environment.environment.get("decrease_min"), decay_rate=0.03,
            decay_steps=2000)
        entropy_callback = EntropyCoefficientCallback(
            initial_ent_coef=0.02,  # Starting entropy coefficient
            min_ent_coef=0.02,  # Minimum entropy coefficient
            decay_rate=0.0001,  # Decay amount per step
            decay_steps=2000,  # Decay every 1000 steps
            verbose=1
        )

        # Instantiate the dynamic clip range
        dynamic_clip_range = DynamicClipRange(initial_clip_range=0.2, min_clip_range=0.1)
        epsilon_callback = SetClipRangeCallback(clip_range=dynamic_clip_range)

        # wandb_callback = WandbCallback(gradient_save_freq=100, verbose=2)
        eval_callback = CustomEvalCallback(
            self.env,
            tensorboard_writer=self.tensorboard,
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
        self.checkpoints_path=f"{self.global_params.models_dir}/{time.strftime('%Y%m%d-%H%M%S')}"
        periodic_save_callback = PeriodicSaveCallback(
            env = self.env,
            params = params,
            save_path=self.checkpoints_path,
            verbose=1
        )

        #callback_list = CallbackList([exploration_rate_callback, eval_callback, periodic_save_callback])
        #callback_list = CallbackList([exploration_rate_callback, periodic_save_callback])
        #callback_list = CallbackList([periodic_save_callback])
        #callback_list = CallbackList([periodic_save_callback, eval_callback])
        callback_list = CallbackList([
            exploration_rate_callback,
            entropy_callback,
            periodic_save_callback,
            epsilon_callback,
            eval_callback])

        if self.environment.environment["mode"] in ["inference"]:
            self.evaluate_ddpg_agent(self.env, self.ppo_agent, 10000)

        self.ppo_agent.learn(total_timesteps=5000000,
                              callback=callback_list)

    # self.restart_training(callback_list)

        # self.env.close()

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


    def restart_training(self, callback_list):
        try:
            with open("/tmp/.carlalaunch_stdout.log", "w") as out, open("/tmp/.carlalaunch_stderr.log", "w") as err:
                print("carla broken. Restarting Carla")
                subprocess.Popen([os.environ["CARLA_ROOT"] + "CarlaUE4.sh",
                                  f"--world-port={self.environment.environment['carla_client']}", "-RenderOffScreen",
                                  "-quality-level=Low"], stdout=out, stderr=err)
                self.ppo_agent = PPO.load(f"{self.checkpoints_path}/best_model.zip")
                # Set the environment on the loaded model
                self.ppo_agent.set_env(self.env)
                self.ppo_agent.learn(total_timesteps=self.params["total_timesteps"],
                                     callback=callback_list)
        except Exception:  # TODO better control the exception type
            self.restart_training(callback_list)

    def rebuild_env(self):
        try:
            self.env = gym.make(self.env_params.env_name, **self.environment.environment)
        except Exception:  # TODO better control the exception type
            print("error")

    # with open("/tmp/.carlalaunch_stdout.log", "w") as out, open("/tmp/.carlalaunch_stderr.log", "w") as err:
            #     print("carla broken. Restarting Carla")
            #     subprocess.Popen([os.environ["CARLA_ROOT"] + "CarlaUE4.sh",
            #                       f"--world-port={self.environment.environment['carla_client']}",
            #                       "-RenderOffScreen",
            #                       "-quality-level=Low"], stdout=out, stderr=err)
            # self.rebuild_env()