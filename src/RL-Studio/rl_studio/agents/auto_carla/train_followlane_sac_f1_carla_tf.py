import logging
from datetime import datetime, timedelta
import glob
import time
import pynvml
import psutil

import torch as th
import torch.nn as nn

import mlflow
import mlflow.sklearn
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
from rl_studio.algorithms.ddpg import (
    ModifiedTensorBoard,
)

from rl_studio.algorithms.ddpg import (
    ModifiedTensorBoard,
    OUActionNoise
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

class CustomActorCriticPolicy(SAC):
    def __init__(self, *args, **kwargs):
        super(CustomActorCriticPolicy, self).__init__(*args, **kwargs)
        self.actor_net = nn.Sequential(
            nn.Linear(self.features_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3)  # 2 continuous actions + 1 binary action
        )
        self.critic_net = nn.Sequential(
            nn.Linear(self.features_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def _predict(self, observations, deterministic=False):
        features = self.extractor(observations)
        action_logits = self.actor_net(features)
        continuous_actions = th.sigmoid(action_logits[:, :2])  # Scale to [0, 1]
        binary_action_logits = action_logits[:, 2]
        binary_actions = (binary_action_logits > 0).float()
        actions = th.cat([continuous_actions, binary_actions.unsqueeze(-1)], dim=-1)
        return actions

    def forward(self, observations, deterministic=False):
        return self._predict(observations, deterministic)


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


class TrainerFollowLaneSACCarla:
    """
    Mode: training
    Task: Follow Line
    Algorithm: SAC
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
        self.environment = LoadEnvVariablesSACCarla(config)
        self.environment.environment["debug_waypoints"] = self.global_params.debug_waypoints
        self.loss = 0
        self.environment.environment["estimated_steps"] = 5000

        self.tensorboard = ModifiedTensorBoard(
            log_dir=f"{self.global_params.logs_tensorboard_dir}/{self.algoritmhs_params.model_name}-{time.strftime('%Y%m%d-%H%M%S')}"
        )
        self.environment.environment["tensorboard"] = self.tensorboard


        self.params = {
            "policy": "MlpPolicy",
            "learning_rate": self.environment.environment["critic_lr"],
            "gamma": self.algoritmhs_params.gamma,
            "epsilon": self.algoritmhs_params.epsilon,
            "total_timesteps": 5000000,
            "batch_size": 1024
        }

        os.makedirs(f"{self.global_params.models_dir}", exist_ok=True)
        os.makedirs(f"{self.global_params.logs_dir}", exist_ok=True)
        os.makedirs(f"{self.global_params.metrics_data_dir}", exist_ok=True)
        os.makedirs(f"{self.global_params.metrics_graphics_dir}", exist_ok=True)

        log_file = f"{self.global_params.logs_dir}/{time.strftime('%Y%m%d-%H%M%S')}_{self.global_params.mode}_{self.global_params.task}_{self.global_params.algorithm}_{self.global_params.agent}_{self.global_params.framework}.log"
        agent_log_file = f"{self.global_params.logs_dir}/{time.strftime('%Y%m%d-%H%M%S')}_baselines_{self.global_params.mode}_{self.global_params.task}_{self.global_params.algorithm}_{self.global_params.agent}_{self.global_params.framework}.log"
        self.log = LoggingHandler(log_file)
        # self.log.logger.basicConfig(filename=log_file, level=logging.INFO)

        ## Load Carla server
        # CarlaEnv.__init__(self)

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

        # TODO This must come from config states in yaml
        state_size = len(self.environment.environment["x_row"]) + 2
        self.ou_noise = OUActionNoise(
            mean=np.zeros(1),
            std_deviation=float(self.exploration) * np.ones(1),
        )
        type(self.env.action_space)

        # Init Agents
        if self.environment.environment["mode"] in ["inference", "retraining"]:
            actor_retrained_model = self.environment.environment['retrain_sac_tf_actor_model_name']
            self.sac_agent = SAC.load(actor_retrained_model)
        else:
            self.sac_agent =  SAC(
                'MlpPolicy',
                self.env,
                learning_rate=3e-4,
                buffer_size=1000000,
                batch_size=256,
                tau=0.005,
                gamma=0.99,
                train_freq=1,
                gradient_steps=1,
                verbose=1
            )

        agent_logger = configure(agent_log_file, ["stdout", "csv", "tensorboard"])

        self.sac_agent.set_logger(agent_logger)
        random.seed(1)
        np.random.seed(1)
        tf.compat.v1.random.set_random_seed(1)

    def save_if_best_epoch(self, episode, step, cumulated_reward):
        if self.current_max_reward <= cumulated_reward:
            self.current_max_reward = cumulated_reward
            self.best_epoch = episode

            save_actorcritic_baselines_model(
                self.sac_agent,
                self.global_params,
                self.algoritmhs_params,
                self.environment.environment,
                self.current_max_reward,
                episode,
                "IMPROVED",
            )

            self.log.logger.info(
                f"\nsaving best lap\n"
                f"in episode = {episode}\n"
                f"current_max_reward = {cumulated_reward}\n"
                f"steps = {step}\n"
            )
        if episode - 100 > self.best_epoch:
            self.best_epoch = episode

            save_actorcritic_baselines_model(
                self.sac_agent,
                self.global_params,
                self.algoritmhs_params,
                self.environment.environment,
                self.current_max_reward,
                episode,
                "BATCH",
            )

    def log_and_plot_rewards(self, episode, step, cumulated_reward):
        # Showing stats in screen for monitoring. Showing every 'save_every_step' value
        if not self.all_steps % self.env_params.save_every_step:
            file_name = save_dataframe_episodes(
                self.environment.environment,
                self.global_params.metrics_data_dir,
                cumulated_reward,
            )
            plot_rewards(
                self.global_params.metrics_data_dir,
                file_name
            )
            git_add_commit_push("automatic_rewards_update")
            self.log.logger.debug(
                f"SHOWING BATCH OF STEPS\n"
                f"current_max_reward = {self.current_max_reward}\n"
                f"current epoch = {episode}\n"
                f"current step = {step}\n"
            )

    def one_step_iteration(self, episode, step, prev_state, cumulated_reward, bad_perception):
        self.all_steps += 1

        # TODO ñapa para decelerar y no hacer giros bruscos cuando se pierda la percepción
        #if bad_perception:
        #    action = [0, 0]
        #    state, reward, done, info = self.env.step(action)
        #    return state, cumulated_reward, done, info["bad_perception"]

        prev_state_fl = prev_state.astype(np.float32)
        tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state_fl), 0)

        action, _states = self.sac_agent.predict(tf_prev_state, deterministic=True)

        act = action[0]
        act[1] = act[1] - 0.5

        self.tensorboard.update_actions(act, self.all_steps)

        state, reward, done, done, info = self.env.step(act)
        self.step_fps.append(info["fps"])
        # in case perception was bad, we keep the previous frame
        if info["bad_perception"]:
            state = prev_state
            self.bad_perceptions += 1
        if info["crash"]:
            self.crash += 1
        self.set_stats(info, prev_state)

        if not self.all_steps % 3000:
            self.cpu_usages, self.gpu_usages = collect_usage()

        if self.all_steps % self.global_params.steps_to_decrease == 0:
            self.exploration = max(self.global_params.decrease_min,
                                   self.exploration - self.global_params.decrease_substraction)
            self.log.logger.info("decreasing exploration to ", self.exploration)
            self.ou_noise = OUActionNoise(
                mean=np.zeros(1),
                std_deviation=float(self.exploration) * np.ones(1),
            )
            # self.tensorboard.update_weights(agent_weights, self.all_steps)

        cumulated_reward += reward

        if self.global_params.show_monitoring:
            self.log.logger.debug(
                f"\nstate = {state}\n"
                f"state type = {type(state)}\n"
                f"prev_state = {prev_state}\n"
                f"prev_state = {type(prev_state)}\n"
                f"action = {action}\n"
                f"actions type = {type(action)}\n"
                f"\nepisode = {episode}\n"
                f"step = {step}\n"
                f"actions_len = {len(self.global_params.actions_set)}\n"
                f"actions_range = {range(len(self.global_params.actions_set))}\n"
                f"actions = {self.global_params.actions_set}\n"
                f"reward_in_step = {reward}\n"
                f"cumulated_reward = {cumulated_reward}\n"
                f"done = {done}\n"
            )
            # render_params(
            #     task=self.global_params.task,
            #     v=action[0],  # for continuous actions
            #     w=action[1],  # for continuous actions
            #     episode=episode,
            #     step=step,
            #     state=state,
            #     reward_in_step=reward,
            #     cumulated_reward_in_this_episode=cumulated_reward,
            #     _="--------------------------",
            #     exploration=self.exploration,
            #     # fps=fps,
            #     # best_episode_until_now=best_epoch,
            #     # with_highest_reward=int(current_max_reward),
            # )
        # if not self.all_steps % 10000:
        #     # Update scatter plot
        #     update_scatter_plot(self.ax1, self.all_steps_velocity, self.all_steps_state0, self.all_steps_reward,
        #                         "Velocity", "State[0]", "Reward")
        #     update_scatter_plot(self.ax2, self.all_steps_velocity, self.all_steps_state11, self.all_steps_reward,
        #                         "Velocity", "State[9]", "Reward")
        #     update_scatter_plot(self.ax3, self.all_steps_steer, self.all_steps_state0, self.all_steps_reward, "Steer",
        #                         "State[0]", "Reward")
        #     update_scatter_plot(self.ax4, self.all_steps_steer, self.all_steps_state11, self.all_steps_reward, "Steer",
        #                         "State[9]", "Reward")

        # if not self.all_steps % 100 and self.environment.environment["mode"] != "inference" and not info["bad_perception"]:
        #     self.sac_agent.replay_buffer.add(prev_state, state, action, reward, float(done), [{}])
        #     self.sac_agent.train(100)

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
        callback_list = CallbackList([periodic_save_callback])

        self.sac_agent.learn(total_timesteps=self.params["total_timesteps"],
                              callback=callback_list)

        return state, cumulated_reward, done, info["bad_perception"]

    def main(self):
        hyperparams = combine_attributes(self.algoritmhs_params,
                                         self.environment,
                                         self.global_params)
        self.tensorboard.update_hyperparams(hyperparams)
        # best_epoch_training_time = 0
        # best_epoch = 1

        self.log.logger.info(
            f"\nstates = {self.global_params.states}\n"
            f"states_set = {self.global_params.states_set}\n"
            f"states_len = {len(self.global_params.states_set)}\n"
            f"actions = {self.global_params.actions}\n"
            f"actions set = {self.global_params.actions_set}\n"
            f"actions_len = {len(self.global_params.actions_set)}\n"
            f"actions_range = {range(len(self.global_params.actions_set))}\n"
            f"logs_tensorboard_dir = {self.global_params.logs_tensorboard_dir}\n"
        )

        ## -------------    START TRAINING --------------------
        for episode in tqdm(
                range(1, self.env_params.total_episodes + 1), ascii=True, unit="episodes"
        ):
            done = False
            self.tensorboard.step = episode
            cumulated_reward = 0
            failures = 0
            step = 1
            last_bad_perception = 0

            prev_state, _ = self.env.reset()
            while failures < 3:
                state, cumulated_reward, done, bad_perception = self.one_step_iteration(episode, step, prev_state,
                                                                                        cumulated_reward, done)
                if bad_perception:
                    last_bad_perception = step
                    self.log.logger.info("bad perception in step " + str(step))
                prev_state = state
                step += 1
                if done:
                    failures += 1
                else:
                    failures = 0

                if step >= self.env_params.estimated_steps:
                    break
                # self.env.display_manager.render()
            episode_time = step * self.environment.environment["fixed_delta_seconds"]
            self.log.logger.info("finished in step " + str(step))

            since_last_perception = step - last_bad_perception
            self.save_if_best_epoch(episode, step, cumulated_reward)
            self.calculate_and_report_episode_stats(episode_time, step, cumulated_reward, since_last_perception)
            # self.env.destroy_all_actors()
            # self.env.display_manager.destroy()
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

    def calculate_and_report_episode_stats(self, episode_time, step, cumulated_reward, since_last_perception):
        avg_speed = np.mean(self.episodes_speed)
        max_speed = np.max(self.episodes_speed)
        cum_d_reward = np.sum(self.episodes_d_reward)
        max_reward = np.max(self.episodes_reward)
        steering_std_dev = np.std(self.episodes_steer)
        advanced_meters = avg_speed * episode_time
        bad_perceptions_perc = self.bad_perceptions / step
        completed = 1 if step >= self.env_params.estimated_steps else 0
        self.tensorboard.update_stats(
            std_dev=self.exploration,
            steps_episode=step,
            cum_rewards=cumulated_reward,
            avg_speed=avg_speed,
            max_speed=max_speed,
            cum_d_reward=cum_d_reward,
            max_reward=max_reward,
            steering_std_dev=steering_std_dev,
            advanced_meters=advanced_meters,
            actor_loss=self.actor_loss,
            critic_loss=self.critic_loss,
            bad_perceptions_perc=bad_perceptions_perc,
            since_last_bad_perception=since_last_perception,
            cpu=self.cpu_usages,
            gpu=self.gpu_usages,
            collisions = self.crash,
            completed = completed
        )
        self.tensorboard.update_fps(self.step_fps)
        self.episodes_speed = []
        self.episodes_d_reward = []
        self.episodes_steer = []
        self.episodes_reward = []
        self.step_fps = []
        self.bad_perceptions = 0
        self.crash = 0