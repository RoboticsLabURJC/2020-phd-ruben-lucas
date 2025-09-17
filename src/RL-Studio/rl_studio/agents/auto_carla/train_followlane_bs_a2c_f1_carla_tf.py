import time
from typing import Callable

import gymnasium as gym
import tensorflow as tf
import torch.nn as nn
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.logger import configure

from rl_studio.agents.f1.loaders import (
    LoadAlgorithmParams,
    LoadEnvParams,
    LoadEnvVariablesA2CCarla,
    LoadGlobalParams,
)
from rl_studio.agents.utilities.extract_reward_function import extract_reward_function
from rl_studio.algorithms.ddpg import (
    ModifiedTensorBoard,
)
from rl_studio.envs.gazebo.gazebo_envs import *

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

            # mlflow.set_experiment("followlane_carla")
            # with mlflow.start_run(nested=True):
            #     mlflow.log_param("model_type", "ppo_bs")
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
            print("Evaluating agent!")
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

            distance_errors.append(info["distance_error"])  # Adjust based on your environment
            velocities.append(info["velocity"])  # Adjust based on your environment
            throttle.append(action[0])  # Assuming action is scalar or index 0 for one component
            steer.append(action[1])  # Assuming action is scalar or index 0 for one component

class TrainerFollowLaneA2CCarla:
    """
    Mode: training
    Task: Follow Line
    Algorithm: A2C
    Agent: F1
    Simulator: Carla
    """

    def __init__(self, config):
        self.actor_loss = 0
        self.critic_loss = 0
        self.algoritmhs_params = LoadAlgorithmParams(config)
        self.env_params = LoadEnvParams(config)
        self.global_params = LoadGlobalParams(config)
        self.environment = LoadEnvVariablesA2CCarla(config)

        logs_dir = f"{self.global_params.logs_tensorboard_dir}/{time.strftime('%Y%m%d-%H%M%S')}"
        self.tensorboard = ModifiedTensorBoard(log_dir=f"{logs_dir}/a2c/overall")
        self.environment.environment["tensorboard"] = self.tensorboard
        self.environment.environment["tensorboard_logs_dir"] = logs_dir

        os.makedirs(f"{self.global_params.models_dir}", exist_ok=True)
        os.makedirs(f"{self.global_params.logs_dir}", exist_ok=True)
        os.makedirs(f"{self.global_params.metrics_data_dir}", exist_ok=True)
        os.makedirs(f"{self.global_params.metrics_graphics_dir}", exist_ok=True)

        agent_log_file = f"{self.global_params.logs_dir}/{time.strftime('%Y%m%d-%H%M%S')}_baselines_{self.global_params.mode}_{self.global_params.task}_A2C_{self.global_params.agent}_{self.global_params.framework}.log"

        # Build Carla environment
        self.env = gym.make(self.env_params.env_name, **self.environment.environment)

        # Init A2C Agent
        if self.environment.environment["mode"] in ["inference", "retraining"]:
            actor_retrained_model = self.environment.environment['retrain_a2c_model_name']
            self.a2c_agent = A2C.load(actor_retrained_model)
            self.a2c_agent.set_env(self.env)
        else:
            self.a2c_agent = A2C(
                "MlpPolicy",
                self.env,
                policy_kwargs=dict(
                    net_arch=dict(
                        pi=[128, 128, 128, 128, 128, 128],
                        vf=[128, 128, 128, 128, 128, 128]
                    ),
                    activation_fn=nn.ReLU
                ),
                learning_rate=linear_schedule(self.environment.environment["critic_lr"]),
                gamma=self.algoritmhs_params.gamma,
                ent_coef=0.05,
                n_steps=5,        # A2C rollout length
                gae_lambda=1.0,   # A2C doesn’t always use GAE, but sb3 allows it
                vf_coef=0.5,
                max_grad_norm=0.5,
                verbose=1,
            )

        print(self.a2c_agent.policy)
        agent_logger = configure(agent_log_file, ["stdout", "csv", "tensorboard"])
        self.a2c_agent.set_logger(agent_logger)

        random.seed(1)
        np.random.seed(1)
        tf.compat.v1.random.set_random_seed(1)

    def main(self):
        # Get hyperparams for tensorboard logging
        hyperparams = self.tensorboard.get_hparams(
            self.algoritmhs_params, self.environment, self.global_params
        )
        all_config = self.tensorboard.combine_attributes(
            self.algoritmhs_params, self.environment, self.global_params
        )
        reward_filename = f"{os.getcwd()}/envs/carla/followlane/followlane_carla_sb_3.py"
        reward_method = 'rewards_easy'
        reward_function = extract_reward_function(reward_filename, reward_method)
        all_config['reward_function'] = reward_function

        self.tensorboard.update_hpparams(hyperparams)
        self.tensorboard.update_hyperparams(all_config)

        # Callbacks (reuse the ones you had for PPO)
        periodic_save_callback = PeriodicSaveCallback(
            env=self.env,
            params={},
            save_path=f"{self.global_params.models_dir}/{time.strftime('%Y%m%d-%H%M%S')}",
            verbose=1
        )
        eval_callback = CustomEvalCallback(
            self.env,
            tensorboard_writer=self.tensorboard,
            best_model_save_path=f"{self.global_params.models_dir}/{time.strftime('%Y%m%d-%H%M%S')}",
            eval_freq=20000,
            deterministic=True,
            render=False
        )
        callback_list = CallbackList([periodic_save_callback, eval_callback])

        # Train or evaluate
        if self.environment.environment["mode"] in ["inference"]:
            self.evaluate_agent(self.env, self.a2c_agent, 10000)
        else:
            self.a2c_agent.learn(total_timesteps=5_000_000, callback=callback_list)

    def evaluate_agent(self, env, agent, num_episodes):
        for episode in range(num_episodes):
            obs, _ = env.reset()
            done = False
            episode_reward = 0
            while not done:
                action, _states = agent.predict(obs, deterministic=True)
                obs, reward, done, done, info = env.step(action)
                episode_reward += reward
            print(f"Episode {episode+1}: Total Reward = {episode_reward}")
