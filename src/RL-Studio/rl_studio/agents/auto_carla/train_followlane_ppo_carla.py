from datetime import datetime, timedelta
import glob
import time

import gymnasium as gym
import tensorflow as tf
from tqdm import tqdm
from rl_studio.agents.utilities.plot_npy_dataset import plot_rewards
from rl_studio.agents.utilities.push_git_repo import git_add_commit_push

import pynvml
import psutil

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
from rl_studio.algorithms.ddpg import (
    ModifiedTensorBoard,
)

from rl_studio.algorithms.ppo_continuous import PPO

from rl_studio.envs.gazebo.gazebo_envs import *
from rl_studio.envs.carla.carla_env import CarlaEnv

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

def collect_usage():
    cpu_usage = psutil.cpu_percent(interval=None)  # Get CPU usage percentage
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    gpu_info = pynvml.nvmlDeviceGetUtilizationRates(handle)
    gpu_usage = gpu_info.gpu
    return cpu_usage, gpu_usage

class TrainerFollowLanePPOCarla:
    """
    Mode: training
    Task: Follow Line
    Algorithm: DDPG
    Agent: F1
    Simulator: Gazebo
    Framework: TensorFlow
    """

    def __init__(self, config):
        self.avg_speed = 0
        self.advanced_meters = 0
        pynvml.nvmlInit()

        self.algoritmhs_params = LoadAlgorithmParams(config)
        self.env_params = LoadEnvParams(config)
        self.global_params = LoadGlobalParams(config)
        self.environment = LoadEnvVariablesPPOCarla(config)
        self.log_file = f"{self.global_params.logs_dir}/{time.strftime('%Y%m%d-%H%M%S')}_{self.global_params.mode}_{self.global_params.task}_{self.global_params.algorithm}_{self.global_params.agent}_{self.global_params.framework}.log"
        self.log = LoggingHandler(self.log_file)
        self.loss = 0
        self.cpu_usages = 0
        self.gpu_usages = 0
        self.environment.environment["estimated_steps"] = 5000
        self.environment.environment["debug_waypoints"] = self.global_params.debug_waypoints
        self.environment.environment["entropy_factor"] = 0

        self.tensorboard = ModifiedTensorBoard(
            log_dir=f"{self.global_params.logs_tensorboard_dir}/{self.algoritmhs_params.model_name}-{time.strftime('%Y%m%d-%H%M%S')}"
        )
        self.environment.environment["tensorboard"] = self.tensorboard

        os.makedirs(f"{self.global_params.models_dir}", exist_ok=True)
        os.makedirs(f"{self.global_params.logs_dir}", exist_ok=True)
        os.makedirs(f"{self.global_params.metrics_data_dir}", exist_ok=True)
        os.makedirs(f"{self.global_params.metrics_graphics_dir}", exist_ok=True)

        ## Load Carla server
        # CarlaEnv.__init__(self)

        self.env = gym.make(self.env_params.env_name, **self.environment.environment)
        self.all_steps = 0
        self.current_max_reward = 0
        self.best_epoch = 0
        self.episodes_speed = []
        self.episodes_d_reward = []
        self.episodes_v_reward = []
        self.step_fps = []
        self.episodes_steer = []
        self.episodes_reward = []

        std_init = self.algoritmhs_params.std_dev if self.global_params.mode != "inference" else 0.0001
        K_epochs = 5

        # TODO This must come from config states in yaml
        state_size = len(self.environment.environment["x_row"]) + 2
        self.ppo_agent = PPO(state_size, len(self.global_params.actions_set), self.algoritmhs_params.actor_lr,
             self.algoritmhs_params.critic_lr, self.algoritmhs_params.gamma,
             K_epochs, self.algoritmhs_params.epsilon, True, std_init)

        random.seed(2)
        np.random.seed(2)
        tf.compat.v1.random.set_random_seed(2)


    def save_if_best_epoch(self, episode, step, cumulated_reward):
        if self.current_max_reward <= cumulated_reward:
            self.current_max_reward = cumulated_reward
            self.best_epoch = episode

            self.ppo_agent.save(
                f"{self.global_params.models_dir}/IMPROVED/"
                f"{time.strftime('%Y%m%d-%H%M%S')}"
                f"MaxReward-{int(cumulated_reward)}_"
                f"Epoch-{episode}_"
                f"AM-{self.advanced_meters}_"
                f"AS-{self.avg_speed}"
            )

            self.log.logger.info(
                f"\nsaving best lap\n"
                f"in episode = {episode}\n"
                f"current_max_reward = {cumulated_reward}\n"
                f"steps = {step}\n"
            )
        if episode - 100 > self.best_epoch:
            self.best_epoch = episode

            self.ppo_agent.save(
                f"{self.global_params.models_dir}/BATCH/"
                f"{time.strftime('%Y%m%d-%H%M%S')}"
                f"MaxReward-{int(cumulated_reward)}_"
                f"Epoch-{episode}_"
                f"AM-{self.advanced_meters}_"
                f"AS-{self.avg_speed}")
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
        # if bad_perception:
        #     action = [0, 0]
        #     state, reward, done, info = self.env.step(action)
        #     return state, cumulated_reward, done, info["bad_perception"]

        prev_state_fl = prev_state.astype(np.float32)
        tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state_fl), 0)

        action = self.ppo_agent.select_action(tf_prev_state)
        action[0] = action[0]  # TODO scale it propperly (now between -1 and 1)
        action[1] = action[1] - 0.5  # TODO scale it propperly (now between -0.5 and 0.5)

        state, reward, done, done, info = self.env.step(action)

        self.set_stats(info)
        fps = info["fps"]
        self.step_fps.append(fps)
        self.ppo_agent.buffer.rewards.append(reward)
        self.ppo_agent.buffer.is_terminals.append(done)

        # update PPO agent
        if self.all_steps % self.algoritmhs_params.episodes_update == 0 and self.environment.environment["mode"] != "inference" and not info["bad_perception"]:
            self.loss, agent_weights = self.ppo_agent.update()
            self.tensorboard.update_weights(agent_weights, self.all_steps)

        if not self.all_steps % 3000:
            self.cpu_usages, self.gpu_usages = collect_usage()


        if self.all_steps % self.global_params.steps_to_decrease == 0:
            self.ppo_agent.decay_action_std(self.global_params.decrease_substraction,
                                            self.global_params.decrease_min)

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
            render_params(
                task=self.global_params.task,
                v=action[0],  # for continuous actions
                w=action[1],  # for continuous actions
                episode=episode,
                step=step,
                state=state,
                reward_in_step=reward,
                cumulated_reward_in_this_episode=cumulated_reward,
                _="--------------------------",
                exploration=self.ppo_agent.action_std,
                fps=fps,
                # best_episode_until_now=best_epoch,
                # with_highest_reward=int(current_max_reward),
            )
        return state, cumulated_reward, done, info["bad_perception"]

    def main(self):
        hyperparams = combine_attributes(self.algoritmhs_params,
                                         self.environment,
                                         self.global_params)
        self.tensorboard.update_hyperparams(hyperparams)
        # best_epoch_training_time = 0
        # best_epoch = 1

        if self.global_params.mode == "retraining" or self.global_params.mode == "inference":
            checkpoint = self.environment.environment["retrain_ppo_tf_model_name"]
            trained_agent=f"{self.global_params.models_dir}/{checkpoint}"
            self.ppo_agent.load(trained_agent)

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
            self.tensorboard.step = episode
            done = False
            failures = 0
            cumulated_reward = 0
            step = 1

            prev_state, _ = self.env.reset()
            start_time = time.time()
            while failures < 2:
                state, cumulated_reward, done, bad_perception = self.one_step_iteration(episode, step, prev_state, cumulated_reward, done)
                prev_state = state
                step += 1

                if done:
                    failures += 1
                else:
                    failures = 0

                #self.env.display_manager.render()
                if step >= self.env_params.estimated_steps:
                    break
            episode_time = time.time() - start_time

            if self.environment.environment["mode"] != "inference":
                self.save_if_best_epoch(episode, step, cumulated_reward)

            self.calculate_and_report_episode_stats(episode_time, step, cumulated_reward)
            # self.env.destroy_all_actors()
            # self.env.display_manager.destroy()

        # self.env.close()

    def set_stats(self, info):
        self.episodes_speed.append(info["velocity"])
        self.episodes_steer.append(info["steering_angle"])
        self.episodes_d_reward.append(info["d_reward"])
        self.episodes_v_reward.append(info["v_eff_reward"])
        self.episodes_reward.append(info["reward"])

        pass

    def calculate_and_report_episode_stats(self, episode_time, step, cumulated_reward):
        if len(self.episodes_speed) == 0:
            print("zero episodes speed")
            return
        avg_speed = np.mean(self.episodes_speed)
        max_speed = np.max(self.episodes_speed)
        cum_d_reward = np.sum(self.episodes_d_reward)
        cum_v_reward = np.sum(self.episodes_v_reward)
        max_reward = np.max(self.episodes_reward)
        steering_std_dev = np.std(self.episodes_steer)
        advanced_meters = avg_speed * episode_time
        self.tensorboard.update_stats(
            std_dev=self.ppo_agent.action_std,
            steps_episode=step,
            cum_rewards=cumulated_reward,
            avg_speed=avg_speed,
            max_speed=max_speed,
            cum_d_reward=cum_d_reward,
            cum_v_reward=cum_v_reward,
            max_reward=max_reward,
            steering_std_dev=steering_std_dev,
            advanced_meters=advanced_meters,
            actor_loss=self.loss if isinstance(self.loss, int) else self.loss.mean().cpu().detach().numpy(),
            cpu=self.cpu_usages,
            gpu=self.gpu_usages
        )
        self.advanced_meters = advanced_meters
        self.avg_speed = avg_speed
        self.tensorboard.update_fps(self.step_fps)
        self.episodes_speed = []
        self.episodes_d_reward = []
        self.episodes_v_reward = []
        self.episodes_steer = []
        self.step_fps = []
        self.episodes_reward = []

