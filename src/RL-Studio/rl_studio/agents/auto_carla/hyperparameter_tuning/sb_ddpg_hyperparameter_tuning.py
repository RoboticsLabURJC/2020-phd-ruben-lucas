import optuna
from stable_baselines3 import DDPG
from stable_baselines3.common.callbacks import EvalCallback
import gym
import copy
from rl_studio.agents.f1.loaders import (LoadEnvVariablesDDPGCarla,
                                         LoadGlobalParams,
                                         LoadAlgorithmParams,
                                         LoadEnvParams)
import yaml
import numpy as np

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.callbacks import CallbackList

class CustomEvalCallback(EvalCallback):
    def __init__(self, eval_env, eval_noise=None, **kwargs):
        super().__init__(eval_env, **kwargs)
        self.eval_noise = eval_noise          # The noise to use during evaluation (set to None for no noise)

    def _on_step(self) -> bool:
        # Make a deep copy of the original noise to avoid modifying it
        current_noise = copy.deepcopy(self.model.action_noise)

        # Set the evaluation noise (None means no noise)
        self.model.action_noise = self.eval_noise

        # Call the parent class method to handle the actual evaluation
        result = super()._on_step()

        # Restore the original noise after evaluation
        self.model.action_noise = current_noise

        return result

class ExplorationRateCallback(BaseCallback):
    def __init__(self, initial_exploration_rate=0.2, decay_rate=0.01, decay_steps=10000, exploration_min=0.005, verbose=1):
        super(ExplorationRateCallback, self).__init__(verbose)
        self.initial_exploration_rate = initial_exploration_rate
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.current_step = 0
        self.exploration_min = exploration_min
        self.exploration_rate = initial_exploration_rate

    def _on_step(self) -> bool:
        self.current_step += 1
        if self.current_step % self.decay_steps == 1:
            self.exploration_rate = max(self.exploration_min, self.exploration_rate - self.decay_rate)
            # Assuming self.model is a DDPG model
            self.model.action_noise = OrnsteinUhlenbeckActionNoise(
                mean=np.zeros(3),
                sigma=self.exploration_rate * np.ones(3)
            )
            if self.verbose > 0:
                print(f"Step {self.current_step}: Setting exploration rate to {self.exploration_rate}")
        return True

def objective(trial):
    steps_completed = 0  # Counter for completed steps

    with open("/home/ruben/Desktop/RL-Studio/rl_studio/config/config_training_followlane_bs_ddpg_f1_carla.yaml",
              'r') as file:
        config_file = yaml.safe_load(file)

    environment = LoadEnvVariablesDDPGCarla(config_file)
    env_params = LoadEnvParams(config_file)
    exploration_params = LoadGlobalParams(config_file)
    algorithm_params = LoadAlgorithmParams(config_file)

    environment.environment["debug_waypoints"] = False
    environment.environment["estimated_steps"] = 5000
    environment.environment["entropy_factor"] = 0
    environment.environment["punish_zig_zag_value"] = trial.suggest_float('punish_zig_zag_value', 0, 5)

    env = gym.make(env_params.env_name, **environment.environment)
    eval_env = gym.make(env_params.env_name, **environment.environment)

    # Define the search space for hyperparameters
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
    buffer_size = trial.suggest_int('buffer_size', 50000, 1000000)
    batch_size = trial.suggest_int('batch_size', 64, 512)
    tau = trial.suggest_float('tau', 0.001, 0.01)
    gamma = trial.suggest_float('gamma', 0.8, 0.99)

    # Policy network architecture
    net_arch_pi = trial.suggest_categorical('net_arch_pi', [[32, 32, 32], [64, 64, 64], [128, 128, 128], [128, 128, 128, 128]])
    net_arch_qf = trial.suggest_categorical('net_arch_qf', [[32, 32, 32], [64, 64, 64], [128, 128, 128], [128, 128, 128, 128]])

    model = DDPG(
        "MlpPolicy",
        env,
        policy_kwargs=dict(net_arch=dict(pi=net_arch_pi, qf=net_arch_qf)),
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        batch_size=batch_size,
        tau=tau,
        gamma=gamma,
        verbose=1
    )

    # Callbacks
    exploration_rate_callback = ExplorationRateCallback(initial_exploration_rate= algorithm_params.std_dev,
                                                        decay_rate= exploration_params.decrease_substraction,
                                                        decay_steps=exploration_params.steps_to_decrease,
                                                        exploration_min=exploration_params.decrease_min,
                                                        verbose=1)

    eval_callback = CustomEvalCallback(eval_env, best_model_save_path='./logs/hyp_tuning/model',
                                 log_path='./logs/hyp_tuning', eval_freq=1000)
    callback_list = CallbackList([exploration_rate_callback, eval_callback])

    # Total number of timesteps for training
    total_timesteps = 5000
    timesteps_per_iteration = 500


    # Loop through to train the model in increments
    while steps_completed < total_timesteps:
        try:
            model.learn(total_timesteps=timesteps_per_iteration, callback=callback_list)
            steps_completed += timesteps_per_iteration

            # Report intermediate result to Optuna
            trial.report(eval_callback.best_mean_reward, steps_completed)

            # Check if the trial should be pruned
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        except Exception as e:
            print(f"Exception occurred at step {steps_completed}: {e}. Continuing the trial.")

    # Clean up the environments to destroy the vehicles
    if env is not None:
        env.close()
    if eval_env is not None:
        eval_env.close()
    if model is not None:
        del model  # Release memory

    # Return the best mean reward after training
    return eval_callback.best_mean_reward


# Create the study
study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())

# Optimize with a try-except block to catch failures
study.optimize(objective, n_trials=8)

# Set logging level
optuna.logging.set_verbosity(optuna.logging.INFO)

# Visualize the optimization history and parameter importance
optuna.visualization.plot_optimization_history(study)
optuna.visualization.plot_param_importances(study)

# Retrieve the best trial
print("Best trial:")
trial = study.best_trial
print(f"  Value: {trial.value}")
print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

# Retrieve and print all trials' results
print("\nCompleted trials so far:")
for trial in study.trials:
    print(f"Trial {trial.number}:")
    print(f"  Value: {trial.value}")
    print(f"  Params: {trial.params}")
    print(f"  State: {trial.state}")  # Shows whether it succeeded, failed, or was pruned
