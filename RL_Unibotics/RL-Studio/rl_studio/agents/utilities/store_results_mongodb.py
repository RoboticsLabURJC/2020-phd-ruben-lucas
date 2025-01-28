import yaml
import pymongo
from pymongo import MongoClient
import datetime
import os
import plot_tensorboard_results
import extract_reward_function
import matplotlib.pyplot as plt
import base64
from io import BytesIO

yaml_file = '/home/ruben/Desktop/2020-phd-ruben-lucas/RL_Unibotics/RL-Studio/rl_studio/config/config_training_followlane_bs_ppo_f1_carla.yaml'
reward_filename = '/home/ruben/Desktop/2020-phd-ruben-lucas/RL_Unibotics/RL-Studio/rl_studio/envs/carla/followlane/followlane_carla_sb_3.py'
reward_method = 'rewards_easy'

tensorboard_logs_dir = os.path.join(
    '/home/ruben/Desktop/2020-phd-ruben-lucas/RL_Unibotics/RL-Studio/rl_studio/logs/training/follow_lane_carla_ppo_continuous_auto_carla_baselines/TensorBoard/PPO_Actor_conv2d32x64_Critic_conv2d32x64-20250106-003205',
    'events.out.tfevents.1736119926.ruben-Alienware-Aurora-Ryzen-Edition.633762.0.v2')

lesson_learned = '''
    After trying and trying with same reward than ddpg, ppo was not able to properly handle curves.
    We realized that the nature of the algorithm may affect the way reward and training is approached.
    Since ppo in sb3 is not optimized to use replay buffer:
        1. It is focused on last states and tends to generalize, so it is struggling to learn more specific scenarios like braking on curves
        2. It works better with short-term rewards while ddpg handle better long-term rewards
    For that reason, we performed the following actions:
        1. Add more waypoints and lidar distance to states
        2. Rewarding throttling instead of speed on centered straights (state[0] close to 0) while keeping the same signal “speed not that important on curves” and “velocities over 120 are bad”. In this way we give a clearer short term signal regarding the taken action
        3. In case we brake on curves, reward is 0 with that configuration, so we applied beta 0.2
        4. Decreasing epsilon and learning rate as training progres to fine tune training
        5. Train with curriculum learning
            1. Just straights and light curves and just w
            2. train also with v
            3. Train harder curves
            4. Train harder initial speeds so it explore also with high speeds and high exploration
        6. Added new metrics and logs to better know what actions agent is learning both on curves and straights
        7. Normalized states and reward to ensure training entropy and other components work as expected
'''

def load_hyperparameters(yaml_file):
    with open(yaml_file, 'r') as file:
        config = yaml.safe_load(file)
    return config


def encode_plots():
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return image_base64


def store_training_results(config, training_results):
    # Set up the MongoDB client (this assumes MongoDB is running locally)
    client = MongoClient('mongodb://localhost:27017/')
    db = client['training_db']  # Database name
    collection = db['training_results']  # Collection name

    del config["states"] # TODO removed for now since it is not mongo compatible
    # Prepare the document to store
    document = {
        'config': config,
        'results': training_results,
        'lessons': lesson_learned,
        'timestamp': datetime.datetime.utcnow()
    }

    # validate_document_keys(document)
    # Insert the document into MongoDB
    result = collection.insert_one(document)
    print(f"Inserted document with ID: {result.inserted_id}")

def validate_document_keys(doc):
    if isinstance(doc, dict):
        for key, value in doc.items():
            if not isinstance(key, str):
                raise ValueError(f"Invalid key: {key}. All keys must be strings.")
            print("validating " + str(value))
            validate_document_keys(value)
    elif isinstance(doc, list):
        for item in doc:
            validate_document_keys(item)


def run_training_and_store_results(yaml_file):
    # Load the YAML configuration
    config = load_hyperparameters(yaml_file)

    reward_function = extract_reward_function.extract_reward_function(reward_filename, reward_method)

    plot_tensorboard_results.plot_metrics(tensorboard_logs_dir, "cum_rewards", True)
    reward_plot = encode_plots()

    plot_tensorboard_results.plot_metrics(tensorboard_logs_dir, "advanced_meters", True)
    advanced_meters_plot = encode_plots()

    plot_tensorboard_results.plot_metrics(tensorboard_logs_dir, "avg_speed", False)
    avg_speed_plot = encode_plots()

    plot_tensorboard_results.plot_metrics(tensorboard_logs_dir, "avg_speed", False)
    std_dev_plot = encode_plots()

    # Simulate training results
    training_results = {
        'reward_function': reward_function,  # Storing reward function as a string
        'plots': {
            'reward_plot': reward_plot,
            'advanced_meters_plot': advanced_meters_plot,
            'avg_speed_plot': avg_speed_plot,
            'std_dev_plot': std_dev_plot
        }
    }

    # Store the results in MongoDB
    store_training_results(config, training_results)


if __name__ == '__main__':
    run_training_and_store_results(yaml_file)
