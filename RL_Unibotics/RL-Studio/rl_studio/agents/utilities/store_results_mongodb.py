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

yaml_file = '/home/ruben/Desktop/2020-phd-ruben-lucas/RL_Unibotics/RL-Studio/rl_studio/config/config_training_followlane_bs_ddpg_f1_carla.yaml'  # Replace with your YAML file path
reward_filename = '/home/ruben/Desktop/2020-phd-ruben-lucas/RL_Unibotics/RL-Studio/rl_studio/envs/carla/followlane/followlane_carla_sb.py'
reward_method = 'rewards_easy'

tensorboard_logs_dir = os.path.join(
    '/home/ruben/Desktop/2020-phd-ruben-lucas/RL_Unibotics/RL-Studio/rl_studio/logs/training/follow_lane_carla_ddpg_auto_carla_baselines/TensorBoard/DDPG_Actor_conv2d32x64_Critic_conv2d32x64-20241113-100011',
    'events.out.tfevents.1731488411.ruben-Alienware-Aurora-Ryzen-Edition.226189.0.v2')

lesson_learned = '''
Now that we got a good agent and we know:
- algorithm and reward is properly tuned
- brake has a positive impact in performance

It is time to seek for excelence and we observed that agent is often throttling and braking at the same time.
We dont want the brakes to break and we would like a comfortable and smooth driving as a human would perform
So we performed the following adjustments:
- instead of having 3 actions outputed from the network, we have 2:
   - v: If positive we apply throttle, if negative we apply break
   - w: steering angle
  with this we make sure the agent chose between braking or throttling
- Stop rewarding the agent speed over 120km/h. At this speed we are not that happy because we may be fined.
- Stop rewarding speed when brake is applied. With this we encourage stop accelerating instead of braking, which
  is preferable.
- Increase punish when out of lane to ensure we are not preferring not braking at the cost of crashing
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
