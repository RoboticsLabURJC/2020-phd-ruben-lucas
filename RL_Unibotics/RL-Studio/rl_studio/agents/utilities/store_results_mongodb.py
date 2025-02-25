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

yaml_file = '/home/ruben/Desktop/2020-phd-ruben-lucas/RL_Unibotics/RL-Studio/rl_studio/config/config_training_followlane_bs_ddpg_f1_carla.yaml'
reward_filename = '/home/ruben/Desktop/2020-phd-ruben-lucas/RL_Unibotics/RL-Studio/rl_studio/envs/carla/followlane/followlane_carla_sb_3.py'
reward_method = 'rewards_easy'

tensorboard_logs_dir = os.path.join(
    '/home/ruben/Desktop/2020-phd-ruben-lucas/RL_Unibotics/RL-Studio/rl_studio/logs/retraining/follow_lane_carla_ddpg_auto_carla_baselines/TensorBoard/DDPG_Actor_conv2d32x64_Critic_conv2d32x64-20250210-100117',
    'events.out.tfevents.1739178077.ruben-Alienware-Aurora-Ryzen-Edition.6413.0.v2')

lesson_learned = '''
    HERE WE HAVE THE STAGE 2 METRICS!!!
    CHECK IT OUT PREVIOUS REPORT FOR 1st STAGE METRICS AND LESSONS LEARNED
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

    plot_tensorboard_results.plot_metrics(tensorboard_logs_dir, "abs_w_no_curves_avg", False)
    abs_w_no_curves_avg_plot = encode_plots()

    plot_tensorboard_results.plot_metrics(tensorboard_logs_dir, "std_dev_v", False)
    std_dev_v_plot = encode_plots()

    plot_tensorboard_results.plot_metrics(tensorboard_logs_dir, "std_dev_w", False)
    std_dev_w_plot = encode_plots()

    plot_tensorboard_results.plot_metrics(tensorboard_logs_dir, "throttle_curves", False)
    throttle_curves_plot = encode_plots()

    plot_tensorboard_results.plot_metrics(tensorboard_logs_dir, "throttle_no_curves", False)
    throttle_no_curves_plot = encode_plots()

    # Simulate training results
    training_results = {
        'reward_function': reward_function,  # Storing reward function as a string
        'plots': {
            'reward_plot': reward_plot,
            'advanced_meters_plot': advanced_meters_plot,
            'avg_speed_plot': avg_speed_plot,
            'abs_w_no_curves_avg_plot': abs_w_no_curves_avg_plot,
            'std_dev_v_plot': std_dev_v_plot,
            'std_dev_w_plot': std_dev_w_plot,
            'throttle_curves_plot': throttle_curves_plot,
            'throttle_no_curves_plot': throttle_no_curves_plot,
        }
    }

    # Store the results in MongoDB
    store_training_results(config, training_results)


if __name__ == '__main__':
    run_training_and_store_results(yaml_file)
