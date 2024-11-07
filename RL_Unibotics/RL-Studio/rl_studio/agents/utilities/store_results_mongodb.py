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
    '/home/ruben/Desktop/2020-phd-ruben-lucas/RL_Unibotics/RL-Studio/rl_studio/logs/retraining/follow_lane_carla_ddpg_auto_carla_baselines/TensorBoard/DDPG_Actor_conv2d32x64_Critic_conv2d32x64-20241106-085923',
    'events.out.tfevents.1730879963.ruben-Alienware-Aurora-Ryzen-Edition.1310161.0.v2')

lesson_learned = '''
When using a perfect perception we dont have to worry about many things that were harming the training
- Specific locations where perception is fine. Now can start at any random position, making the training richer.
- Overcomplicating reward. Now there is no need to indicate that many things. It will learn even better with simpler reward. 
- Not perceived image scenario. Now we dont need specific manual actions when image not perceived

However new challenges had to be faced:
- Reward not need to be complicated, but need to be precise, since now the agent learn to exploit it optimally.
  As an example, in previous reward we were not punishing some car deviations and velocity was highly rewarded
  so agent learned to drive fast when perception was totally centered and not missed. Now that perception is
  always perfect, it learned that it was preferable to accelerate A LOT though it provoked some crashes.
  With new reward, all bad consequences are highly punished and balanced with rewards. Additionally, we are
  not that focused on staying in the exact center to avoid some zig-zag the agent was performing to retrieve some
  optimal rewards.
- Now that perception is stable, hyperparameter tuning is more useful. With this we found out that smaller network
  and lower learning rate was achieving a better performance.

Finally and unrelated to perception, we implemented mlflow to record the saved models explicit and implicit metrics to
better catch suboptimal models reached before network get out from the local (or even optimal) minima 
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
