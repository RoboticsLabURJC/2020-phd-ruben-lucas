import os
import time
import yaml
import subprocess
from datetime import datetime, timedelta
import json
import yaml
import argparse

def load_experiments(file_path="experiments.yaml"):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)["experiments"]

SCRIPT_PATH = "rl-studio.py"  # Your main training script name

# Utility to patch the YAML config
def patch_config(config_path, town, carla_client):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    config['carla_environments']['follow_lane']['town'] = town
    config['carla']['carla_client'] = carla_client

    temp_config_path = f"/tmp/patched_{os.path.basename(config_path)}"
    with open(temp_config_path, 'w') as f:
        yaml.dump(config, f)

    return temp_config_path


def launch_training(config_path):
    print(f"[INFO] Launching training with config: {config_path}")
    process = subprocess.Popen(["python3", SCRIPT_PATH, "-f", config_path])
    return process

def orchestrate(experiments):
    for exp in experiments:
        print("=" * 60)
        print(f"{datetime.now()} Starting experiment: {exp['algorithm'].upper()}")
        print(f"{datetime.now()} Config file: {exp['config']}")
        print(f"{datetime.now()} Town: {exp['town']}")
        print(f"{datetime.now()} CARLA client port: {exp['carla_client']}")
        print("="*60)

        patched_config = patch_config(exp['config'], exp['town'], exp['carla_client'])
        process = launch_training(patched_config)

        reward_file = f"/tmp/rlstudio_reward_monitor_{process.pid}.json"
        reward_history = []
        start_time = time.time()

        while process.poll() is None:
            if should_stop_early(start_time, reward_file, reward_history):
                process.terminate()
                print(f"✅ Training {process.pid} stopped early.")
                break
            time.sleep(60)  # Check once per minute

        print(f"[END] {datetime.now()} Training completed or stopped: {exp['algorithm']}\n")

def get_avg_reward_from_file(path):
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return data.get("avg_reward"), data.get("timestamp")
    except Exception:
        return None, None

def should_stop_early(start_time, reward_file, reward_history, patience_hours=6, batch_size=100, trim=5):
    import time

    current_time = time.time()
    avg_reward, timestamp = get_avg_reward_from_file(reward_file)

    if avg_reward is not None:
        reward_history.append((timestamp, avg_reward))

    # Only check after training for at least patience_hours
    if (current_time - start_time) < patience_hours * 3600:
        return False

    # Require at least 4 full batches to start comparison
    required_length = batch_size * 4
    if len(reward_history) < required_length:
        return False

    # Create batches
    batches = [
        [r for _, r in reward_history[i:i + batch_size]]
        for i in range(0, len(reward_history) - batch_size + 1, batch_size)
    ]

    def trim_batch(batch, trim):
        if len(batch) <= 2 * trim:
            return batch  # not enough to trim, return original
        sorted_batch = sorted(batch)
        return sorted_batch[trim:-trim]

    # Prepare first batch
    first_batch = trim_batch(batches[0], trim)
    first_avg = sum(first_batch) / len(first_batch)
    first_max = max(first_batch)

    stagnation_count = 0

    # Check each of the last 3 batches
    for batch in batches[-3:]:
        trimmed = trim_batch(batch, trim)
        batch_avg = sum(trimmed) / len(trimmed)
        batch_max = max(trimmed)
        if batch_avg <= first_avg and batch_max <= first_max:
            stagnation_count += 1

    if stagnation_count == 3:
        print("🛑 Early stopping: reward stagnation over 3 batches of 100 episodes.")
        return True

    return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", default="experiments.yaml", help="Path to experiment config file")
    args = parser.parse_args()

    experiments = load_experiments(args.experiments)
    orchestrate(experiments)
