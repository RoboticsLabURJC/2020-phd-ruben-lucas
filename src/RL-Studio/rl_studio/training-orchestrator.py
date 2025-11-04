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

def deep_update(target, updates):
    for key, value in updates.items():
        if isinstance(value, dict) and key in target and isinstance(target[key], dict):
            deep_update(target[key], value)
        else:
            target[key] = value

def patch_config(exp):
    config_path = exp['config']
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Merge exp into config recursively (skip 'config' key itself)
    exp_copy = {k: v for k, v in exp.items() if k != 'config'}
    deep_update(config, exp_copy)

    # Optional: overwrite original or write to /tmp
    # Overwrite original:
    # with open(config_path, 'w') as f:
    #     yaml.dump(config, f)

    # Or save as a patched temp config:
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
        print(f"{datetime.now()} Starting experiment: {exp['settings']['algorithm'].upper()}")
        print("="*60)

        patched_config_path = patch_config(exp)
        process = launch_training(patched_config_path)

        reward_file = f"/tmp/rlstudio_reward_monitor_{process.pid}.json"
        reward_history = []
        start_time = time.time()

        while process.poll() is None:
            if should_stop_early(start_time, reward_file, reward_history):
                process.kill()
                print(f"✅ Training {process.pid} stopped early.")
                break
            time.sleep(10)

def get_avg_reward_from_file(path):
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return data.get("avg_reward"), data.get("timestamp")
    except Exception:
        return None, None

def should_stop_early(start_time, reward_file, reward_history, patience_hours=12, batch_size=200):
    import time

    n_batches = 25
    current_time = time.time()
    avg_reward, timestamp = get_avg_reward_from_file(reward_file)

    if avg_reward is not None:
        reward_history.append((timestamp, avg_reward))

    # Only check after training for at least patience_hours
    if (current_time - start_time) < patience_hours * 3600:
        return False

    # Require at least n_batches full batches to start comparison
    required_length = batch_size * n_batches
    if len(reward_history) < required_length:
        return False

    # Create batches
    batches = [
        [r for _, r in reward_history[i:i + batch_size]]
        for i in range(0, len(reward_history) - batch_size + 1, batch_size)
    ]

    first_batch = batches[0]
    first_avg = sum(first_batch) / len(first_batch)
    first_max = max(first_batch)

    stagnation_count = 0

    for batch in batches[-n_batches:]:
        batch_avg = sum(batch) / len(batch)
        batch_max = max(batch)
        # if batch_avg <= first_avg and batch_max <= first_max:
        if batch_avg <= first_avg:
                stagnation_count += 1

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if stagnation_count == n_batches:
        print(f"[{now_str}] 🛑 Early stopping: reward stagnation over {n_batches} batches of {batch_size} episodes.")
        return True
    else:
        print(f"[{now_str}] there were {stagnation_count} batches that not improved from {n_batches} batches ago.")
        if len(reward_history) > batch_size * 2:
            del reward_history[:batch_size]
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", default="experiments.yaml", help="Path to experiment config file")
    args = parser.parse_args()

    experiments = load_experiments(args.experiments)
    orchestrate(experiments)
