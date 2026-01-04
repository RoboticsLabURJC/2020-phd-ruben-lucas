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

def find_latest_checkpoint(models_dir):
    """Finds the latest model checkpoint file in a directory."""
    import glob
    import os
    
    # Look for .zip files, which are the saved models from stable-baselines3
    checkpoints = glob.glob(os.path.join(models_dir, '**', '*.zip'), recursive=True)
    if not checkpoints:
        return None
    
    # Return the file with the most recent modification time
    latest_checkpoint = max(checkpoints, key=os.path.getmtime)
    return latest_checkpoint

def launch_training(config_path):
    print(f"[SUPERVISOR] Launching training process with config: {config_path}")
    # Popen is non-blocking, we will use wait() to monitor it
    process = subprocess.Popen(["python3", SCRIPT_PATH, "-f", config_path])
    return process

def orchestrate(experiments):
    for exp in experiments:
        print("=" * 80)
        print(f"[{datetime.now()}] SUPERVISOR: Starting experiment run for algorithm: {exp['settings']['algorithm'].upper()}")
        print("=" * 80)

        recovery_needed = False
        while True:  # This is the supervisor loop
            if not recovery_needed:
                # Initial cleanup of heartbeat file before starting
                if os.path.exists("/tmp/rl_studio_heartbeat.txt"):
                    os.remove("/tmp/rl_studio_heartbeat.txt")

            patched_config_path = patch_config(exp)
            process = launch_training(patched_config_path)
            
            # --- Robust Heartbeat Monitoring Loop ---
            crashed = False
            while process.poll() is None:
                time.sleep(30) # Check every 30 seconds
                heartbeat_path = "/tmp/rl_studio_heartbeat.txt"
                if os.path.exists(heartbeat_path):
                    last_heartbeat = os.path.getmtime(heartbeat_path)
                    if (time.time() - last_heartbeat) > 60: # 60-second timeout
                        print(f"[{datetime.now()}] SUPERVISOR: 🛑 Heartbeat is stale. Worker process has frozen.")
                        process.kill() # Terminate the frozen worker
                        crashed = True
                        break
                else:
                    # If the file doesn't exist yet, just wait for it to be created
                    pass

            # --- Post-mortem Analysis ---
            return_code = process.returncode
            heartbeat_exists = os.path.exists("/tmp/rl_studio_heartbeat.txt")

            # A successful exit is a 0 return code AND the heartbeat file being cleaned up.
            if return_code == 0 and not heartbeat_exists:
                print(f"[{datetime.now()}] SUPERVISOR: ✅ Training process completed successfully.")
                break # Exit the supervisor loop for this experiment
            else:
                # Any other condition is a crash (non-zero code, or unexpected exit where cleanup didn't run)
                crashed = True
            
            if crashed:
                print(f"[{datetime.now()}] SUPERVISOR: 🛑 Training process crashed or was terminated. Initiating recovery.")
                recovery_needed = True
                
                # Give the external CARLA server supervisor time to restart the server
                print("[SUPERVISOR] Waiting 20 seconds for CARLA server to restart...")
                time.sleep(20)

                # Determine the models directory from the config
                models_dir = exp.get("settings", {}).get("models_dir", "./checkpoints")
                latest_model = find_latest_checkpoint(models_dir)

                if latest_model:
                    print(f"[SUPERVISOR] Found latest model for recovery: {latest_model}")
                    exp['settings']['mode'] = 'retraining'
                    if 'retraining' not in exp: exp['retraining'] = {}
                    if exp['settings']['algorithm'] not in exp['retraining']: exp['retraining'][exp['settings']['algorithm']] = {}
                    exp['retraining'][exp['settings']['algorithm']]['retrain_sac_tf_model_name'] = latest_model
                else:
                    print("[SUPERVISOR] Could not find a model checkpoint to resume from. Restarting from scratch.")
                    exp['settings']['mode'] = 'training'
                
                print("[SUPERVISOR] Relaunching training process...")
                # The 'while True' loop will now restart the training



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
