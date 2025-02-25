import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import tensorflow as tf
import numpy as np

def extract_tensorboard_data(log_dir, key):
    # event_acc = event_accumulator.EventAccumulator(log_dir,
    #                              size_guidance={event_accumulator.SCALARS: 0},)
    # event_acc.Reload()
    tensors = []
    steps = []
    # advanced_meters = event_acc.Tensors('advanced_meters')  # Key name for advanced meters
    # avg_speed = event_acc.Tensors('avg_speed')  # Key name for average speed
    # std_dev = event_acc.Tensors('std_dev')  # Key name for standard deviation
    try:
        for e in tf.compat.v1.train.summary_iterator(log_dir):
            for v in e.summary.value:
                if v.tag == key:
                    tensors.append(v.tensor)
                    steps.append(e.step)
    except Exception as outer_error:
        print(f"Failed to read TensorBoard logs: {outer_error}")
    values = [tf.make_ndarray(t) for t in tensors]
    return steps, values

def smooth_values(values, window_size=5):
    """Smooth the input values using a moving average.

    Args:
        values (list or np.ndarray): Input values to be smoothed.
        window_size (int): The size of the moving window.

    Returns:
        np.ndarray: Smoothed values.
    """
    if window_size < 1:
        raise ValueError("Window size must be at least 1")

    return np.convolve(values, np.ones(window_size) / window_size, mode='valid')


def plot_metrics(file, metric, smooth):
    window_size = 10

    rewards_steps, rewards_values = extract_tensorboard_data(file, metric)

    # Plot Reward evolution
    plt.plot(rewards_steps, rewards_values)
    if smooth:
        smoothed_rewards_values = smooth_values(rewards_values, window_size=window_size)
        plt.plot(rewards_steps[window_size - 1:], smoothed_rewards_values, label=f"Smoothed {metric}", color='orange',
                 linewidth=2)

    plt.xlabel('Steps')
    plt.ylabel(metric)

if __name__ == '__main__':
    file = os.path.join(
        '/home/ruben/Desktop/2020-phd-ruben-lucas/RL_Unibotics/RL-Studio/rl_studio/logs/retraining/follow_lane_carla_ddpg_auto_carla_baselines/TensorBoard/DDPG_Actor_conv2d32x64_Critic_conv2d32x64-20241008-124335',
        'events.out.tfevents.1728384215.ruben-Alienware-Aurora-Ryzen-Edition.749605.0.v2')


    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plot_metrics(file, "cum_rewards", True)

    plt.subplot(2, 2, 2)
    plot_metrics(file, "advanced_meters", True)

    plt.subplot(2, 2, 3)
    plot_metrics(file, "std_dev", False)

    plt.subplot(2, 2, 4)
    plot_metrics(file, "avg_speed", False)

    plt.show()
