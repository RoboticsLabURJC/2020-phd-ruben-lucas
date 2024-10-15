import math

import numpy as np
import matplotlib.pyplot as plt
import threading


def normalize_range(num, a, b):
    return (num - a) / (b - a)


def linear_function(cross_x, slope, x):
    return cross_x + (slope * x)


def sigmoid_function(start, end, x, slope=10):
    slope = slope / (end - start)
    sigmoid = 1 / (1 + np.exp(-slope * (x - ((start + end) / 2))))
    return sigmoid


def reward_proximity(state):
    if abs(state) > 0.7:
        return 0
    else:
        # return 1 - abs(state)
        # return linear_function(1, -1.4, abs(state))
        return pow(1 - abs(state), 4)
        # return 1 - sigmoid_function(0, 1, abs(state), 5)


def scale_velocity_log(velocity, v_max=30):
    # Logarithmic scaling formula
    return math.log(1 + velocity) / math.log(1 + v_max)

def scale_velocity(velocity, v_max=30, alpha=2):
    """
    Scale velocity exponentially between 0 and 1, where higher velocities result
    in values closer to 1 with exponential-like growth.

    Args:
    - velocity: the velocity to be scaled.
    - v_max: the maximum velocity for scaling reference.
    - alpha: the exponent controlling the curve's growth rate (alpha > 1 for exponential).

    Returns:
    - Scaled velocity value between 0 and 1.
    """
    # Ensure that velocity doesn't exceed v_max to prevent values > 1
    scaled = (velocity / v_max) ** alpha
    return min(scaled, 1)

def rewards_followline_velocity_center(v, pos, range_v):
    p_reward = reward_proximity(pos)
    v_norm = normalize_range(v, range_v[0], range_v[1])
    v_r = v_norm * pow(p_reward, 2)
    beta_pos = 0.7
    reward = (beta_pos * p_reward) + ((1 - beta_pos) * v_r)
    v_punish = reward * (1 - p_reward) * v_norm
    reward = reward - v_punish

    return reward


def rewards_easy(v, pos):
    if v < 1:
        return 0

    if abs(pos) > 0.3:
        return 0

    d_reward = math.pow(1 - abs(pos), 1)
    base_reward = (np.log1p(v) / np.log1p(3)) * math.pow(d_reward, (v/8) + 1)
    v_eff_reward = math.pow(base_reward, 2)

    beta = 0
    # TODO Ver que valores toma la velocity para compensarlo mejor
    function_reward = beta * d_reward + (1 - beta) * v_eff_reward

    return function_reward


#
# def rewards_easy2(v, pos):
#     if v < 1:
#         return 0
#
#     if abs(pos) > 0.3:
#         return 0
#
#     d_reward = math.pow(1 - abs(pos), 2)
#
#     # reward Max = 1 here
#
#
#     # v_reward = v /5
#     # v = scale_velocity(v)
#
#     v_eff_reward = v/5 * math.pow(d_reward, 10)
#     # smooth_factor = 1 / (1 + math.exp(-0.1 * (v - 20)))
#
#     # Blend the two behaviors
#     # v_eff_reward = v * (d_reward * (1 - smooth_factor) + (d_reward ** 3) * smooth_factor)
#
#     beta = 0.7
#     # TODO Ver que valores toma la velocity para compensarlo mejor
#     function_reward = beta * d_reward + (1 - beta) * v_eff_reward
#
#     return function_reward


range_v = [0, 30]

# Define the ranges for v, w, and pos
v_range = np.linspace(range_v[0], range_v[1],  1000)  # Adjust the range as needed
pos_range = np.linspace(-1, 1, 1000)  # Adjust the range as needed

# Create a grid of values for v, w, and pos
V, POS = np.meshgrid(v_range, pos_range)

# Calculate the rewards for each combination of v, w, and pos
rewards = np.empty_like(V)
rewards2 = np.empty_like(V)
for i in range(V.shape[0]):
    for j in range(V.shape[1]):
        # rewards[i, j] = rewards_followline_velocity_center(V[i, j], POS[i, j], range_v)
        rewards[i, j] = rewards_easy(V[i, j], POS[i, j])
        # rewards2[i, j] = rewards_easy2(V[i, j], POS[i, j])

# Create a 3D plot
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(111, projection='3d')

# ax = fig.add_subplot(121, projection='3d')
# ax2 = fig.add_subplot(122, projection='3d')

# Plot the 3D surface
surf = ax.plot_surface(POS, V, rewards, cmap='viridis')
# surf2 = ax2.plot_surface(POS, V, rewards2, cmap='viridis')

# Set labels for the axes
ax.set_xlabel('Center Distance (pos)')
ax.set_ylabel('Linear Velocity (v)')
ax.set_zlabel('Reward1')

# ax2.set_xlabel('Center Distance (pos)')
# ax2.set_ylabel('Linear Velocity (v)')
# ax2.set_zlabel('Reward2')

# Show the color bar
# fig.colorbar(surf, shrink=0.5, aspect=5)
plt.ion()
plt.show()

def input_loop():
    while True:
        try:
            v = float(input("Enter the velocity (v): "))
            pos = float(input("Enter the position (pos): "))

            # Print the result of the rewards_easy function
            print(f"Reward: {rewards_easy(v, pos)}")

        except ValueError:
            print("Please enter valid numbers for velocity and position.")

# Start the input loop in a separate thread
input_thread = threading.Thread(target=input_loop)
input_thread.start()

# Main loop to keep the plot responsive
while True:
    plt.pause(0.1)
