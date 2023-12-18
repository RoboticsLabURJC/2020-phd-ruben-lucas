---
title: "Gazebo F1 Follow line ddpg"
excerpt: "Algorithms comparison"

sidebar:
  nav: "docs"

toc: true
toc_label: "TOC installation"
toc_icon: "cog"


categories:
- your category
tags:
- tag 1
- tag 2
- tag 3
- tag 4

author: Rubén Lucas
pinned: false
---
# Configuration

## QLEARNING

### Hyperparameters

```yaml
alpha: 0.2
epsilon: 0.95
epsilon_min: 0.05      
gamma: 0.9

```

### States

1 input indicating how close is the pixel 10 (starting from the lower part of the image) to he center of the image

### Actions

3 possible actions 

```yaml
0: [ 3, 0 ]
1: [ 2, 1 ]
2: [ 2, -1 ]
```

### Reward function

Let \( \text{center} \) represent a value. The reward is computed based on the following conditions:

1. If \( \text{center} > 0.9 \):
   - \( \text{reward} = 0 \)

2. If \( 0 \leq \text{center} \leq 0.2 \):
   - \( \text{reward} = 100 \)

3. If \( 0.2 < \text{center} \leq 0.4 \):
   - \( \text{reward} = 50 \)

4. Otherwise:
   - \( \text{reward} = 10 \)


## DQN


### Hyperparameters

```yaml
alpha: 0.9
gamma: 0.9
epsilon: 0.2
epsilon_discount: 0.99965
epsilon_min: 0.03
replay_memory_size: 50_000
update_target_every: 5
batch_size: 64

```

### States

10 input indicating how close the following pixels are (starting from the lower part of the image) to he center of the image

[3, 15, 30, 45, 50, 70, 100, 120, 150, 190]

### Actions

9 possible actions 

```yaml
0: [7, 0]
1: [4, 1,5]
2: [4, -1,5]
3: [3, 1]
4: [3, -1]
5: [2, 1]
6: [2, -1]
7: [1, 0.5]
8: [1, -0.5]
```

### Reward function

Let \(P_1\), \(P_2\), \(P_3\) be proximity rewards at different positions, \(v\) be linear velocity, \(w\) be angular velocity, and \(R\) be the overall reward.

1. Compute Proximity Reward:
   - \(P_1, \text{done}_1 = \text{reward\_proximity}(P_{\text{state}_1})\)
   - \(P_2, \text{done}_2 = \text{reward\_proximity}(P_{\text{state}_2})\)
   - \(P_3, \text{done}_3 = \text{reward\_proximity}(P_{\text{state}_3})\)
   - Average proximity reward: \(P = \frac{P_1 + P_2 + P_3}{3}\)

   being reward\_proximity: (1 - abs(state))⁵

2. Normalize Linear Velocity:
   - \(v_{\text{norm}} = \text{normalize\_range}(v, \text{min\_linear\_velocity}, \text{max\_linear\_velocity})\)

3. Compute Velocity Reward:
   - \(v_{\text{reward}} = v_{\text{norm}} \cdot P^2\)

4. Combine Proximity and Velocity Rewards:
   - \(\beta = \text{beta\_1}\)
   - \(R = \beta \cdot P + (1 - \beta) \cdot v_{\text{reward}}\)

5. Penalize Angular Velocity (prevent zig-zag behavior):
   - \(w_{\text{punish}} = \text{normalize\_range}(|w|, 0, \text{max\_angular\_velocity}) \cdot \text{punish\_zig\_zag\_value}\)
   - \(R = R - (R \cdot w_{\text{punish}})\)

6. Penalize Ineffective Acceleration (accelerates convergence):
   - \(v_{\text{punish}} = R \cdot (1 - P) \cdot v_{\text{norm}} \cdot \text{punish\_ineffective\_vel}\)
   - \(R = R - v_{\text{punish}}\)

## DDPG

### Hyperparameters

```yaml
steps_to_decrease: 20000
decrease_substraction: 0.01
decrease_min: 0.03
reward_params:
  punish_zig_zag_value: 1
  punish_ineffective_vel: 0
  beta_1: 0.7
gamma: 0.9
tau: 0.01
std_dev: 0.2
replay_memory_size: 50_000
critic_lr: 0.003
actor_lr: 0.002
buffer_capacity: 100_000
batch_size: 256

```

### States

Same than in DQN algorithm

### Actions

v and w continuous within following ranges:

```yaml
v: [1, 20]
w: [-2, 2]
```

### Reward function

Same than in DQN algorithm

## PPO

### Hyperparameters

```yaml
steps_to_decrease: 20000
decrease_substraction: 0.01
decrease_min: 0.03
reward_params:
  punish_zig_zag_value: 1
  punish_ineffective_vel: 0
  beta_1: 0.5
gamma: 0.9
epsilon: 0.1
std_dev: 0.2
episodes_update: 1_000
replay_memory_size: 50_000
critic_lr: 0.003
actor_lr: 0.002
```

### States

Same than in DQN algorithm

### Actions

v and w continuous within following ranges:

```yaml
v: [1, 10]
w: [-1.5, 1.5]
```

### Reward function

Same than in DQN algorithm

---

# Training

TODO ----- GRAPH WITH TRAINING TIMES. IF NOT THERE, UES LOGS TO GET AN IDEA AND BUILD THEM ARTIFICIALLY
INCLUDE IN THE GRAPH STEPS AND TIMES

---

# Inference

TODO ----- USE BM TO GET THE GRAPHHS

---

# Key lessons 

- Discrete actions algorithms may converge faster, but actions must be chosen wisely and their optimization may be
  tedious or even unfeasible
- It is important to keep track of training and inference frequency so we can train with the same (or lower) frequency
  than inference
- When using continuous actions, it is important to visualize rewards functions so they are as accurate as possible
  Otherwise, training will be too slow or even suboptimal
- It is key to properly monitor on real time the state-action-rewards, loss function, average reward over time, epsilon
  (Tensorboard) fps and received image
- If high speed is needed, we must use farther inputs together with the closest ones which will be used to reward the 
  proximity of the agent to the line (car is not in the horizon, car is in the proximal camera image input)
- Actor learning rate must be slightly below critic learning rate so learning rate can adapt to the actor learnings
- Training and infererence must be totally aligned regarding inputs-outputs and all other neural network hyperparameters
- When building a neural network, start with a small one with not too big learning rates to avoid vanishing gradient and
  overfitting. Then you can iteratively start making it more complex and abrupt
- QLearning and DQN are fine for simpler tasks, but unfeasible for real AD
  - TODO CONCLUSIONS AFTER GRAPHS
- PPO Vs DDPG
  - PPO
    - Pros:
      - Stability: PPO is known for its stability in training, providing more robust convergence properties compared to DDPG.
      - Sample Efficiency: PPO tends to be more sample-efficient than DDPG, requiring fewer samples to achieve comparable or better performance.
      - Noisy Environments: PPO can handle noisy environments better due to its policy optimization approach.
      - Simultaneous Learning: PPO can update its policy using multiple epochs with the same batch of data, allowing for more effective policy updates.
    - Cons:
      - Slower Training: PPO often exhibits slower training compared to DDPG, especially in terms of wall-clock time.
      - Hyperparameter Sensitivity: Fine-tuning PPO can be challenging due to its sensitivity to hyperparameters. 
  - DDPG 
    - Pros: 
      - Learning from Off-Policy Data: DDPG is an off-policy algorithm, enabling learning from previously collected data.
    - Cons:
      - Instability: DDPG can be sensitive to hyperparameters and initial conditions, leading to instability during training.
      - Exploration Challenges: DDPG might face challenges in exploration, especially in high-dimensional and complex environments.
      - Limited to Deterministic Policies: DDPG is designed for deterministic policies, which might be a limitation in scenarios where stochastic policies are more suitable.
  - TODO CONCLUSIONS AFTER GRAPHS

---