---
title: "Gazebo F1 Follow line dqn comparisson"
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

# Summary

The goal of this post is to show different versions of the optimized dqn algorithm
and compare them to the already trained qlearning one.

DQN Agents:
- Balance between following line and maximizing speed highly prioritizing position
- Balance between following line and maximizing speed prioritizing position independently of speed:

---

# input 

the input used consists of the distance between 5 line points and the center of the image.
The height of those line points are the following y-axis pixels: [13, 25, 60, 110, 160]

<p><img src="/2020-phd-ruben-lucas/assets/images/results_images/f1-follow-line/gazebo/dqn/sp5_discrete/input_image.png" alt="map" class="img-responsive" /></p>

# Reward

## highly prioritizing position

For this one we rewarded also the alignment of the car including a second line point in the equation. Additionally, 
the speed is rewarded by adding it to the reward in the following way (velocity does not take a role if car is badly
positioned:
   
 R = (C1 - X1) + (C2 - X2) + ((C1 - X1) + (C2 - X2)) * abs(v)

## position & speed balance

Balance between following line and maximizing speed prioritizing position independently of speed.
For this one, the speed is rewarded by adding it to the reward in the following way:
   
 R = (C1 - X1) + (C2 - X2) + abs(v)


# Output

Different set of actions have been evaluated. Regardless of the number of actions provided to 
the agent. It always uses a maximum of 4 actions.

## highly prioritizing position

<p><img src="/2020-phd-ruben-lucas/assets/images/results_images/f1-follow-line/gazebo/dqn/sp5_discrete/7_actions_balance.png" alt="map" class="img-responsive" /></p>

## position & speed balance

<p><img src="/2020-phd-ruben-lucas/assets/images/results_images/f1-follow-line/gazebo/dqn/sp5_discrete/11_actions_speed_priority.png" alt="map" class="img-responsive" /></p>


# Training/Configuration

Both of them used the same hyperparameters configuration

  - **alpha**: 0.9
  - **gamma**: 0.9
  - **epsilon**: 0.99
  - **epsilon discount**: 0.993
  - **epsilon min**: 0.05
  - **model architecture**: dense neural network with an input layer with 1 neuron, 
    two intermediate layers of 16 neurons with relu activation function 
    and an output layer with linear activation function. All of this using
    mse as loss function, an Adam optimizer and a learning rate of 0.005.
  - **replay memory size**: 50_000 # How many last steps to keep for model training
  - **min replay memory_size**: 500 # Minimum number of steps in a memory to start training
  - **minibatch size:** 64 # How many steps (samples) to use for training
  - **update target every**: 5  # How many steps until target dqn is updated

Note that, for enabling long trainings, I had to replace model.perdict by model and model.fit by model.train_on_batch 
to avoid memory leaks provoking the training to exit with code 137

## highly prioritizing position

Converged in 5:30 hours & 1074 epochs

## position & speed balance

Converged in 4:30 hours & 921 epochs

---

# Inference and Behavior metrics comparison

## highly prioritizing position

### DEMO

<iframe width="560" height="315" src="https://www.youtube.com/embed/hKo8Tkktea0" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

### BM Metrics

<p><img src="/2020-phd-ruben-lucas/assets/images/results_images/f1-follow-line/gazebo/dqn/sp5_discrete/BM_7_position.png" alt="map" class="img-responsive" /></p>

## position & speed balance

### DEMO

<iframe width="560" height="315" src="https://www.youtube.com/embed/76hDcDVbRkQ" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

### BM Metrics

<p><img src="/2020-phd-ruben-lucas/assets/images/results_images/f1-follow-line/gazebo/dqn/sp5_discrete/BM_11_speed.png" alt="map" class="img-responsive" /></p>

## Inreasing speeds in "position & speed balance"

As we can see, in both of them the cars travel slowly to not deviate from the line due to the fps during training were lower than in qlearning.
In inference the fps are bigger so we can increase a little the angular velocities, obtaining the following results:

### new outputs using highly prioritizing position agent

    0: [1, 0]
    1: [3, 0]
    2: [10, 0]
    3: [3, -1]
    4: [3, 1]
    5: [2, -0.5]
    6: [2, 0.5]
    7: [1.5, -1]
    8: [1.5, 1]
    9: [1, -0.5]
    10: [1, 0.5]
    11: [1, -0.5]
    12: [1, 0.5]

### BM metrics

<p><img src="/2020-phd-ruben-lucas/assets/images/results_images/f1-follow-line/gazebo/dqn/sp5_discrete/BM_11_speed_upgrade.png" alt="map" class="img-responsive" /></p>
