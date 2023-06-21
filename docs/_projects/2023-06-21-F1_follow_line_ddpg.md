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

# Summary

Now that we achieved to get a good discrete inputs discrete actions agent, lets try
with continuous actions.
For that purpose we will use DDPG.

This is another actor critic algorithm also closely related to Q-learning.
Deep Deterministic Policy Gradient (DDPG) is an algorithm which concurrently learns a Q-function and a policy. 
It uses off-policy data and the Bellman equation to learn the Q-function, and uses the Q-function to learn the policy.

---

# input 

the input used consists of the distance between 5 line points and the center of the image.
The height of those line points are the following y-axis pixels: [13, 45, 100, 140, 200]


<p><img src="/2020-phd-ruben-lucas/assets/images/results_images/f1-follow-line/gazebo/ddpg/sp5/input.png" alt="map" class="img-responsive" /></p>

# Reward

The different experiments we tried are exposed below in the order we tried:

- We tried giving much more importance to the position than the velocity using the following reward function and using
a singularity exponential function:

```
p_reward1&2 = 1/abs(d_p1&2) -> when d_p1&2 < 0.7
p_reward1&2 = -1            -> when d_p1&2 >= 0.7

p_reward = p_reward1 + p_reward2
v_reward = v/20

reward * p_reward + v_reward
```

It seems to work but the velocity is not being taking into account so the car always go as slow as possible and 
really close to the line.

* Note that this experiment was done with 2 different neural networks each one being used to infere one velocity (v and w)
and it might impact in the results


- Then we tried to use the sigmoid function and modularize it to reward both position and velocity:

```
    def sigmoid_function(start, end, x):
        slope = 10/end
        sigmoid = 1 / (1 + np.exp(-slope * (x - (0.5*end) - start)))
        return sigmoid

p_reward = sigmoid_function(0, 1, abs(d_p1)) +  sigmoid_function(0, 1, abs(d_p1))
v_reward = sigmoid_function(v_lower_bound, v_upper_bound, abs(v)))

reward = p_reward + (p_reward * v_reward)

```

- Now we are trying just giving a linear reward according to the proximity and reward the velocity 
in a slightly inferior order of magnitude:

```

    def reward_proximity(self, state, rewards):
        # sigmoid_pos = self.sigmoid_function(0, 1, state)
        if abs(state) > 0.8:
            return 0, True
        else:
            return 1 - abs(state), False
            
p_reward1, done1 = self.reward_proximity(state[4], rewards)
p_reward2, done2 = self.reward_proximity(state[3], rewards)
p_reward = 5*(p_reward1 + p_reward2)
done = done1 and done2

v_reward = v/10

reward = p_reward+v_reward
```

# Output

We use continous actions in the following ranges:

    v: [7, 25]

    w: [-2, 2]

# Training/Configuration

Both of them used the same hyperparameters configuration

- gamma: 0.9
- tau: 0.005
- std_dev: 0.2
- replay_memory_size: 50_000
- memory_fraction: 0.20
- critic_lr: 0.002
- actor_lr: 0.002
- buffer_capacity: 100_000
- batch_size: 64
- model architecture: 

<p><img src="/2020-phd-ruben-lucas/assets/images/results_images/f1-follow-line/gazebo/ddpg/sp5/model.png" alt="map" class="img-responsive" /></p>

Converged in 2:30 hours & 8000 epochs

<p><img src="/2020-phd-ruben-lucas/assets/images/results_images/f1-follow-line/gazebo/ddpg/sp5/training.png" alt="map" class="img-responsive" /></p>

---

# Inference and Behavior metrics comparison

## highly prioritizing position

### DEMO

<iframe width="560" height="315" src="https://www.youtube.com/embed/hKo8Tkktea0" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>


<p><img src="/2020-phd-ruben-lucas/assets/images/results_images/f1-follow-line/gazebo/ddpg/sp5/actions_used.png" alt="map" class="img-responsive" /></p>

### BM Metrics

<p><img src="/2020-phd-ruben-lucas/assets/images/results_images/f1-follow-line/gazebo/dqn/sp5_discrete/BM_7_position.png" alt="map" class="img-responsive" /></p>

