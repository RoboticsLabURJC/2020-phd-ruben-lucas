---
title: "Gazebo F1 Follow line ddpg (months 33 and 34)"
excerpt: "Qlearning and DQN f1 follow line gazebo comparison"

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

author: Tony Stark
pinned: false
---

Finished cartpole!!

# Next goal

Now that we achieved to get a good discrete inputs discrete actions agent, lets try
with continuous actions.
For that purpose we will use DDPG.

This is another actor critic algorithm also closely related to Q-learning.
Deep Deterministic Policy Gradient (DDPG) is an algorithm which concurrently learns a Q-function and a policy. 
It uses off-policy data and the Bellman equation to learn the Q-function, and uses the Q-function to learn the policy.

# Progress

- ddpg learning to follow the line
- ddpg learning to follow the line as fast as possible when adding v to reward function

For more  details, see [f1 follow line ddpg blog](https://roboticslaburjc.github.io/2020-phd-ruben-lucas/projects/2023-06-21-F1_follow_line_ddpg/).

# Next steps:

- study why sigmoid activation function was not working so we need tanh
- comparison between ddpg, dqn and qlearning both on training and inferencing
- output training rewards and epsilon over time to enable ploting the training of 3 algorithms together
- further investigate reward functions in literature (e.g misdesign of reward function)
- explore L1 and L2 regularization if experiencing overfiting
- Use batch normalization: Applying batch normalization to the output of the layer before the sigmoid activation function can help stabilize and control the values. This can potentially prevent the inputs to the sigmoid function from becoming too large.
- explore training and retraining with slight different policy to fine tune pretrained agents with different rewards