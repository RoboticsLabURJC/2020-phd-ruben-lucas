---
title: "Qlearning cartpole problem included"
excerpt: "Cartpole as a guinea pig OpenAIGym simulator related problem"

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

<strong>MIGRATION</strong>

You can find the cartpole proposed problem, actions, states and rules to consider the problem solved in [the official openAiGym website](https://www.gymlibrary.ml/environments/classic_control/cart_pole/?highlight=cartpole)
and in [this post](https://www.linkedin.com/pulse/solving-cart-pole-puzzle-ai-gym-using-ga-logistic-regression-kenny-g-/)

The steps to migrate cartpole to rl-studio were the following:
1. Adapt rl-studio project structure and configuration files, so it is adequate to a new simulator (openAiGym)
2. Adapt the qlearning algorithm to accept different type of states (make it more general)
3. Create a new environment to enable different configuration of actions, goals, steps, rewards, etc. So we can adapt the problem at our preference
4. Create inference mode in cartpole problem

you can find all the iterations tested in the [results uploaded](https://github.com/RoboticsLabURJC/2020-phd-ruben-lucas/tree/master/RL_Unibotics/RL-Studio/results/cartpole/qlearning) in the repository.

<strong>GOAL</strong>

The problem is considered solved when the average accumulated reward is at least 195 out of 100 trials.

<strong>DEMO</strong>

As it can be seen in the following video, the goal defined for cartpole_v0 in openAiGym was achieved and surpassed.

The algorithm used in this experiment was qlearning.

The hyperparameters are indicated below:
-      alpha: 0.8
-      gamma: 0.95
-      epsilon discount: 0.99999995


<iframe width="560" height="315" src="https://www.youtube.com/embed/Cb0vg969T54" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

