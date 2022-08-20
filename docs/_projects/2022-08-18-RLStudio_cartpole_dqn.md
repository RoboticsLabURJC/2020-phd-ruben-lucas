---
title: "DQN cartpole problem included"
excerpt: "Cartpole optimal solution reached"

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

<strong>Implementation</strong>

You can find the cartpole proposed problem, actions and states in [the official openAiGym website](https://www.gymlibrary.ml/environments/classic_control/cart_pole/?highlight=cartpole)


The steps followed in this project have been the following:
1. Create dqn algorithm to accept the cartpole problem states and provide the expected outputs
2. Create a new dqn cartpole trainer respecting the rl-studio project structure and sw development style
3. Create inference mode respecting the rl-studio project structure and sw development style

you can find all the iterations tested in the [results uploaded](https://github.com/RoboticsLabURJC/2020-phd-ruben-lucas/tree/master/RL_Unibotics/RL-Studio/results/cartpole/dqn) in the repository.

<strong>GOAL</strong>

The problem is considered solved when it always reach 500 steps without failing and located in the center of the image

<strong>DEMO</strong>

As it can be seen in the following video, the goal defined was achieved.
The algorithm used in this experiment was dqn.

The hyperparameters are indicated below:
-      learning rate: 0.001
-      gamma: 0.95
-      epsilon subtraction factor: 0.00025


<iframe width="560" height="315" src="https://www.youtube.com/embed/pzL87ut6unA" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
