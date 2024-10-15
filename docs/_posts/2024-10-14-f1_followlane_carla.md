---
title: "Carla follow lane (2024)"
excerpt: "migration to carla and follow lane problem"

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

# Next goal

Now that we achieved to manage discrete and continuous actions algorithms, let's try to optimize them so 
they reward both the proximity to the center line and the effective speed.

Since the algorithms behavior is clear from previous posts, this one will focus on the following aspects:

- Configuration of each algorithm
  - hyperparameters
  - reward function
  - inputs
  - actions
- Training time of each algorithm until convergence
- Inference behavior metrics of each algorithm
- Key lessons learned and takeaways of each algorithm

[f1 follow line gazebo_comparison](https://roboticslaburjc.github.io/2020-phd-ruben-lucas/projects/2023-12-18-F1_follow_line_comparison/)

# Next steps:

- Moving to Carla, where those algorithms will be used together with other promising ones like SAC
- Start using there the baselines prebuilt algorithms now that we already understand them
- Deep learning specialization course