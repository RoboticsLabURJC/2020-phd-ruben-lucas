---
title: "Gazebo F1 Follow line dqn & qlearning comparison (months 29, 30, 31, 32 and 33)"
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

Next challenge is to revisit the F1 follow line problem in RL-Studio
and accomplish a performance analysis of already implemented algorithms making use of Behavior
metrics.

After that, we will be ready to step out into carla, which will be the tool used for my thesis.

That said, the following is the progress got from the two first months of work in F1 Follow line:

# Progress

- qlearning running and learning with simplified perception and discrete actions
- dqn running and learning (some adaptations were needed to run) with simplified perception and discrete actions
- qlearning v1.2 migrated to Behavior metrics
- dqn v1.2 migrated to Behavior metrics
- Behavior metrics script to automatically compare several algorithms in several circuits
  in Gazebo fixed
- dqn and qlearning trained in RLStudio brains compared and plotted in Behavior metrics

For more  details, see [the qlearning - dqn gazebo f1 follow line comparison blog](https://roboticslaburjc.github.io/2020-phd-ruben-lucas/projects/2023-03-20-F1_follow_line_qlearingn_dqn/)
and [the follow-line dqn refinement](https://roboticslaburjc.github.io/2020-phd-ruben-lucas/projects/2023-04-23-F1_follow_line_dqn_refinement.md/)


# Blockers:

- ddpg is not running in rlStudio and therefore barely learning anything
