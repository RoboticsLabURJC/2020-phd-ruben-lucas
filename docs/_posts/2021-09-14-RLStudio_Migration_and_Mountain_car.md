---
title: "Migrating to new RLStudio and migrating mountain car to RLStudio 0.1.0 (Month 12)"
excerpt: "Run Nacho TFM in new RLStudio and adapt mountain car made problem to work on it"

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

The goals of this month are:
- Migration of the robot_mesh problem to the new RLStudio 0.1.0
- getting the mountain car problem working in RLStudio 0.1.0

In the meanwhile:
- Some ros tutorials has been done to understand how the communication with
the robots and the environment occurs.
- some other modifications were suggested to be included in the Nacho
TFM:
  1. Install manual miss some steps
  2. There is one logs folder not created which gives error when executing the train_qlearn.py script
  3. The lines 46 and 49 of train_qlearn requires a modification

## Lectures

The only formal lecture done this month was for understanding the inference accomplished by Nacho in his TFM:

-  [qlearning inference with python tutorial](https://www.pythonprogramming.net/q-learning-analysis-reinforcement-learning-python-tutorial)


## Lab work

-  [basic ros tutorial](http://wiki.ros.org/turtlesim/Tutorials)

-  [ros deeper practice in gazebo](https://gazebosim.org/tutorials/?tut=ros_comm)
