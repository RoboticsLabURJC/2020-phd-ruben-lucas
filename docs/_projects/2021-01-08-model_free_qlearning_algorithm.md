---
title: "Robot following path to goal with q learning and sarsa"
excerpt: "using q learning and sarsa to learn a path from origin to destination through a mesh board"

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

## Reqs

To execute this program you just need to install the following libraries:
- Python3
- PyQt5
- numpy
- Pandas
- matplotlib

## Manual

The main goal of this kind of exercises is to learn how to develop a simple reinforced learning algorithm to make an agent learn the optimal path to the goal as soon as possible.

<strong>GRAPHS:</strong>

To learn the behaviour of our agent based on what we are implementing it is important to have any kind of metric to measure the performance of each test.
To accomplish that we have two graphs in the upper part of a window that will be prompted each time the agent complete a configured number of maximum attempts to reach the goal.

- The first graph indicates how good the agent is learning a path so the total reward is for each run increases. This graph shows the total reward per run.

- The second graph indicates how many steps were needed in each run to either reach the goal or fail (getting out of the board or stepping a bomb)

<strong>BOARD:</strong>

Nevertheless, to ease the analysis of each try, it is always a good idea to represent the real scenario as clos to reality as possible. For that reason, a board is represented in the down side of the window with the number of times our agent steps each cell of the board before finishing the run. To better have an idea of the final learning of our agent we will display the most recurrents paths followed and the number of occurrences of those paths.

<strong>CONFIGURATION</strong>

Instead of not being really configurable, the code is easily modifiable to try different scenarios and to visualize more or less number of occurrences. In order to do so, comments and constants with descriptive name are added at the beginning of the program.


## Videos

<iframe width="560" height="315" src="https://www.youtube.com/embed/5pHcHyNFSP4" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>



<iframe width="560" height="315" src="https://www.youtube.com/embed/HHlRMhiZWCM" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## Code

[Path learning to reach a goal using qlearning](https://github.com/RoboticsLabURJC/2020-phd-ruben-lucas/tree/master/robot_mesh)

## Results

To be honest, the first prototype I developed was aimed to be q learning but it wasnt. Afterwards, a real q learning was developed and the convergence last much more than in the previous example. It was because in the previous example the q tables were directly filled with the best learn option instead of adjusting little by little the q values until the q tables convergence to the optimal path.

After this experiment I felt it was not enough and Sarsa was implemented to solve the same problem.
Note that Sarsa tries to convergence with as less risk as possible, avoiding paths close to a low reward. For that reason, the map shown in the q learning example never converged using Sarsa. Once the map was a little more clear with wider paths to the goal, Sarsa got to learn a path to the goal even not being the optimal one.
You can find the explanation of this in the best answer to [this question](https://stats.stackexchange.com/questions/326788/when-to-choose-sarsa-vs-q-learning).



<span style="color:green">*Feel free to share a better implementation or discrepancies with our conclussions!! We are humbly open to learn more from more experts!!*</span>

