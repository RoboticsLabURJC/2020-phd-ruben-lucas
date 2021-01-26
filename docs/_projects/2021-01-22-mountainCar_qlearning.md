---
title: "MountainCar openAI Gym exercise solved using q learning"
excerpt: "Note that q learning is not the best option to solve this exercise"

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

The main goal of this exercise is similar to the last proposed. To dig into the reinforcement learning basics.
In this case, the exercise is not that simple, since some times you will need to get farther from your goal to be able to reach it.
It makes thigs harder in terms of defining a reward function and the hyperparameters configuration to enable the agent to learn the either the optimal or a feasible sequence of steps to accomplish the objective.

In this case we are trying to solve the "mountainCar" problem proposed by openAIGym in which the objective is to teach a car how to reach the peak of a mountain just applying one of the following three actions:
- apply a little force to right
- apply a little force to left
- do nothing

For more details regarding this exercise, refer to the [openAIGym mountainCar website](https://gym.openai.com/envs/MountainCar-v0/)

Since it is still not possible to implement a deep reinforcement learning algorithm due to hardware constraints, the problem has been solved using a q learning algorithm.

<strong>GRAPHS:</strong>

To learn the behaviour of our agent based on what we are implementing it is important to have any kind of metric to measure the performance of each test.
To accomplish that we have three graphs in the upper part of a window that will be prompted each time the agent complete a configured number of maximum attempts to reach the goal.

- The first graph indicates how good the agent is learning a path so the total reward is for each run increases. This graph shows the total reward per run.

- The second graph indicates how many steps were needed in each run to either reach the goal or to fail. In our case, since we configured the environment to be reset when a total of 500 steps were executed, we wiil have an indication of failure when the run displayed reach a number of 500 steps. Otherwise it will indicate that the goal was reached before completing the 500 steps and that means a successfull simulation.

- The third graph indicates how close to the goal was the car in each run/simulation. It gives us an indication of how correlated is the reward and the performance of the simulations.

Additionally, ten matrices will be provided to indicate the qtables evolution, which will give us an idea of the learning evolution of our agent. Note that the matrix represents the following:

- In the horizontal axis, the car position is represented.

- In the vertical axis, the car speed is represented

- blue color means that the optimal action learned by the moment for that position-speed is "do nothing"

- green color means that the optimal action learned by the moment for that position-speed is "step left"

- yellow color means that the optimal action learned by the moment for that position-speed is "step right"


<strong>CONFIGURATION</strong>

Insted of not being really configurable, the code is easily modifiable to try different configurations. In order to do so, comments and constants with descriptive name are added at the beginning of the program.

However, to perform a try the suggestions are to modify the following:

- Hyperparameters

- Maximum number of steps per run

- Maximum number of runs before showing the results

- Reward function

In case of doubt and for seeking inspiration, you can check the "results document" uploaded in [the github repository](https://github.com/RoboticsLabURJC/2020-phd-ruben-lucas/tree/master/openAI_exercises/mountainCar/qlearning/results/)


## Video

{% youtube OifHupQe3KQ %}



## Code

[mountainCar openAIGym exercise solved using qlearning](https://github.com/RoboticsLabURJC/2020-phd-ruben-lucas/tree/master/openAI_exercises/mountainCar/qlearning)

## Results

As referenced in CONFIGURATION section, you can check the results of each qlearning configuration implemented [here](https://github.com/RoboticsLabURJC/2020-phd-ruben-lucas/tree/master/openAI_exercises/mountainCar/qlearning/results/)

As you will see, plenty of different hyperparameters configurations and reward functions were tried based on the interpretation of the ongoing results.
Finally, the best approximation was surprisingly found based on an error designing the reward function which consist of a gap between two levels of the reward function which make the agent learn faster that the optimal reward will be at the top of the mountain.

Additionally, we discover that the agent needs time and exploration to find a sequence of steps since the possibilities are really high and most of them drive to a failure simulation.



<span style="color:green">*Feel free to share a better implementation or discrepancies with our conclussions!! We are humbly open to learn more from more experts!!*</span>

