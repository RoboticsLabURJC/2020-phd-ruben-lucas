---
title: "MountainBall own environment using OpenAI Gym"
excerpt: "Feel free to review physics and remove libraries dependencies"

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
- Python3.7
- PyQt5
- numpy 1.16.5
- Pandas
- matplotlib
- tensorflow 1.14.0
- gym
- sympy
- stable-baselines

## Manual

The main goal of this exercise is being able to configure owr own environment with its own physics.

The biggest challenge here was to make it easily configurable without the need of adjusting the environment physics.

In the following sections you will find the explanation of the "simulator" and how to play with it to build your own challenges.

<strong>INTRODUCTION:</strong>

With the purpose of easing the building of your own environment, a file called environment.py is provided.
To do so, the following three milestones were achieved:

- Enable the shaping of the world (mountain heigh, floor inclination, stairs, etc.)

- Create a real physics applicable to any environment using the acceleration force over an inclined plane (note that it is an approximation to the reality but not exact. E.g frictional force is not considered)

- Reduce the number of parameters to configure to make it easy

- Make also configurable the goal position, the start position, the car mass, the granularity and the speed of the simulation

- Fix the rendering to work with any world configuration


<strong>CONFIGURATION</strong>

The parameters you can play with are the following

- <strong>car_mass:</strong>
<br>
Weight of the car in kilograms. the heavier it is, less effect will have the force applied with each action at each step

- <strong>heigh function:</strong>
<br>
Function that shape the world. This is the heigh of each point in terms of the x axis (position)

- <strong>lower_position/higher_position:</strong>
<br>
World boundaries in metters

- <strong>max_speed:</strong>
<br>
Speed limit in metters per second. It must be positive and will be translated into a [-max_speed, max_speed] speed boundaries. Considering negative speed when the car is running to the left and positive speed when the car is running to the right

- <strong>seconds_per_step:</strong>
<br>
Simulations speed. Note that higher this variable, less precisse the physics will be, adding some performance inneficiencies. Try to make it as low as possible without harming the training speed too much.

- <strong>force:</strong>
<br>
It is the force applied to the car due to the action executed at each step. Bigger the force configured, easier will be to get the car to the goal.

- <strong>discretization_level:</strong>
<br>
<span style="color:darkgrey">*Note that this parameter has not been propperly tested with other value than 0*</span>
<br>
This is the granularity level of the environment positions. Higher DISCRETIZATION_POS means more precise physics but also slower simulations. It is similar to "seconds_per_step" parameter, but in this case we are indicating how many positions will be considered in our world (note that we are working with a discrete world instead of a continuous one to make the physics behave in a coherent way avoiding some randomness with the calculations)
  -  0 means we are sampling with units
  -  1 means we are sampling with decimes
  -  2 means we are sampling with centesimes
  -  3 means we are sampling with millesimes
  -  ...


- <strong>goal_position:</strong>
<br>
Position to be reached

- <strong>initial_position_low:</strong>
<br>
minimum position where the car will start

- <strong>initial_position_high:</strong>
<br>
maximum position where the car will start

- <strong>initial_speed:</strong>
<br>
Initial speed given to the car

- <strong>gravity:</strong>
<br>
It should not be configurable, but in the case you want to simulate a martian mountain car, you could modify this parameter

Note that with any new environment configuration, the algorithm to solve it may be adjusted, considering new hyperparameters (including the reward function) and a new discretization level if the algorithm samples (or interpolates) the environment states.

Additionally, note that the world will be rescaled so the minimum position is always 0 and the maximum position is the configured maximum position minus the configured minimal position. In this way, the displayed world will go from minimum position = 0 to the (maximum_position - lower_position). it is done to ease the debugging of physics possible deviations from the reality (physics flaws).

You will also find some different worlds (heigh_function) commented to give you an idea of gow you can configure your own scenario.

## Video

<iframe width="560" height="315" src="https://www.youtube.com/embed/2lIvJWTqDUI" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## Code

[configurable environment to create your own mountainCar openAIGym problem worlds](https://github.com/RoboticsLabURJC/2020-phd-ruben-lucas/tree/master/mountain_ball)


<span style="color:green">*Feel free to share a better implementation or discrepancies with our conclussions!! We are humbly open to learn more from more experts!!*</span>
