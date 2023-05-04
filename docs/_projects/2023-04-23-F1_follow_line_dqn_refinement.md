---
title: "Gazebo F1 Follow line dqn refinement"
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

The goal of this post is to provide a deeper view on how the environment and dqn have been tuned to reach an optimal policy with:

- F1 follow line in gazebo with simplified perception and discrete actions.

---

## Setup improvement

In the previous post we showed a comparison between dqn and qlearning.
Both of them were working on an environment in wich all the input processing, algorithms training and actions decission process
were happening on real time without pausing the environment physics.

What we learned experimenting with dqn algorithm training is that training process takes too long so the time between one
sensors input until the action is taken is not representative of what will happen in inference time.

That said, we paused the environment before starting training and then unpaused the simulation once training is completed.
Doing that we saw a clear improvement of convergence time (from ~24 hours to ~5 hours).

### input 

We did not change the way the input was obtained but we ensured that input was properly received before starting the 
training and the action selection process.
Sometimes the camera did not receive the image after a reset. Calling the "getImage" twice after a reset fixed the problem.
Moreover, the reset state and the step state were not aligned. Now they are ok

Additionally, 1 point is not enough to determine where the car is. We need more than 1 point to univocal 
tell the agent when the car is above the line so we used 7.

### training

After approximately 1100 episodes, the training was systematically being broken. That happened because
calling model.predict() inside a loop (the episode for action in range(): step/train loop) provoked a known
memory leak. That memory leak can be fixed calling the garbage collector after each call or replacing model.predict()
by model().

The second approach was the chosen for RL-Studio dqn algorithm implementation.

### Reward

After hours of training, the algorithm always converge to an optimal solution.
The problem is that given the reward function configured (stay at least 2 cm close to the line) there were
several optimal policies. Our agent always decided that the best one was going right and left following the line
in a non straight way.
The intuition told us that it was due to that the "go straight" action had a superior linear velocity than "turn to xx" actions,
giving the agent more reaction time upon the next step.

Because of that we decided to fine tune the reward a couple of times.
  - The first attempt consisted of rewarding staying extremely close to the line. It did not work.
  - The second attempt was including in the equation the linear velocity (increasing reward) and the angular velocity (punish).
    with this reward, the car reached our expected goal. Go above the line as fast as possible (and implicitly straight) but it did
    not work either
  - The point used to reward the agent is far from the car. Tried using a closer one but did not work.

We finally returned to the original reward function, using the point upper in the camera image
and just rewarding according to its distance to the center of the image.

### Output

Til now we had 3 actions, straight fast and turn to left or right slightly slower.
We wanted to give more freedom to the agent increasing the number of actions to not to limit the agent behavior
because it does not have enough actions to accomplish the optimal policy.
Note that the faster the car goes, the harder for the agent to learn the optimal policy

We decided to go for the next set of actions:

    Action 0:
        linear velocity push = 2
        angular velocity push = 0

    Action 1:
        linear velocity push = 2
        angular velocity push = 1

    Action 2:
        linear velocity push = 2
        angular velocity push = -1

    Action 3:
        linear velocity push = 1
        angular velocity push = 1

    Action 4:
        linear velocity push = 1
        angular velocity push = -1

    Action 5:
        linear velocity push = 1
        angular velocity push = 0.5

    Action 6:
        linear velocity push = 1
        angular velocity push = -0.5

Note that we also tried adding the width of the line in the image as an input so the agent can perceive
when the line is proximal, but this input confused the agent more than helped.

## Training/Configuration

It converged now in around 5 hours with the next configuration:

  - **alpha**: 0.9
  - **gamma**: 0.9
  - **epsilon**: 0.99
  - **epsilon discount**: 0.993
  - **epsilon min**: 0.05
  - **model architecture**: dense neural network with an input layer with 1 neuron, 
    two intermediate layers of 16 neurons with relu activation function 
    and an output layer with linear activation function. All of this using
    mse as loss function, an Adam optimizer and a learning rate of 0.005.
  - **replay memory size**: 50_000 # How many last steps to keep for model training
  - **min replay memory_size**: 500 # Minimum number of steps in a memory to start training
  - **minibatch size:** 64 # How many steps (samples) to use for training
  - **update target every**: 5  # How many steps until target dqn is updated

It is important to mention also that a periodic evaluation episode with epsilon 0 helped to realize
when the agent converged before the end of the training and identifying good agents before a catastrophical
learning occurs

---

## Inference and Behavior metrics comparison

As you can see in the following video and metrics, dqn better agent was got with 7 inputs

### with 3 inputs

#### performance comparison

<p><img src="/2020-phd-ruben-lucas/assets/images/results_images/f1-follow-line/gazebo/sp3_discrete/comparison_dqn_qlearning.png" alt="map" class="img-responsive" /></p>

#### states histogram

<p><img src="/2020-phd-ruben-lucas/assets/images/results_images/f1-follow-line/gazebo/sp3_discrete/2d_actions.png" alt="map" class="img-responsive" /></p>

<p><img src="/2020-phd-ruben-lucas/assets/images/results_images/f1-follow-line/gazebo/sp3_discrete/3d_actions.png" alt="map" class="img-responsive" /></p>

### with 7 inputs

#### performance comparison

<p><img src="/2020-phd-ruben-lucas/assets/images/results_images/f1-follow-line/gazebo/sp7_discrete/comparison_dqn_qlearning.png" alt="map" class="img-responsive" /></p>

#### states histogram

<p><img src="/2020-phd-ruben-lucas/assets/images/results_images/f1-follow-line/gazebo/sp7_discrete/2d_actions.png" alt="map" class="img-responsive" /></p>

<p><img src="/2020-phd-ruben-lucas/assets/images/results_images/f1-follow-line/gazebo/sp7_discrete/3d_actions.png" alt="map" class="img-responsive" /></p>

### Best agent (7 inputs) video demo

<iframe width="560" height="315" src="https://www.youtube.com/embed/dx8RKuI-Afk" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
___

Note that in here we are comparing the best dqn agent against the best qlearning agent, but this is not a 100% fair
comparison since the number of actions and perception inputs are not exactly the same for both agents 

## Frames per second

### training

**discounting the time the environment is paused during training**

~5.5 fps

### training

~6.5 fps
