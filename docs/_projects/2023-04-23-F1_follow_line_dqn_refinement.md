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

# Refinement 1

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
tell the agent when the car is above the line so we used 5.

Regarding the input discretization, we also must make sure that we granularize the proper amount (not more not less) so the
agent is able to assign unique coherent rewards to unique states/actions.

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
learning occurs and not to get the optimal agent. An example of this catastrphic forgeting is ilustrated below:

<p><img src="/2020-phd-ruben-lucas/assets/images/results_images/f1-follow-line/gazebo/sp5_discrete/catastrophic_learning_case.png" alt="map" class="img-responsive" /></p>

---

## Inference and Behavior metrics comparison

As you can see in the following video and metrics, dqn better agent was got with 3 inputs

### with 3 inputs

#### performance comparison

<p><img src="/2020-phd-ruben-lucas/assets/images/results_images/f1-follow-line/gazebo/sp3_discrete/comparison_dqn_qlearning.png" alt="map" class="img-responsive" /></p>

#### states histogram

<p><img src="/2020-phd-ruben-lucas/assets/images/results_images/f1-follow-line/gazebo/sp3_discrete/2d_actions.png" alt="map" class="img-responsive" /></p>

<p><img src="/2020-phd-ruben-lucas/assets/images/results_images/f1-follow-line/gazebo/sp3_discrete/3d_actions.png" alt="map" class="img-responsive" /></p>

### Best agent (3 inputs) video demo

<iframe width="560" height="315" src="https://www.youtube.com/embed/p68O5sixkjs" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>___

Note that in here we are comparing the best dqn agent against the best qlearning agent, but this is not a 100% fair
comparison since the number of actions and perception inputs are not exactly the same for both agents 

## Frames per second

### training

**discounting the time the environment is paused during training**

~20 fps

### training

~25 fps


# Refinement 2:

The goal this week is exploring the following actions:
- Is it better using normalization than discretization in the input?
- Is it better to use the 5 inputs for rewarding than using the 2 more close to the car?
- Try increasing speed to 7 m/s
- Implement and test DDPG

## Training configuration

The training hyperparameters are the same than in refinement 1. 
The difference are the inputs line points:

[13, 25, 60, 110, 160]

The provided actions to the agent:

0: [1, 0]
1: [3, 0]
2: [6, 0]
3: [3, -1]
4: [3, 1]
5: [2, -0.5]
6: [2, 0.5]
7: [1.5, -1]
8: [1.5, 1]
9: [1, -0.5]
10: [1, 0.5]
11: [0.5, -0.5]
12: [0.5, 0.5]

And obviously, the inputs range (between 0 and 1) and the reward function that will be commented in
each section.

##  Is it better using normalization than discretization in the input?

Definitely yes.
The agent not only converge faster (~3 hours), its behavior is optimal without needing the discretization step
which consumes processing time and is error-prone.

Additionally, using normalization we got the agent to use all the available actions as convenience:

<p><img src="/2020-phd-ruben-lucas/assets/images/results_images/f1-follow-line/gazebo/sp5_discrete/actions_normalization.png" alt="map" class="img-responsive" /></p>

Note that now, instead of having discrete states, we have continuous states, releasing much more power to our agent performance.

## Is it better to use the 5 inputs for rewarding than using the 2 more close to the car?

From what I saw, it is not providing any performance improvement. May be for a different scenario
it is beneficial.

The conclusions so far are the following:

- On a straight line, rewarding the farther point does not give any advantage.
- On a curve, rewarding the farther point means rewarding going in the interior of the curve so:
  - It may be confusing the agent in the situation in which it is able to be always adhered to the line
  - It may not giving any advantage at high speed because we are not rewarding being close to the line here, we are rewarding
    going in the interior. Imagine we have a "rotonda", it will go on the exterior antil it has to turn in the oposite direction,
    falling out of the road. In the case of a curve where the car has to go in the interior to not to fall out, it will know it
    just rewarding the two closest points, it is not needed to reward the farther ones (however, it will still use them for knowing what is happening)
  - Moreover, the stop condition dilemma arises again if we are forced to use the 5 points in the reward. Should we stop when we lose all the points on 
    the image or just the closest ones? In our test we decided to be coherent and stop the simulation when all
    the points are too far from the center of the image

### Rewarding all points

#### Reward function

rp1 = c - x1
rp2 = c - x2
rp3 = c - x3
rp4 = c - x4
rp5 = c - x5
rp = rp1 + 0.8*rp2 + 0.6*rp3 + 0.4*rp4 + 0.2*rp5
rv = rp * abs(v)

R =  rp + rv

c is the x coordinate in the center of the image and x the x coordinates of the lines in the image at the configured
heights.
Note that the x1 is the point closest to the car, x2 the following one and so on, being x5 the farthest point in the image.
 
#### DEMO

<iframe width="560" height="315" src="https://www.youtube.com/embed/1rlsaWCMNww" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

#### BM Metrics

<p><img src="/2020-phd-ruben-lucas/assets/images/results_images/f1-follow-line/gazebo/sp5_discrete/bm_normalization_reward_all.png" alt="map" class="img-responsive" /></p>

### Rewarding closest points

#### Reward function

rp1 = c - x1
rp2 = c - x2
rp = rp1 + rp2
rv = rp * abs(v)

R =  rp + rv

c is the x coordinate in the center of the image and x the x coordinates of the lines in the image at the configured
heights.
Note that the x1 is the point closest to the car and x2 the second closest.
Note that not rewarding the farthest points does not mean not providing them to the network.
In order to make the agent work, it is highly recommendable to provide farther points so the 
agent learns to apply a twist before reaching the curve.

#### DEMO

<iframe width="560" height="315" src="https://www.youtube.com/embed/UY2_w15Nb6M" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

#### BM Metrics

<p><img src="/2020-phd-ruben-lucas/assets/images/results_images/f1-follow-line/gazebo/sp5_discrete/bm_normalization_reward_closest.png" alt="map" class="img-responsive" /></p>


