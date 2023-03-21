---
title: "Gazebo F1 Follow line dqn & qlearning comparison"
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

The goal of this post is to provide a comparison of qlearning and dqn algorithms performance in the following problem:

- F1 follow line in gazebo with simplified perception and discrete actions.

---

## Configuration

### input

Both algorithms processed the image to get 1 point of the red line to follow
and the horizontal distance between the center of the frontal image and that retrieved point.

Note that this processing is performed in real time, without pausing the simulation.

### Output

Both algorithms are able to perform one of the following actions during each control iteration:
- action 1:  
  - linear velocity push = 3
  - angular velocity push = 0
  - 
- action 2:  
  - linear velocity push = 2 
  - angular velocity push = 1

- action 3:  
  - linear velocity push = 2
  - angular velocity push = -1

Note that this outputs inferences are performed in real time, without pausing the simulation.

### QLearning

the used hyperparameters are the following:
  - **alpha:** 0.2
  - **epsilon:** 0.95
  - **epsilon_min:** 0.05
  - **gamma:** 0.9


### DQN

the used hyperparameters are the following:
  - **alpha**: 0.8
  - **gamma**: 0.9
  - **epsilon**: 0.99
  - **epsilon discount**: 0.9986
  - **epsilon min**: 0.05
  - **model architecture**: dense neural network with an input layer with 1 neuron, 
    two intermediate layers of 16 neurons with relu activation function 
    and an output layer with linear activation function. All of this using
    mse as loss function, an Adam optimizer and a learning rate of 0.005.
  - **replay memory size**: 50_000 # How many last steps to keep for model training
  - **min replay memory_size**: 1000 # Minimum number of steps in a memory to start training
  - **minibatch size:** 64 # How many steps (samples) to use for training
  - **update target every**: 5  # How many steps until target dqn is updated

---

## Training

Both of them lasted around 24 hours to converge, showing great instability along the way.
However, once the policy was learned, Qlearning was able to keep the right pace, while 
dqn eventually forgot its learnings.

That is not a problem since we periodically save the trained models both in pickle and h5 (which is 
more keras/tf standard) formats

---

## Inference and Behavior metrics comparison

As you can see in the following video and metrics, dqn is slower and sometimes deviates from the center of the line.

<iframe width="560" height="315" src="https://www.youtube.com/embed/SKuoXAekYBw" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

<p><img src="/2020-phd-ruben-lucas/assets/images/results_images/f1-follow-line/gazebo/sp1_discrete/comparison_dqn_qlearning.png" alt="map" class="img-responsive" /></p>
<p><img src="/2020-phd-ruben-lucas/assets/images/results_images/cartpole/solidityExperiments/refinement/refinementOfRefinement/init_pos.png" alt="map" class="img-responsive" /></p>

The deviation can be explained since dqn is not the ideal algorithm ofr such a simple kind of inputs.
It is behaving good, learning that it should stay close to the line, but it is not fully aware of the optimal
policy.

Regarding its slowness, it can be explained since the neural network needs a little more of processing than qlearning,
normalizing the line processed point distance before getting it into the neural network.
That is not pretty much time, but being the actions so small, the impact is tangible when working on real time.
