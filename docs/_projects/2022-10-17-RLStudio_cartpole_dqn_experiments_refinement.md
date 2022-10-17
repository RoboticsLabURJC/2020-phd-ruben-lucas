---
title: "Adding additive random noise and improved monitorization graphs"
excerpt: "Experiments refinement"

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

## INTRODUCTION

After implementing the dqn refined solution in which we can try several complexities of cartpole problem,
we will try to make the agent as stable as possible.

## RANDOM PERTURBATIONS

As we can see in the following graph, when we configure:

- **random_perturbations_level = 0.2** (perturbation in 20% of control iterations)
- **perturbation_intensity_std = 1** 

Our agent nearly always reach the 500 steps goal.

Let's plan a more complex scenario to solve:

### RANDOM PERTURBATIONS MORE INTENSE

- **random_perturbations_level = 0.2** (perturbation in 20% of control iterations)
- **perturbation_intensity_std = 10** 

<p><img src="/2020-phd-ruben-lucas/assets/images/results_images/cartpole/solidityExperiments/base_agent_intensity_std_10.png" alt="map" class="img-responsive" /></p>

#### Training with additive random perturbations

Now, instead of training the agent in ideal conditions, we will set the perturbation frequency to 0.1 and we will
play with the intensity standard deviation of those perturbations.

##### Trained with intensity standard deviation

<p><img src="/2020-phd-ruben-lucas/assets/images/results_images/cartpole/solidityExperiments/intense_trained_agent.png" alt="map" class="img-responsive" /></p>

As we could see, none of them worked better than the agent trained with 0 perturbations noise.
Our conclusion is that it didn't help because the agent did not know if a perturbation was the problem or its action was not the right one so it made the situation worse.

[//]: # (### RANDOM PERTURBATIONS MORE FREQUENT)

[//]: # ()
[//]: # (- **random_perturbations_level = 0.4** &#40;perturbation in 20% of control iterations&#41;)

[//]: # (- **perturbation_intensity_std = 1** )

[//]: # ()
[//]: # (<p><img src="/2020-phd-ruben-lucas/assets/images/results_images/cartpole/solidityExperiments/base_agent_intensity_frq_04.png" alt="map" class="img-responsive" /></p>)

[//]: # ()
[//]: # (#### Training with different random perturbations frequency)

[//]: # ()
[//]: # (<p><img src="/2020-phd-ruben-lucas/assets/images/results_images/cartpole/solidityExperiments/frequence_trained_agent.png" alt="map" class="img-responsive" /></p>)

[//]: # ()
[//]: # (Now we will check if training with random perturbations help the agent to solve them changing the frequency.)


___

### INITIAL STATE

let's see how solid is our agent in terms of initial position pole angle

#### Initial pole angle of 0.25

<p><img src="/2020-phd-ruben-lucas/assets/images/results_images/cartpole/solidityExperiments/init_angle_solidity_ok.png" alt="map" class="img-responsive" /></p>

#### Initial pole angle of 0.3

<p><img src="/2020-phd-ruben-lucas/assets/images/results_images/cartpole/solidityExperiments/init_angle_solidity_nok.png" alt="map" class="img-responsive" /></p>


After the visualizations, we observed that the actions the agent was taken were correct (all the time push to the right to correct the pole angle), but it was not possible
to recover the position within the constraints of the problem (iteration controls frequency and actions force)
So it is not a problem of the agent not being able to recover, it is problem of the problem imposed, which given the
environment conditions and the problem design, makes the solution impossible.
It may work incresing the power of the actions and the granularity of them.

We also saw that adding initial pole angle to the training did not help to train a better agent


___


## CONCLUSIONS

- Training conditions needs to be feasible enough so the agent randomly learns the optimal actions to reach the objective
- If you have a problem or an external agent that makes your scenario more complex, it is useless training in that scenario without giving a
new input that model the problem external agent behavior.

## TO EXPLORE

- If the problem actions are too wide and the optimal actions impossible to learn in a randomly way, we can use iterative learning
to make our agent learn how it has to behave (going gradually from the most basic scenario to the complex one)
- Adaptative learning rate could help. (e.g [torch implementation](https://pytorch.org/docs/stable/optim.html))


