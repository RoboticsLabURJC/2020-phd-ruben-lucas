---
title: "Included random initial positions and random perturbations to cartpole problem"
excerpt: "Cartpole as a guinea pig OpenAIGym simulator related problem"

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

## IMPLEMENTATION

To success with the basic approach ("easy" start position and no perturbations) we implemented **double dqn** making use of
a target neural network that execute the steps and is periodically updated with other neural network that
is being trained from the experience.
This, together wih the replay buffer was not enough to make it stable:
- The success of the training highly depends on how the training starts. If the neural network starts learning, it will behave well, otherwise, it will not learn nothing from the very beginning.
- After some iterations learning (and most of the time reaching the optimal learning) the performance decrease and the neural network forget to solve the problem. This is a common problem called "catastrophic forgetting".

So we tried to solve the problems with the following actions:
- Don't decrease the epsilon (exploration rate) until some configurable iterations. In that way we ensure our agent starts exploring the world without taking wrong learned decissions until we feel it explored enough. 
- We blocked some entries in the replay buffer so we ensure that when the agent learns to solve the problem, we still have some unsuccessful scenarios to train the neural network (and it may not forget the learned policies)

None of them worked as expected, so we had to: 
- Save the model periodically upon configuration so we got some brain if the neural network forgets at some point.
- We set by configuration a parameter "objective_reward" to stop learning when the goal is already reached
- Tune the hyperparameters so we start taking optimal actions not too late and we learn at a velocity and a duration in which our problem solution is nearly always learned.

Once we got the most stable solution we could reach, we started making the problem harder with the two following configuration parameters:
- random_start_level: it indicates how wide is the spectrum of possible initial states which the agent must recover from
- random_perturbations_level: it indicates how often a random perturbation is provoked to the pole.
- perturbation_intensity: it indicates how intense is the provoked perturbation to the pole.

After tons of trials, the agent could not learn when we change the problem to a more difficult one.
However, after learning in the simple scenario, the pole was able to recover from a wide spectrum of situations as we will see in the following section.
For more info ou can check the [simulated anealing algorithm](https://es.wikipedia.org/wiki/Algoritmo_de_recocido_simulado)

## DEMO

That said, our goal is to prove that we are able to train an agent that can recover the pole even trying to boycott it.
All of the following scenarios were run with the agent shown in the following video (same than previous post), that was trained with the following hyperparameters:

- gamma: 0.95
- epsilon_discount: 0.9998
- batch_size: 256

<iframe width="560" height="315" src="https://www.youtube.com/embed/Cb0vg969T54" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## EXPERIMENTS

### WITH RANDOM INITIAL STATES
____
#### RANDOM INITIAL STATES CASE 1 

**random_start_level = 0.2** (all initial states attributes will be between -0.2 and 0.2)

##### Successful iterations

<p><img src="/2020-phd-ruben-lucas/assets/images/results_images/cartpole/case_random_init_02_success" alt="map" class="img-responsive" /></p>


##### Failed iterations

<p><img src="/2020-phd-ruben-lucas/assets/images/results_images/cartpole/case_random_init_02_fails" alt="map" class="img-responsive" /></p>

---

#### RANDOM INITIAL STATES CASE 2 

**random_start_level = 0.3** (all initial states attributes will be between -0.3 and 0.3)

##### Successful iterations

<p><img src="/2020-phd-ruben-lucas/assets/images/results_images/cartpole/case_random_init_03_success" alt="map" class="img-responsive" /></p>

##### Failed iterations

<p><img src="/2020-phd-ruben-lucas/assets/images/results_images/cartpole/case_random_init_03_fails" alt="map" class="img-responsive" /></p>

___

#### RANDOM INITIAL STATES CASE 3 

**random_start_level = 0.4** (all initial states attributes will be between -0.4 and 0.4)

##### Successful iterations

<p><img src="/2020-phd-ruben-lucas/assets/images/results_images/cartpole/case_random_init_04_success" alt="map" class="img-responsive" /></p>

##### Failed iterations

<p><img src="/2020-phd-ruben-lucas/assets/images/results_images/cartpole/case_random_init_04_fails" alt="map" class="img-responsive" /></p>

___

### WITH RANDOM PERTURBATIONS

#### RANDOM PERTURBATIONS CASE 1

**random_perturbations_level = 0.1** (perturbation in 10% of control iterations)

##### Successful iterations

No success (i.e iteration reached 500 steps)

##### Failed iterations

<p><img src="/2020-phd-ruben-lucas/assets/images/results_images/cartpole/case_random_pert_01_fails" alt="map" class="img-responsive" /></p>

___

#### RANDOM PERTURBATIONS CASE 2

**random_perturbations_level = 0.2** (perturbation in 10% of control iterations)

##### Successful iterations

No success (i.e iteration reached 500 steps)

##### Failed iterations

<p><img src="/2020-phd-ruben-lucas/assets/images/results_images/cartpole/case_random_pert_02_fails" alt="map" class="img-responsive" /></p>

___

#### RANDOM PERTURBATIONS CASE 3

**random_perturbations_level = 0.3** (perturbation in 10% of control iterations)

##### Successful iterations

No success (i.e iteration reached 500 steps)

##### Failed iterations

<p><img src="/2020-phd-ruben-lucas/assets/images/results_images/cartpole/case_random_pert_03_fails" alt="map" class="img-responsive" /></p>

___

### WITH DIFFERENT PERTURBATIONS INTENSITIES

#### PERTURBATIONS INTENSITIES CASE 1

random_perturbations_level = 0.1 (perturbation in 10% of control iterations)<br>
**perturbation_intensity = 1.5** (1.5 times more intense than in previous experiments)

##### Successful iterations


##### Failed iterations

___

#### PERTURBATIONS INTENSITIES CASE 2

random_perturbations_level = 0.1 (perturbation in 10% of control iterations)<br>
**perturbation_intensity = 2** (2 times more intense than in previous experiments)

##### Successful iterations


##### Failed iterations

___

## CONCLUSIONS

- It is important to start from an easy problem when training and then, iteratively train this trained brain with different more complex situations
- If you keep training too much time when uising neural networks, you may run into the catastrophic forgetting problem.
Save your model periodically and make sure you train with the correct hyperparameters (mainly learning rate and  replay buffer size) 
and the correct duration
- The dqn algorithm trains better than qlearning and it is able to recover from quite intense perturbations and random initial positions.
The thresholds from with the agent is able to recover in terms of initial position and perturbations is the following:
  - Initial position between the following ranges
    - cart position between -0.4 and 0.4
    - pole angle between -0.3 and 0.3
  - Perturbations with the following configurations and frequencies
    - max intensity of 2 when frequency = 0.1
    - max frequency of 0.2 when intensity = 1 
For more info regarding this thresholds, see the following graph:

METER AQUÍ GRÁFICA HECHA A MANO CON TODOS LOS EXPERIMENTOS (FAILURES Y SUCCESS Y AVG REWARDS DE LOS FAILS)







