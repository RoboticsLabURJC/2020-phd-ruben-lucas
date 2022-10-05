---
title: "Trainings and implementation experiments to make cartpole solution more solid"
excerpt: "Experiments"

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
Our starting point for the following experiments is the initial solution presented in [previous post](https://roboticslaburjc.github.io/2020-phd-ruben-lucas/projects/2022-09-21-RLStudio_cartpole_refinement/)

## EXPERIMENTS

### RANDOM PERTURBATIONS

As we could see in previous post, when we configured any combination of random_perturbations_level - perturbations_intensity 
different from "no perturbations" (any of those parameters set to 0), the agent was not able to reach the end of the episode.

Now, instead of training the agent with ideal conditions, we set those parameters to the following values:
- **random_perturbations_level = 0.1** (perturbation in 10% of control iterations)
- **perturbation_intensity = 1** 

After multiple attempts, we got just 1 trained agent and it was not better than the other.
It didn't help because the agent did not know if a perturbation was the problem or its action was not the correct.

That said, we tried to include this previous perturbation to the state that the agent is using to learn.
It may be able to sensorize this perturbation in any way in the real world and it may helps the agent to know how
to behave when the perturbation occur.

Adding this and using the same configuration previously indicated, we got a much better and solid agent for both situations. No
perturbations and perturbations in action (indicating the "sensorized" perturbation of course).

[//]: # (#### RANDOM PERTURBATIONS CASE 1)

[//]: # ()
[//]: # (- **random_perturbations_level = 0.1** &#40;perturbation in 10% of control iterations&#41;)

[//]: # (- **perturbation_intensity = 1** )

[//]: # ()
[//]: # (<p><img src="/2020-phd-ruben-lucas/assets/images/results_images/cartpole/trained_case_1_perturb_intense_1_comparison" alt="map" class="img-responsive" /></p>)

[//]: # ()
[//]: # (##### Successful iterations)

[//]: # ()
[//]: # (<p><img src="/2020-phd-ruben-lucas/assets/images/results_images/cartpole/trained_case_1_perturb_intense_1_success" alt="map" class="img-responsive" /></p>)

[//]: # ()
[//]: # (##### Failed iterations)

[//]: # ()
[//]: # (<p><img src="/2020-phd-ruben-lucas/assets/images/results_images/cartpole/trained_case_1_perturb_intense_1_fails" alt="map" class="img-responsive" /></p>)

[//]: # ()
[//]: # (___)

[//]: # ()
[//]: # (#### RANDOM PERTURBATIONS CASE 2)

[//]: # ()
[//]: # (**random_perturbations_level = 0.2** &#40;perturbation in 10% of control iterations&#41;)

[//]: # (**perturbation_intensity = 1** )

[//]: # ()
[//]: # (<p><img src="/2020-phd-ruben-lucas/assets/images/results_images/cartpole/trained_case_1_random_pert_02_comparison" alt="map" class="img-responsive" /></p>)

[//]: # ()
[//]: # (##### Successful iterations)

[//]: # ()
[//]: # (<p><img src="/2020-phd-ruben-lucas/assets/images/results_images/cartpole/trained_case_1_random_pert_02_success" alt="map" class="img-responsive" /></p>)

[//]: # ()
[//]: # (##### Failed iterations)

[//]: # ()
[//]: # (<p><img src="/2020-phd-ruben-lucas/assets/images/results_images/cartpole/trained_case_1_random_pert_02_fails" alt="map" class="img-responsive" /></p>)

[//]: # ()
[//]: # (___)



## CONCLUSIONS

- logging is important to discover bugs where you assumed a correct behavior
- Training conditions needs to be feasible enough so the agent randomly learns the optimal actions to reach the objective
- Sometimes the agent abruptly learns how to behave. The learning is not linear
- reward function tuning is key
- Sometimes the agent learned to reach the 500 episodes moving little by little to the right
and in episode 500 is almost jumping out of the environment bounds (but reach the 500 steps goal).
Not always reaching the goal means that your agent learned the optimal behavior. I may have learned the minimum possible to reach the goal, so define good the goal is important.
- neural network size and number of layers are capricious. Nor too much nor too little.
- Training with perturbations without letting the agent now that it was perturbed not worked at all.
If you have a problem or an external agent that make your excenario more complex. try to add this agent actions as an input
or your model will suffern to figure it out what it is fighting.
- TOEXPLORE: If the problem actions are too wide and the optimal actions impossible to learn in a randomly way, we can use iterative learning
to make our agent learn how it has to behave (going gradually from the most basic scenario to the complex one)
- TOEXPLORE: adaptative learning rate


