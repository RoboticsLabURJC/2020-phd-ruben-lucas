---
title: "Still migrating mountain car to RL-Studio and year 1 phd formalization (Month 13 - First half)"
excerpt: "Keep working on mountain car (adapted to RL-Studio version) and formalize the phd burocracy"

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

- enroll in the doctoral program for the second year and pay the fees.
- Propperly share the robot mesh implementation and videos both in github (via pull request) and slack (via forum posting)
- write the first prototype of the investigation plan which will be followed during the following years to accomplish the final thesis.
- progress with mountain car problem migration to RL-Studio 0.1.0.

**The enrollment and fees payment was accomplished.**

**The robot mesh implementation in rl-studio was shared with the teammates both in github and slack**

**Regarding the investigation plan**, it was written and can be found in the [URJC platform](https://gestion.urjc.es/RAPI/faces/task-flow-alumno/planesInvestigacion). The overall idea is the following:

   - Year 1: Study of the state of the art for the problem of automatic robot behavior. In-depth study of current deep learning and reinforcement learning techniques to have a clear basis on which to develop the research. Creation of an objective measurement tool to evaluate the developed solutions.  First experiments with constrained circuits and solutions based on classical reinforcement learning for a mobile terrestrial robot.

   - 2nd Year: Implementation in a complex simulated environment of classical reinforcement learning and deep learning algorithms to solve basic practical problems that will allow to gather a number of techniques that will provide the PhD student with the necessary tools to tackle complex problems. It will also contribute to the development of the tool to provide it with the necessary power to facilitate the implementation of these complex problems that will be addressed in the coming years.

   - Year 3: Development of solutions combining deep learning and reinforcement learning to provide robustness to the robot for the previously developed environments and formalization of the solved problems in order to serve as a basis for solving future problems and to clearly illustrate the work done so that it can be used for teaching purposes.

   - 4th Year: Application of what has been previously learned to real environments, with real physical robots and resolution of more complex problems in simulators.

   - Year 5: Continuation of research of complex solutions transferred to real mobile robots after extraction of knowledge in simulation and application of the lessons learned to real environments that have a utility for the end user



That said, this plan is likely to be modified and extended in the future.


**Regarding the mountain car adaptation/migration**, the following actions has been performed:

   - Create new environment simpler and feasibler trying to make the algorithm able to solve it before analysing what has to be done to solve the more compmlex one.

   - Include a statistics display after the 2 hours trainig to enable the algorithm monitoring and the subsequent analysis of what is and what is not working as expected.

   - Some minor fixes to the qlearning implementation (still insufficient)

## Lectures

The following lecture was useful to understand some concepts abut pickle files, used in RL-Studio.

- [pickle documentation](https://networkx.org/documentation/stable/reference/readwrite/gpickle.html)

Additionally, find below a forum were pull requests are explained and was useful to propperly share material with the teammates

-  [how to create a pull request in github]https://opensource.com/article/19/7/create-pull-request-github). Note that in RL-Studio team, this procedure also includes creating an incidence in the original repository and making the pull request from a branch named with this incidence id in the forked/cloned repository.



## Lab work

-  This week, the laboratory work consisted of keeping implementing the adapted/migrated to Rl-Studio mountain-car problem. No consistent results yet.
