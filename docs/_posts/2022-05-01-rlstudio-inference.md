---
title: "JdeRobot team allignment & inference rlstudio module created (month 17)"
excerpt: "final allignment within the team to progress in same direction and creation of rl-studio inferencer module"

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

author: Tony Stark
pinned: false
---

First of all. We finally got a powerfull personal computer.
This is relevant not just because it will enable us to use more resources-demanding algorithms from now on, but also because the final allignment within the team was possible.
Lets dig into it:

Computer Alienware Aurora R10 requirements:
- AMD Ryzen 9 5900X 3.7GHz
- NVIDIA GeForce RTX 3080 10GB GDDR6X
- 32GB DDR4-3400 RAM1TB Solid State Drive
- 1TB Solid State Drive

Team alignment through the f1 rlstudio problem:
- Due to the lack of a goot computer, the problem was not performing well. The camera image was a little bit lagged so the actions learnt by the f1 and the actions accomplished due to the algorithm decision were not working as expected.
- Now, using the uploaded config.yaml. The robot really learn what it has to learn and complete the lap successfully.

That said, the tasks for this month were:
- Merging into the main branch the robot-mesh and mountain car code and documentation
- Creating an inference module to enable rl-studio to provide a way of using the trained brains in an inference mode out of the rl-studio environment (e.g taking the brain out of the tool and using in real world in inference mode)

So, the outcomes for this months are listed below:
- [robot-mesh problem integrated in rl-studio 1](https://github.com/JdeRobot/RL-Studio/tree/main/rl_studio/agents/robot-mesh)
- [mountain-car problem integrated in rl-studio 1](https://github.com/JdeRobot/RL-Studio/tree/main/rl_studio/agents/mountain-car)
- [inference mode proposed to be integrated in rl-studio 1](https://roboticslaburjc.github.io/2020-phd-ruben-lucas/projects/2022-05-01-RLStudio_inference/)

