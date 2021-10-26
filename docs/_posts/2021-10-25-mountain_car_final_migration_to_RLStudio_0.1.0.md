---
title: "Final implementation of mountain_car basic scenario (Month 13 - Second half)"
excerpt: "Implemented first prototype for mountain car in RL-Studio"

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

The goals of this two weeks were:

- get a productive solution for the mountain-car problem in RL-Studio 0.1.0.
- Allig whith the rest of the JDeRobot team so we cooperate to evolve RL-Studio.


**Regarding the mountain car adaptation/migration**, the following actions has been performed to make the simple environment built in the previous two weeks work:

   - Stable robot was imported an modified and the environment was refined to make the simulation controlable.

   - The statistics are propperly displayed while the simulation is run (in a different thread)

   - After lot of work and tunings to make it learn, the brain is finally learning!!

   - A new mode to execute actions manually was created (pressing 0 and enter the robot goes in one direction and pressiong 2 and enter it goes in the other direction). The name of the scrip is manual_runner.py

Note that, when the new version of RL-Studio evolved by Pedro is updated, this and the robot mesh implementations will be
adapted and a new video to show this and the next two weeks work will be provided in "projects" section of this blog

## Lectures

No new lectures were used this two (actually three) weeks


## Lab work

-  This week, the laboratory work consisted of keeping implementing the adapted/migrated to Rl-Studio mountain-car problem. **Finally successfully!!!!**.
