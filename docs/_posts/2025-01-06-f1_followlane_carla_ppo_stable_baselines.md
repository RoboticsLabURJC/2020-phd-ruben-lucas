---
title: "Carla follow lane DDPG vs PPO [December 2nd half]"
excerpt: "PPO good performance"

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

Follow lane ppo is getting more complicated than expected.
We discovered that ppo is not working in the same way as ddpg using same inputs, reward and learning process.

We tried a different setup, reward and learning process explained in the [following slides](https://docs.google.com/presentation/d/1X7quLeBx2wa28VabnF1wQj9ZwCETZf7oDwSXwTVeKok/edit#slide=id.g322b1182fee_0_36)

Additionally, we are already implementing the next stage => adapt speed to other cars in the road