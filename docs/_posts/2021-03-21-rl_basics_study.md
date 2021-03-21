---
title: "Study of first section of sutton book (month 7)"
excerpt: "Reinforcement of RL basics to be able to perform good practices when developing"

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

The goal of this month is to fully read the first section of sutton book (8 chapters) and to fully understand
the basics of reincorcement learning. Once we got enough background, it will be easier to follow the good practices
and state-of-the-art of reinforcement learning. Additionally, since more algorithms will be known, more tools will
be at our hands so we can achieve better results for each one of the exercises and projects that will be accomplished.

Additionally, to prove the know-how improvement, the mountain car exercise developed in previous posts will be revisited
and redesigned to achieve better results either modifying the current proposed algorithm or using a more suitable one.


## Lectures

Since the work of this month consisted of learning from the sutton book referenced in [resources section](https://roboticslaburjc.github.io/2020-phd-ruben-lucas/about/), what we are gathering in this section is a summary of headlines with the key lessons learned from this reading. (Note that there were some lessons already learned in previous iterations and note also that a summary of all the lessons learned in each chapter are provided in the book in the last item of every chapter).


<strong><span style="text-decoration: underline">CHAPTER 1</span></strong>  

-  <strong>Subsection 1.5:</strong> Introduction to Reinforcement learning with Tic-Tac-Toe example.



<strong><span style="text-decoration: underline">CHAPTER 2</span></strong>  

-  <strong>Subsection 2.1:</strong> The importance of balancing between exploration and exploitation and some simple methods to do so.

-  <strong>Subsections 2.2, 2.3 and 2.5:</strong> Introduction to stationary vs nonstationary problems.

-  <strong>Subsection 2.3:</strong> The importance of e-greedy selection in a RL policy to increase the average reward taking benefit of exploration.

-  <strong>Subsection 2.6:</strong> The importance of the assigned initial values to each scenario state.

-  <strong>Subsection 2.7:</strong> UBC as an example of deterministically exploration policy.



<strong><span style="text-decoration: underline">CHAPTER 3</span></strong>  

-  <strong>Subsection 3.2:</strong> It is not a good practice to give higher rewards to the agent when the performed action brings it closer to the goal instead of rewarding it for those actions which next state is actually the goal.

-  <strong>Subsections 3.5 and 3.6:</strong> Introduction to the Bellman equation and value and quality concepts (already known but included in the summary due to its importance and the good explanation in the book).

-  <strong>Subsection 3.7:</strong> The Bellman equation is not feasible to be solved in plenty of real situations, so approximation may be needed



<strong><span style="text-decoration: underline">CHAPTER 4</span></strong>  

Concepts of policy, value iteration and policy iteration.
For a better understanding of the difference between value iteration and policy iteration, this [forum discussion](https://stackoverflow.com/questions/37370015/what-is-the-difference-between-value-iteration-and-policy-iteration#:~:text=As%20much%20as%20I%20understand,the%20reward%20of%20that%20policy.) may be useful.



<strong><span style="text-decoration: underline">CHAPTER 5</span></strong>  

-  <strong>Subsections 5.1, 5.2 and 5.3:</strong> Introduction to the Monte Carlo algorihm and the importance of exploring to discover the optimal action when having an (a-priori) unknown model.

-  <strong>Subsections 5.1, 5.2 and 5.3:</strong> The importance of e-greedy and e-soft policies when using the Monte Carlo method.

-  <strong>Subsection 5.5:</strong> Off-policy Vs On-policy methods and introduction to importance sampling (ordinary and weighted).



<strong><span style="text-decoration: underline">CHAPTER 6</span></strong>  

-  <strong>Subsections 6.1, 6.2, 6.3 and 6.4:</strong> TD(0) vs DP and Monte Carlo methods.

-  <strong>Subsections 6.4->:</strong>  Sarsa (and the following takeaway: if St+1 is terminal, then Q(St+1)=0), expected sarsa, Q-learning, and double learning

Note that in this chapter, there are some specially interesting exercises to revisit (random walk, cliff mountain, driving home, windy windworld, etc.) all of them with a suggested algorithm to be applied.



<strong><span style="text-decoration: underline">CHAPTER 7</span></strong>  

-  <strong>Subsection 7.0:</strong> Introduction to the concept of n-step bootstraping.

-  <strong>Subsections 7.2 and 7.3:</strong>  n-step sarsa and n-step expected sarsa (derived from the prediction of TD-n step - intermediate
algorithm between monte-carlo and TD(0)).

-  <strong>Subsection 7.5:</strong> The n-step tree backup algorithm.



<strong><span style="text-decoration: underline">CHAPTER 8</span></strong>  

-  <strong>Subsection 8.1:</strong> Consolidation of model-free Vs model-based and learning/planing paradigms + Introduction to distribution-model Vs sample-model.

-  <strong>Subsections 8.2 and 8.3:</strong> Dyna-q and Dyna-q+: how to apply planning and direct RL learning simultaneously.

-  <strong>Subsection 8.4:</strong> Prioritized sweeping.

-  <strong>Subsection 8.5:</strong> Useful diagram to recap about some learned algorithms and its classification (Figure 8.6: Backup diagrams for all the one-step updates considered in this book).

-  <strong>Subsections 8.6 and 8.7:</strong> Introduction to trajectory sampling and RTDP as a given example.

-  <strong>Subsection 8.8:</strong> Background planning vs decision-time planning.



## Lab work

- [Mountain car exercise revisited](https://roboticslaburjc.github.io/2020-phd-ruben-lucas/projects/2021-03-21-revisited_mountain_car)
