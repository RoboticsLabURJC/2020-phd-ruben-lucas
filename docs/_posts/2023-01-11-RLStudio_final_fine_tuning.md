---
title: "Cartpole problem solution final tuning and conclusions (months 27 and 28)"
excerpt: "Qlearning and DQN cartpole problems included"

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

Now that we solved cartpole problem with discrete actions, we want to explore how it would work some continuous actions
algorithms. However, we want to compare their behaviour in a fair scenario, so we are also iterating the cartpole environment
to behave as close as possible to the real world, improving the physics provided by openAI to not to have a fixed iteration
control time.

Gathered conclussions:


- logging is important to discover bugs where you assumed a correct behavior
- Training conditions needs to be feasible enough so the agent randomly learns the optimal actions to reach the objective
- Sometimes the agent abruptly learns how to behave. The learning is not linear
- reward function tuning is key
- Sometimes the agent learned to reach the 500 episodes moving little by little to the right
and in episode 500 is almost jumping out of the environment bounds (but reach the 500 steps goal).
Not always reaching the goal means that your agent learned the optimal behavior. I may have learned the minimum possible to reach the goal, so define good the goal is important.
- neural network size and number of layers are capricious. Nor too much nor too little. Additionally, more inputs should mean that
more neurons in intermediate layers are needed, but it does not means that the increase must be linear. In our case, when adding
one more action (5 instead of 4), we needed to quatriplicate the number of neurons in the intermediate layer.
- Training conditions needs to be feasible enough so the agent randomly learns the optimal actions to reach the objective
- It is important to sensorize as frequent as possible (and it is critical o sensorize just before taking the action).
Otherwise you may miss some important information or even you will decide with delayed state, what means the agent will surely fail.
-  If you have a problem or an external agent that makes your scenario more complex, it is useless training in that scenario without giving a
new input that model the problem external agent behavior. Training with perturbations letting the agent now that it was perturbed helped a little.
If you have a problem or an external agent that make your scenario more complex. try to add this agent actions as an input
or your model will suffer to figure it out what it is fighting against. BUT make sure you are adding this information because
it is crucial to know that information to reach the goal and tune this input to scale it according to its importance (all the inputs
should be normalized to indicate the agent their importance)
- We showed that when a wide possible actions and states are encountered, a DRL algorithm is more adequate.
- when the problem is simple and we do not have too many inputs-outputs, an continuous actions algorithm works better than a discrete one.
- ppo still being the most sample efficient and best performance agent of these 4.
- DRL discrete actions algorithm behaves quite well considering that they converge faster (TODO to demonstrate)
- the iteration control is quite similar in all algorithms, not being in this simple problem a key factor to choose one or another
- neural networks with 1 hidden layer is more than enough for this. In case the task is not that easy (and solution seems to not to be
  a linear combination of the inputs) we could consider adding
  more layers, but we must note that the training will be longer and more unstable. However, adding neurons to the hidden layer
  can improve the training, being however 128 more than enough for a simple problem like this (according to some lectures it should work even with around 4).
- 1 solidity test could give enough information to know which algorithm is better solving the problem. However, it could happen 
  that unexpected situations or a modification in the problem to solve makes a different algorithm/agent better for that specific
  use case (we showed that DDPG tolerates better adverse initial position while PPO tolerates better random perturbances)
- When training with continuous actions and a robust algorithm, adding perturbations at training time can help
- training and inferencing iteration control frequencies must be controlled in order to decide at a reasonable frequency
- increasing this control iteration frequency can help to improve the agent behavior, but keep also in mind that actions
  must be adjusted accordingly (mostly when using a discrete actions algorithm)
- When working with tensors and GPU, it is important to keep track of all the tensors to free all of them in each experiment before the next one to not to
  incur in a "segmentation fault"
