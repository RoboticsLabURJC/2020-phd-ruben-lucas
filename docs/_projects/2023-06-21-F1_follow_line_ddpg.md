---
title: "Gazebo F1 Follow line ddpg"
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

Now that we achieved to get a good discrete inputs discrete actions agent, lets try
with continuous actions.
For that purpose we will use DDPG.

This is another actor critic algorithm also closely related to Q-learning.
Deep Deterministic Policy Gradient (DDPG) is an algorithm which concurrently learns a Q-function and a policy. 
It uses off-policy data and the Bellman equation to learn the Q-function, and uses the Q-function to learn the policy.

---

# input 

the input used consists of the distance between 5 line points and the center of the image.
The height of those line points are the following y-axis pixels: [13, 45, 100, 140, 200]


<p><img src="/2020-phd-ruben-lucas/assets/images/results_images/f1-follow-line/gazebo/ddpg/sp5/input.png" alt="map" class="img-responsive" /></p>

# Reward

The different experiments we tried are exposed below in the order we tried:

- Funtion 1:

  - We tried giving much more importance to the position than the velocity using the following reward function and using
  a singularity exponential function:

      ```
      p_reward1&2 = 1/abs(d_p1&2) -> when d_p1&2 < 0.7
      p_reward1&2 = -1            -> when d_p1&2 >= 0.7
      
      p_reward = p_reward1 + p_reward2
      v_reward = v/20
      
      reward * p_reward + v_reward
      ```

  - **Result** It seems to work but the velocity is not being taken into account so the car always go as slow as possible and 
  really close to the line. It converged in 5 hours

* Note that this experiment was done with 2 different neural networks each one being used to infer one velocity (v and w)
and it might impact in the results


- Function 2: 

  - Then we tried to use the sigmoid function and modularize it to reward both position and velocity:

    ```
        def sigmoid_function(start, end, x):
            slope = 10/(end-start-1)
            sigmoid = 1 / (1 + np.exp(-slope * (x - (0.5*(end-start-1)) - (start-1))))
            return sigmoid
    
    p_reward = sigmoid_function(0, 1, abs(d_p1)) +  sigmoid_function(0, 1, abs(d_p1))
    v_reward = sigmoid_function(v_lower_bound, v_upper_bound, abs(v)))
    
    reward = p_reward + (p_reward * v_reward)
    
    ```

  - **Result** With this sigmoid function, we experienced some problems:
    - The slope must be configurable so the curves are as close as possible to the
      end of the sigmoid range, otherwise, the agent won't know the difference between
      states close to the optimal one.
    - The start and end ranges must also be configurable so we make sure that we are using the part
      of the sigmoid function where we are interested in
    - Previous two points make the generalization of the function hard to implement, additionally,
      it plays against the intuition about what is happening without plotting
      or debugging, which make it slow to explore the behavior of the network and implement it
    - Pure sigmoid function is not suitable when we want to increase the reward close to the optimal
      state or the opposite. It is needed some tuning that make the sigmoid senseless
    - If we want the agent to excel, does it make sense to provide a clear difference of all states
      so sigmoid function could cause problems in some situations. The following alternatives are more attractive instead:
      - Exponential Curve
      - Gaussian Curve
      - Piecewise Linear Function

We opted for the Puecewise linear function exposed below:

- Function 3: 
  - Now we are trying just giving a linear reward according to the proximity and reward the velocity 
  in a slightly inferior order of magnitude:

    ```
    
        def reward_proximity(self, state, rewards):
            # sigmoid_pos = self.sigmoid_function(0, 1, state)
            if abs(state) > 0.8:
                return 0, True
            else:
                return 1 - abs(state), False
                
    p_reward1, done1 = self.reward_proximity(state[4], rewards)
    p_reward2, done2 = self.reward_proximity(state[3], rewards)
    p_reward = 5*(p_reward1 + p_reward2)
    done = done1 and done2
    
    v_reward = v/10
    
    reward = p_reward+v_reward
    ```

    - **Result** Just adding the velocity (or multiplying it, what was also tried) seems to be confusing the agent
      and it does not converge (neither seems to be learning nothing) in 10 hours of training



- Function 4: 
  - Now we are trying doing reward shaping and just rewarding the velocity when the car is really close to the line, otherwise, 
    we just reward the position. It will also contribute to to not to shadow the position factor when the car is far from 
    the line.

    ```
    def reward_proximity(self, state, v):
        # sigmoid_pos = self.sigmoid_function(0, 1, state)
        if abs(state) > 0.7:
            return 0, True
        if abs(state) > 0.3:
            return 1 - abs(state), False
        else:
            return (1 - abs(state))+v/10, False
    ```
    
- Other possible future attempts could be:
  - play with learning rates
  - play with more layers/neurons to add complexity to the network
  - Different reward scaling to make sure velocity does not overshadow position
  - Train more time. 20 or 30 hours.
  
# Output

We use continous actions in the following ranges:

    v: [2, 25]

    w: [-2, 2]

# Training/Configuration

Both of them used the same hyperparameters configuration

- gamma: 0.9
- tau: 0.005
- std_dev: 0.2 (note that we had to change how it was applied, since it was summed and we wanted to explore in the same way
  for v and w independently of the actions range, therefore we decided to multiply this factor)
- replay_memory_size: 50_000
- memory_fraction: 0.20
- critic_lr: 0.003
- actor_lr: 0.0015
- buffer_capacity: 100_000
- batch_size: 64
- model architecture: 
  - Actor network
    <p><img src="/2020-phd-ruben-lucas/assets/images/results_images/f1-follow-line/gazebo/ddpg/sp5/model_actor.png" alt="map" class="img-responsive" /></p>
    As a keynote, we observed that sharing a first layer for 2 branches of v and w actions enhance the result since there are implications
    of one action in the other action to achieve the best result
    
  - Critic network
    <p><img src="/2020-phd-ruben-lucas/assets/images/results_images/f1-follow-line/gazebo/ddpg/sp5/model_critic.png" alt="map" class="img-responsive" /></p>

  Note that both networks weights were initialized mainly with negative values because the combinations of relu tend to favor possitive
  values in the output layer.
  Additionally, we needed 2 tanh functions (sigmoid functions were not converging here) at the end of the network to make it work. Possible reasons:
    - Increased Non-linearity: Tanh is a non-linear activation function that allows the network to model complex, non-linear relationships between inputs and outputs. By using two Tanh activation functions in the last layer, you are introducing an additional non-linearity, which can help the network capture more intricate patterns in the data.
    - Enhanced Gradient Flow: The derivative of the Tanh function has a larger range compared to the derivative of a sigmoid function. This can promote better gradient flow during backpropagation, allowing for more efficient learning and convergence. With two Tanh activation functions, the gradient signal can propagate through two layers, potentially leading to more stable and effective training.
    - Increased Model Capacity: By adding an extra layer with Tanh activations, you are increasing the capacity of your neural network. This additional layer introduces more parameters and flexibility to the model, enabling it to learn and represent more complex functions.

Converged in TODO!!!!! hours & 8000 epochs and the fps are between 20 and 30 when disabling monitorization

<p><img src="/2020-phd-ruben-lucas/assets/images/results_images/f1-follow-line/gazebo/ddpg/sp5/training.png" alt="map" class="img-responsive" /></p>

---

# Inference and Behavior metrics comparison

## highly prioritizing position

### DEMO

[//]: # (<iframe width="560" height="315" src="https://www.youtube.com/embed/hKo8Tkktea0" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>)
// TODO

<p><img src="/2020-phd-ruben-lucas/assets/images/results_images/f1-follow-line/gazebo/ddpg/sp5/actions_used.png" alt="map" class="img-responsive" /></p>

### BM Metrics

<p><img src="/2020-phd-ruben-lucas/assets/images/results_images/f1-follow-line/gazebo/ddpg/sp5/BM_continuous.png" alt="map" class="img-responsive" /></p>

