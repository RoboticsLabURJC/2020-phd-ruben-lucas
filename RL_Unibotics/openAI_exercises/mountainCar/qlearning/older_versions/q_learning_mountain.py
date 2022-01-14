import sys

import random
import gym
import time
import numpy as np
from collections import deque

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from PyQt5.QtWidgets import QVBoxLayout
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QLabel
from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

ENV_NAME = "MountainCar-v0"

GAMMA = 0.98
LEARNING_RATE = 0.2

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.02
EXPLORATION_DECAY = 0.9995


class QSolver:

    def __init__(self, env):
        self.exploration_rate = EXPLORATION_MAX
        self.action_space = env.action_space.n
        print(env.observation_space.low)
        print(env.observation_space.high)
        # Determine size of discretized state space
        num_states = (env.observation_space.high - env.observation_space.low)*\
                        np.array([40, 200])
        num_states = np.round(num_states, 0).astype(int) + 1
        print(num_states)

        # Initialize Q table
        self.q_values = np.random.uniform(low = -1, high = 1,
                              size = (num_states[0], num_states[1],
                                      env.action_space.n))

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)
        action = np.argmax(self.q_values[state[0], state[1]])
        return action

    def experience_replay(self, done, state_adj, action, reward, state2_adj, state_next, run):

        #Allow for terminal states
        if done and state_next[0] >= 0.5:
            self.q_values[state_adj[0], state_adj[1], action] = reward
        # Adjust Q value for current state
        else:
            delta = LEARNING_RATE*(reward +
                             GAMMA*np.max(self.q_values[state2_adj[0],
                                               state2_adj[1]]) -
                             self.q_values[state_adj[0], state_adj[1],action])
            self.q_values[state_adj[0], state_adj[1],action] += delta

        #q_update = reward
        #if not terminal:
        #    q_update = (reward + GAMMA * np.amax(self.q_values[state_next]))
        #q_values = self.q_values[state]
        #q_values[action] = q_update
        #print ("qvalues in state " + str(state_adj[0]) + " " + str(state_adj[1]))
        #print(self.q_values[state_adj[0], state_adj[1]])
        #print ("state_next " + str(state_next[0]) + " " + str(state_next[1]))
        #print ("reward " + str(reward))
        if run>50:
            self.exploration_rate *= EXPLORATION_DECAY
            self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)


def get_reward(state):
    if state[0] >= 0.5:
        print("Car has reached the goal")
        return 500
#    if state[0] > -0.4:
#        return (1+state[0])**2
#    if state[0] <-0.49 or state[0]>-0.51:
#        return -0.2
#    return 0
    if state[0]<-0.7:
        return ((state[0]+0.7))
    if state[0]>-0.7 and state[0]<-0.3:
        return 9*(state[0]+0.3)
    if state[0]>-0.2:
        return (9*(state[0]+0.3))**2

    return 0

def plot_rewards_per_run(axes, dataSet):
    rewards_graph=pd.DataFrame(dataSet)
    ax=rewards_graph.plot(ax=axes[0], title="rewards per run");
    ax.set_xlabel("runs")
    ax.set_ylabel("rewards")
    ax.legend().set_visible(False)

def plot_steps_per_run(axes, dataSet):
    rewards_graph=pd.DataFrame(dataSet)
    ax=rewards_graph.plot(ax=axes[1], title="steps per run");
    ax.set_xlabel("run")
    ax.set_ylabel("steps")
    ax.legend().set_visible(False)

def get_stats_figure(steps, rewards):
    fig, axes = plt.subplots(nrows=1, ncols=2)
    fig.set_size_inches(12, 4)
    plot_steps_per_run(axes, steps)
    plot_rewards_per_run(axes, rewards)
    return fig

def show_in_same_window(stats):
    app = QApplication.instance()
    ui = QWidget()
    vbox = QGridLayout()
    vbox.addWidget(stats.canvas, 1, 1, 1 ,1)
    ui.setLayout(vbox)
    ui.show()
    sys.exit(app.exec_())

def mountain():
    env = gym.make(ENV_NAME)
    q_solver = QSolver(env)
    run = 0
    rewards=[]
    steps=[]
    while True:
        run += 1
        if run>1000:
            stats_figure=get_stats_figure(steps, rewards)
            show_in_same_window(stats_figure)
            exit()
        state = env.reset()

        # Discretize state
        state_adj = (state - env.observation_space.low)*np.array([40, 200])
        state_adj = np.round(state_adj, 0).astype(int)

        step = 0
        tot_reward=0
        while True:
            step += 1
            env.render()
            action = q_solver.act(state_adj)
            #time.sleep(1)
            state_next, reward, done, info = env.step(action)
            #print (state_next)
            updated_reward=get_reward(state_next)

            # Discretize state2
            state2_adj = (state_next - env.observation_space.low)*np.array([40, 200])
            state2_adj = np.round(state2_adj, 0).astype(int)


            tot_reward+=updated_reward
            q_solver.experience_replay(done, state_adj, action, updated_reward, state2_adj, state_next, run)
            state_adj = state2_adj
            if state_next[0]>=0.5:
                print ("GOAL!!!!!")
                print("Run: " + str(run) + ", exploration: " + str(q_solver.exploration_rate) + ", tot reward: " + str(tot_reward) + ", steps: " + str(step))
                rewards.append(tot_reward)
                steps.append(step)
                break
            if step>=500:
            #if done:
                print("Run: " + str(run) + ", exploration: " + str(q_solver.exploration_rate) + ", tot reward: " + str(tot_reward) + ", steps: " + str(step))
                rewards.append(tot_reward)
                steps.append(step)
                break




if __name__ == "__main__":
    mountain()
