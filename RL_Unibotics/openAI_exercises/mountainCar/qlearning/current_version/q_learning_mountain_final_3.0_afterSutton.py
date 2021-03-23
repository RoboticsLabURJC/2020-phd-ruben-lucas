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

#It should be multiple of 10
MAX_RUNS=3000
MAXIMUM_STEPS=500
EXPLORATION_STEPS_PER_STATE=100

INTERPOLATION=MAX_RUNS/10

ENV_NAME = "MountainCar-v0"

GAMMA = 0.95
LEARNING_RATE = 0.2

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.1
EXPLORATION_DECAY = 0.99995

LEVEL_GRANULARITY=0.01

class QSolver:

    def __init__(self, env):
        self.exploration_rate = EXPLORATION_MAX
        self.action_space = env.action_space.n
        print(env.observation_space.low)
        print(env.observation_space.high)
        # Determine size of discretized state space
        self.num_states = (env.observation_space.high - env.observation_space.low)*\
                        np.array([40, 200])
        self.num_states = np.round(self.num_states, 0).astype(int) + 1
        print(self.num_states)

        # Initialize Q table
        self.q_values = np.random.uniform(low = 0, high = 0,
                              size = (self.num_states[0], self.num_states[1],
                                      env.action_space.n))
        # Initialize Q table
        # self.q_values = np.random.uniform(low = -1, high = 1,
        #                      size = (self.num_states[0], self.num_states[1],
        #                              env.action_space.n))


    def act(self, state, occurrences):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)
        action = np.argmax(self.q_values[state[0], state[1]])
        return action

    def experience_replay(self, done, state_adj, action, reward, state2_adj, state_next, run, occurrences):

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


def get_reward(state, step, last_best_state):
    if state[0] >= 0.5:
        return 1
    else:
        return 0
    return 0


def plot_rewards_per_run(axes, dataSet):
    rewards_graph=pd.DataFrame(dataSet)
    ax=rewards_graph.plot(ax=axes[0], title="reward per run");
    ax.set_xlabel("runs")
    ax.legend().set_visible(False)

def plot_steps_per_run(axes, dataSet):
    rewards_graph=pd.DataFrame(dataSet)
    ax=rewards_graph.plot(ax=axes[1], title="step per runs");
    ax.set_xlabel("run")
    ax.legend().set_visible(False)

def plot_max_states(axes, dataSet):
    rewards_graph=pd.DataFrame(dataSet)
    ax=rewards_graph.plot(ax=axes[2], title="closest position to the goal");
    ax.set_xlabel("run")
    ax.legend().set_visible(False)

def get_stats_figure(steps, rewards, max_states):
    fig, axes = plt.subplots(nrows=1, ncols=3)
    fig.set_size_inches(12, 4)
    plot_max_states(axes,max_states)
    plot_steps_per_run(axes, steps)
    plot_rewards_per_run(axes, rewards)
    return fig

def toggle_plot(ui, vbox, figures, counter):
    i=0
    while i<len(figures):
        if figures[i].canvas.isVisible():
            figures[i].canvas.hide()
        i=i+1
    figures[counter].canvas.show()

def get_figure_from_data(data):

    cmap = colors.ListedColormap(['green', 'blue', 'yellow'])
    bounds = [-0.5, 0.5, 1.5, 2.5]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    fig, ax = plt.subplots()
    matrix=ax.imshow(data, cmap=cmap, norm=norm)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.2)
    plt.colorbar(matrix, cax=cax, boundaries=bounds)
    # draw gridlines
    ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=1)
    ax.set_xticks(np.arange(-.5, len(data[0]), 1))
    ax.set_yticks(np.arange(-.5, len(data), 1))
    ax.grid(True)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    return fig


def get_figure(matrixes):
    figures=[]
    for matrix in matrixes:
        fig = get_figure_from_data(matrix)
        figures.append(fig)
    return figures


def show_in_same_window(stats, q_values_figures):
    app = QApplication.instance()
    ui = QWidget()
    vbox = QGridLayout()
    vbox.addWidget(stats.canvas, 1, 1, 1 , len(q_values_figures))
    buttonsCounter=0
    for qvalues in q_values_figures:
        buttons_canvas = QPushButton("qvalues " + str(buttonsCounter + 1), ui)
        buttons_canvas.clicked.connect((lambda buttonsCounter: lambda:toggle_plot(ui, vbox, q_values_figures, buttonsCounter))(buttonsCounter))
        buttonsCounter=buttonsCounter+1
        vbox.addWidget(buttons_canvas, 2, buttonsCounter)
    for qvalues in q_values_figures:
        #falta el tema del policy
        qvalues.canvas.setFixedHeight(600)
        vbox.addWidget(qvalues.canvas, 3, 1,  1 ,len(q_values_figures))
        qvalues.canvas.hide()

    ui.setLayout(vbox)
    ui.show()
    sys.exit(app.exec_())

def set_q_values_in_graph(q_values_matrix, index, qsolver):
    i=0
    while i<qsolver.num_states[0]:
        j=0
        while j<qsolver.num_states[1]:
            q_values_matrix[index][j][i]=np.argmax(qsolver.q_values[i][j])
            j+=1
        i+=1
    return q_values_matrix

def mountain():
    env = gym.make(ENV_NAME)
    q_solver = QSolver(env)
    run = 0
    rewards=[]
    steps=[]
    max_states=[]
    q_values_matrix=[[[0 for x in range(int(q_solver.num_states[0]))] for y in range(int(q_solver.num_states[1]))] for j in range(int(MAX_RUNS/INTERPOLATION)) ]
    state_occurrences=[[0 for x in range(int(q_solver.num_states[0]))] for y in range(int(q_solver.num_states[1]))]
    last_best_state=-100

    while True:
        run += 1
        if run>MAX_RUNS:
            stats_figure=get_stats_figure(steps, rewards, max_states)
            qvalues_fig=get_figure(q_values_matrix)
            show_in_same_window(stats_figure, qvalues_fig)
            exit()

        state = env.reset()

        # Discretize state
        state_adj = (state - env.observation_space.low)*np.array([40, 200])
        state_adj = np.round(state_adj, 0).astype(int)

        step = 0
        tot_reward=0
        states=[]
        while True:
            step += 1
            env.render()
            state_occurrences[state_adj[1]][state_adj[0]]+=1
            action = q_solver.act(state_adj, state_occurrences[state_adj[1]][state_adj[0]])
            #time.sleep(1)
            state_next, reward, done, info = env.step(action)
            states.append(state_next)
            #print (state_next)
            updated_reward=get_reward(state_next, step, last_best_state)
            if state_next[0]>=last_best_state+LEVEL_GRANULARITY:
                last_best_state=state_next[0]

            # Discretize state2
            state2_adj = (state_next - env.observation_space.low)*np.array([40, 200])
            state2_adj = np.round(state2_adj, 0).astype(int)


            tot_reward+=updated_reward
            #print("state " + str(state_adj[1]) + " " +  str(state_adj[0])  + " occurrences " +str(state_occurrences[state_adj[1]][state_adj[0]]))

            q_solver.experience_replay(done, state_adj, action, updated_reward, state2_adj, state_next, run, state_occurrences[state_adj[1]][state_adj[0]])
            state_adj = state2_adj
            if state_next[0]>=0.5:
                print ("GOAL!!!!!")
                print("Run: " + str(run) + ", exploration: " + str(q_solver.exploration_rate) + ", tot reward: " + str(tot_reward) + ", steps: " + str(step))
                rewards.append(tot_reward)
                steps.append(step)
                print("max position -> " + str(np.max(states)))
                max_states.append(np.max(states))
                if run%INTERPOLATION==0:
                    q_values_matrix=set_q_values_in_graph(q_values_matrix, int((run/INTERPOLATION)-1), q_solver)
                break
            if step>=MAXIMUM_STEPS:
            #if done:
                print("Run: " + str(run) + ", exploration: " + str(q_solver.exploration_rate) + ", tot reward: " + str(tot_reward) + ", steps: " + str(step))
                rewards.append(tot_reward)
                steps.append(step)
                max_states.append(np.max(states))
                print("max position -> " + str(np.max(states)))
                if run%INTERPOLATION==0:
                    q_values_matrix=set_q_values_in_graph(q_values_matrix, int((run/INTERPOLATION)-1), q_solver)
                break




if __name__ == "__main__":
    mountain()
