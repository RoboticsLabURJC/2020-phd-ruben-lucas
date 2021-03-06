#Q-LEARNING
from environment import meshEnvironment
import sys
import logging
import random
import numpy as np
from collections import deque
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from collections import Counter

from PyQt5.QtWidgets import QVBoxLayout
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QLabel
from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *



GAMMA = 0.95

LEARNING_RATE = 0.9

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.000001
EXPLORATION_DECAY = 0.999

environment=meshEnvironment()

NUMBER_OF_LAST_OCCURRENCES_TO_PLOT=10
last_matrix_occurrences=[[[0 for x in range(environment.NUM_OF_X_BLOCKS)] for y in range(environment.NUM_OF_Y_BLOCKS)] for j in range(NUMBER_OF_LAST_OCCURRENCES_TO_PLOT) ]
last_occurrence=[]

runs_rewards=[]
steps_per_run=[]

class QSolver:

    def __init__(self):
        self.exploration_rate = EXPLORATION_MAX
        self.q_values = [ [ np.random.rand() for i in range(len(environment.ACTIONS)) ] for j in range(environment.NUM_OF_X_BLOCKS*environment.NUM_OF_Y_BLOCKS) ]
        logging.info("q_values init")
        logging.info(str(self.q_values))
        self.memory = deque(maxlen=MEMORY_SIZE)


    def calculate_quality(self, state, action, reward, state_next):
        if state_next not in environment.BLOCKS or environment.BLOCKS[state_next]=="BOMB" or environment.BLOCKS[state_next]=="GOAL":
            q_update = LEARNING_RATE * (reward - self.q_values[state-1][action])
        else:
            q_update = LEARNING_RATE *(reward + GAMMA * np.amax(self.q_values[state_next-1]) - self.q_values[state-1][action])
        self.q_values[state-1][action] =self.q_values[state-1][action] + q_update
        logging.info("updated state " + str(state) + " action " + str(action) + " to " + str(q_update))
        logging.info("exploration_rate " + str(self.exploration_rate))
        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)


def get_reward(state):
    if state not in environment.BLOCKS or environment.robot_pos_y>=environment.NUM_OF_Y_BLOCKS or environment.robot_pos_x>=environment.NUM_OF_X_BLOCKS  or environment.robot_pos_y<0 or environment.robot_pos_x<0  or environment.BLOCKS[state]=="BOMB":
        reward = -1
    elif environment.BLOCKS[state]=="GOAL":
        reward = 10
    else:
        reward = 0

    return reward

def plot_rewards_per_run(axes):
    rewards_graph=pd.DataFrame(runs_rewards)
    ax=rewards_graph.plot(ax=axes[0], title="rewards per run");
    ax.set_xlabel("runs")
    ax.set_ylabel("rewards")
    ax.legend().set_visible(False)

def plot_steps_per_run(axes):
    rewards_graph=pd.DataFrame(steps_per_run)
    ax=rewards_graph.plot(ax=axes[1], title="steps per run");
    ax.set_xlabel("run")
    ax.set_ylabel("steps")
    ax.legend().set_visible(False)


def get_path_figure_from_data(data, count):
    top = cm.get_cmap('Blues', 128)
    bottom = cm.get_cmap('Blues', 128)

    newcolors = np.vstack((top(np.linspace(0, 1, 128)),
                           bottom(np.linspace(0, 1, 128))))
    newcmp = colors.ListedColormap(newcolors, name='OrangeBlue')

    cmap = colors.ListedColormap(['white', newcmp(0.65), newcmp(0.7), newcmp(0.8), newcmp(0.9), newcmp(0.95)])
    cmap.set_over('green')
    cmap.set_under('black')
    bounds = [-0.5, 0.5, 1.5, 2.5, 4.5, 6.5, 9.5]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    fig, ax = plt.subplots()
    matrix=ax.imshow(data, cmap=cmap, norm=norm)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.2)
    plt.colorbar(matrix, cax=cax, boundaries=bounds)
    # add the text indicating the number of occurrences of more executed path in last 100 times
    plt.gcf().text(0.43, 0.91, str(count) + " of last " + str(NUMBER_OF_LAST_OCCURRENCES_TO_PLOT), fontsize=14)
    # draw gridlines
    ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
    ax.set_xticks(np.arange(-.5, environment.NUM_OF_X_BLOCKS, 1))
    ax.set_yticks(np.arange(-.5, environment.NUM_OF_Y_BLOCKS, 1))
    ax.grid(True)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    return fig

def get_last_successfull_paths_figure(occurences):
    figures=[]
    for element in occurences.keys():
        fig = get_path_figure_from_data(element, occurences.get(element))
        figures.append(fig)

    return figures


def get_stats_figure():
    fig, axes = plt.subplots(nrows=1, ncols=2)
    fig.set_size_inches(12, 4)
    plot_steps_per_run(axes)
    plot_rewards_per_run(axes)
    return fig

def show_in_same_window(stats, succesfull_paths, most_frequent_paths_figure):
    app = QApplication.instance()
    ui = QWidget()
    vbox = QGridLayout()
    vbox.addWidget(stats.canvas, 1, 1, 1 ,len(succesfull_paths))
    buttonsCounter=0
    for path in succesfull_paths:
        buttons_canvas = QPushButton("path " + str(buttonsCounter + 1), ui)
        buttons_canvas.clicked.connect((lambda buttonsCounter: lambda:toggle_plot(ui, vbox, most_frequent_paths_figure, succesfull_paths, buttonsCounter))(buttonsCounter))
        buttonsCounter=buttonsCounter+1
        vbox.addWidget(buttons_canvas, 2, buttonsCounter)
    for path in succesfull_paths:
        #falta el tema del policy
        path.canvas.setFixedHeight(300)
        vbox.addWidget(path.canvas, 3, 1,  1 ,len(succesfull_paths))
        path.canvas.hide()

    most_frequent_paths_figure.canvas.setFixedHeight(300)
    vbox.addWidget(most_frequent_paths_figure.canvas, 3, 1,  1 ,len(succesfull_paths))
    ui.setLayout(vbox)
    ui.show()
    sys.exit(app.exec_())

def store_last_occurrence_matrix(run):
    global last_occurrence
    global last_matrix_occurrences
    last_matrix_occurrences[run%NUMBER_OF_LAST_OCCURRENCES_TO_PLOT]=last_occurrence

def get_last_occurrences_tuple():
    sublists_tuples = [[tuple(ele) for ele in sub] for sub in last_matrix_occurrences]
    lists_tuples=[tuple(l) for l in sublists_tuples]
    last_occurrences_tuples=tuple(lists_tuples)
    return last_occurrences_tuples


def get_most_frequent_occurrence():
    most_frequent_occurrence, count=most_frequent(get_last_occurrences_tuple())
    return most_frequent_occurrence, count

def toggle_plot(ui, vbox, most_frequent, figures, counter):
    i=0
    while i<len(figures):
        if figures[i].canvas.isVisible():
            figures[i].canvas.hide()
        i=i+1
    figures[counter].canvas.show()

    # This function is called by a keypress to hide/show the figure
    if most_frequent.canvas.isVisible():
        most_frequent.canvas.hide()
    #ui.adjustSize()


  #plt.draw()

def store_stats_and_reset(steps, run_reward, run):
    global last_occurrence

    runs_rewards.append(run_reward)
    steps_per_run.append(steps)

    environment.reset()
    store_last_occurrence_matrix(run)
    init_last_occurrence_matrix()


def init_last_occurrence_matrix():
    global last_occurrence

    last_occurrence = [[0 for x in range(environment.NUM_OF_X_BLOCKS)] for y in range(environment.NUM_OF_Y_BLOCKS)]
    last_occurrence[environment.INIT_Y][environment.INIT_X]=100
    for  x in range(environment.NUM_OF_Y_BLOCKS):
        for y in range(environment.NUM_OF_X_BLOCKS):
            if environment.BLOCKS[(y*environment.NUM_OF_Y_BLOCKS)+(environment.NUM_OF_Y_BLOCKS-x)] == "GOAL":
                last_occurrence[x][y] = 100
            elif environment.BLOCKS[(y*environment.NUM_OF_Y_BLOCKS)+(environment.NUM_OF_Y_BLOCKS-x)] == "BOMB":
                last_occurrence[x][y] = -1

def most_frequent(List):
    occurence_count = Counter(List)
    most_frequent_occurrence=occurence_count.most_common(1)[0][0]
    return most_frequent_occurrence, occurence_count[most_frequent_occurrence]

def get_unique_paths_and_count(List):
    occurence_count = Counter(List)
    return dict(occurence_count)

def robot():
    init_last_occurrence_matrix()
    q_solver = QSolver()
    run = 0
    while True:
        run_reward=0
        run += 1
        state = environment.get_state()
        step = 0
        while True:
            step += 1
            logging.info("STEP " + str(step) + " RUN " + str(run) + " -------------------------")
            action = environment.act(q_solver, state)
            logging.info("executing step")
            state_next= environment.execute_step(state, action)
            reward = get_reward(state_next)
            logging.info("reward " + str(reward))
            run_reward+=reward
            q_solver.calculate_quality(state, action, reward, state_next)
            state = state_next
            if state not in environment.BLOCKS or environment.robot_pos_y>=environment.NUM_OF_Y_BLOCKS or environment.robot_pos_x>=environment.NUM_OF_X_BLOCKS  or environment.robot_pos_y<0 or environment.robot_pos_x<0:
                logging.info("OUT OF LIMITS!!! run: " + str(run) + ", exploration: " + str(q_solver.exploration_rate) + ", step: " + str(step))
                store_stats_and_reset(step, run_reward, run)
                break
            elif environment.BLOCKS[state] == "GOAL" or environment.BLOCKS[state] == "BOMB":
                logging.info(environment.BLOCKS[state] + "!!! run: " + str(run) + ", exploration: " + str(q_solver.exploration_rate) + ", step: " + str(step))
                store_stats_and_reset(step, run_reward, run)
                break
            else:
                logging.info("pos robot ->" + str(environment.robot_pos_y) + ", " + str(environment.robot_pos_x))
                last_occurrence[environment.robot_pos_y][environment.robot_pos_x]=last_occurrence[environment.robot_pos_y][environment.robot_pos_x]+1

            if q_solver.exploration_rate==EXPLORATION_MIN:
                stats_figure=get_stats_figure()
                most_frequent_occurrence, count=get_most_frequent_occurrence()
                most_frequent_paths_figure=get_path_figure_from_data(most_frequent_occurrence, count)
                unique_paths_dict=get_unique_paths_and_count(get_last_occurrences_tuple())
                last_succesfull_paths_figure=get_last_successfull_paths_figure(unique_paths_dict)
                show_in_same_window(stats_figure, last_succesfull_paths_figure, most_frequent_paths_figure)

if __name__ == "__main__":
    logging.basicConfig(filename='app.log', filemode='w', level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
    robot()
