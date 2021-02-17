#Q-LEARNING

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

LEARNING_RATE = 0.001

MEMORY_SIZE = 1000000

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.000001
EXPLORATION_DECAY = 0.99



NUM_OF_X_BLOCKS=10
NUM_OF_Y_BLOCKS=10

NUMBER_OF_LAST_OCCURRENCES_TO_PLOT=10
last_matrix_occurrences=[[[0 for x in range(NUM_OF_X_BLOCKS)] for y in range(NUM_OF_Y_BLOCKS)] for j in range(NUMBER_OF_LAST_OCCURRENCES_TO_PLOT) ]
last_occurrence=[]

INIT_Y=9
INIT_X=0
robot_pos_y=INIT_Y
robot_pos_x=INIT_X

#NOTE THAT blocks must be numbered considering that this will be the layout:
# For NUM_OF_X_BLOCKS=5 and  NUM_OF_Y_BLOCKS=4
# 4| 8| 12| 16| 20|
# 3| 7| 11| 15| 19|
# 2| 6| 10| 14| 18|
# 1| 5|  9| 13| 17|
BLOCKS={
1:"OK",
2:"OK",
3:"OK",
4:"OK",
5:"OK",
6:"OK",
7:"OK",
8:"OK",
9:"OK",
10:"BOMB",
11:"OK",
12:"BOMB",
13:"OK",
14:"BOMB",
15:"BOMB",
16:"OK",
17:"OK",
18:"OK",
19:"OK",
20:"BOMB",
21:"OK",
22:"OK",
23:"OK",
24:"OK",
25:"BOMB",
26:"BOMB",
27:"BOMB",
28:"BOMB",
29:"OK",
30:"BOMB",
31:"OK",
32:"BOMB",
33:"OK",
34:"BOMB",
35:"OK",
36:"BOMB",
37:"OK",
38:"OK",
39:"OK",
40:"BOMB",
41:"OK",
42:"OK",
43:"OK",
44:"BOMB",
45:"BOMB",
46:"OK",
47:"OK",
48:"OK",
49:"OK",
50:"BOMB",
51:"BOMB",
52:"OK",
53:"OK",
54:"OK",
55:"BOMB",
56:"BOMB",
57:"BOMB",
58:"OK",
59:"OK",
60:"OK",
61:"OK",
62:"OK",
63:"OK",
64:"OK",
65:"OK",
66:"BOMB",
67:"OK",
68:"BOMB",
69:"OK",
70:"BOMB",
71:"OK",
72:"BOMB",
73:"OK",
74:"BOMB",
75:"BOMB",
76:"OK",
77:"OK",
78:"OK",
79:"OK",
80:"BOMB",
81:"OK",
82:"OK",
83:"OK",
84:"OK",
85:"BOMB",
86:"BOMB",
87:"BOMB",
88:"BOMB",
89:"OK",
90:"BOMB",
91:"OK",
92:"BOMB",
93:"OK",
94:"BOMB",
95:"OK",
96:"BOMB",
97:"OK",
98:"OK",
99:"OK",
100:"GOAL"
}

ACTIONS={0:"RIGHT", 1:"LEFT", 2:"UP", 3:"DOWN"}

runs_rewards=[]
steps_per_run=[]

class QSolver:

    def __init__(self, action_space):
        self.exploration_rate = EXPLORATION_MAX
        self.q_values = [ [ 1 for i in range(len(ACTIONS)) ] for j in range(NUM_OF_X_BLOCKS*NUM_OF_Y_BLOCKS) ]
        logging.info("q_values init")
        logging.info(str(self.q_values))
        self.action_space = action_space
        self.memory = deque(maxlen=MEMORY_SIZE)

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            logging.warning("executed random action")
            return random.choice(list(self.action_space.keys()))
        return np.argmax(self.q_values[state-1])

    def calculate_quality(self, state, action, reward, state_next):
        if state_next not in BLOCKS or BLOCKS[state_next]=="BOMB" or BLOCKS[state_next]=="GOAL":
            q_update=reward
        else:
            q_update = (reward + GAMMA * np.amax(self.q_values[state_next-1]))

        self.q_values[state-1][action] = q_update
        logging.info("updated state " + str(state) + " action " + str(action) + " to " + str(q_update))
        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)

def one_action(state):
    global robot_pos_y
    global robot_pos_x

    state_next=state+NUM_OF_Y_BLOCKS
    robot_pos_x=robot_pos_x+1
    return state_next


def two_action(state):
    global robot_pos_y
    global robot_pos_x

    state_next=state-NUM_OF_Y_BLOCKS
    robot_pos_x=robot_pos_x-1
    return state_next


def three_action(state):
    global robot_pos_y
    global robot_pos_x

    state_next = state+1
    robot_pos_y=robot_pos_y-1
    return state_next


def four_action(state):
    global robot_pos_y
    global robot_pos_x

    state_next = state-1
    robot_pos_y=robot_pos_y+1
    return state_next

def undefined_action(state):
    logging.info("invalid action ")
    return 0

def execute_step(state, action):
    switcher = {
            0: one_action, #RIGHT
            1: two_action, #LEFT
            2: three_action, #UP
            3: four_action #DOWN
    }
    logging.info("action " + ACTIONS[action])
    state_next=switcher.get(action, undefined_action)(state)
    logging.info("state_next " +str(state_next))
    return state_next


def get_reward(state):
    if state not in BLOCKS or robot_pos_y>=NUM_OF_Y_BLOCKS or robot_pos_x>=NUM_OF_X_BLOCKS  or robot_pos_y<0 or robot_pos_x<0  or [state]=="BOMB":
        reward = -10
    elif BLOCKS[state]=="GOAL":
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
    ax.set_xticks(np.arange(-.5, NUM_OF_X_BLOCKS, 1))
    ax.set_yticks(np.arange(-.5, NUM_OF_Y_BLOCKS, 1))
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

def store_stats_and_reset(solver, steps, run_reward, run):
    global robot_pos_y
    global robot_pos_x
    global last_occurrence

    runs_rewards.append(run_reward)
    steps_per_run.append(steps)
    if solver.exploration_rate==EXPLORATION_MIN:
        stats_figure=get_stats_figure()
        most_frequent_occurrence, count=get_most_frequent_occurrence()
        most_frequent_paths_figure=get_path_figure_from_data(most_frequent_occurrence, count)
        unique_paths_dict=get_unique_paths_and_count(get_last_occurrences_tuple())
        last_succesfull_paths_figure=get_last_successfull_paths_figure(unique_paths_dict)
        show_in_same_window(stats_figure, last_succesfull_paths_figure, most_frequent_paths_figure)
    robot_pos_y=INIT_Y
    robot_pos_x=INIT_X
    store_last_occurrence_matrix(run)
    init_last_occurrence_matrix()


def init_last_occurrence_matrix():
    global last_occurrence

    last_occurrence = [[0 for x in range(NUM_OF_X_BLOCKS)] for y in range(NUM_OF_Y_BLOCKS)]
    last_occurrence[INIT_Y][INIT_X]=100
    for  x in range(NUM_OF_Y_BLOCKS):
        for y in range(NUM_OF_X_BLOCKS):
            if BLOCKS[(y*NUM_OF_Y_BLOCKS)+(NUM_OF_Y_BLOCKS-x)] == "GOAL":
                last_occurrence[x][y] = 100
            elif BLOCKS[(y*NUM_OF_Y_BLOCKS)+(NUM_OF_Y_BLOCKS-x)] == "BOMB":
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
    action_space = ACTIONS
    q_solver = QSolver(action_space)
    run = 0
    while True:
        run_reward=0
        run += 1
        state = (robot_pos_x*NUM_OF_Y_BLOCKS)+(NUM_OF_Y_BLOCKS-robot_pos_y)
        step = 0
        while True:
            step += 1
            logging.info("STEP " + str(step) + " RUN " + str(run) + " -------------------------")
            action = q_solver.act(state)
            logging.info("executing step")
            state_next= execute_step(state, action)
            reward = get_reward(state_next)
            logging.info("reward " + str(reward))
            run_reward+=reward
            q_solver.calculate_quality(state, action, reward, state_next)
            state = state_next
            if state not in BLOCKS or robot_pos_y>=NUM_OF_Y_BLOCKS or robot_pos_x>=NUM_OF_X_BLOCKS  or robot_pos_y<0 or robot_pos_x<0:
                logging.info("OUT OF LIMITS!!! run: " + str(run) + ", exploration: " + str(q_solver.exploration_rate) + ", step: " + str(step))
                store_stats_and_reset(q_solver, step, run_reward, run)
                break
            elif BLOCKS[state] == "GOAL" or BLOCKS[state] == "BOMB":
                logging.info(BLOCKS[state] + "!!! run: " + str(run) + ", exploration: " + str(q_solver.exploration_rate) + ", step: " + str(step))
                store_stats_and_reset(q_solver, step, run_reward, run)
                break
            else:
                logging.info("pos robot ->" + str(robot_pos_y) + ", " + str(robot_pos_x))
                last_occurrence[robot_pos_y][robot_pos_x]=last_occurrence[robot_pos_y][robot_pos_x]+1

if __name__ == "__main__":
    logging.basicConfig(filename='app.log', filemode='w', level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
    robot()
