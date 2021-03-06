import sys
import logging
import random
import numpy as np
from collections import deque
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors

from PyQt5.QtWidgets import QVBoxLayout
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QLabel
from PyQt5.QtWidgets import QWidget

GAMMA = 0.95

LEARNING_RATE = 0.001

MEMORY_SIZE = 1000000
BATCH_SIZE = 20

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.002
EXPLORATION_DECAY = 0.995



NUM_OF_X_BLOCKS=5
NUM_OF_Y_BLOCKS=4
#NUMBER_OF_LAST_OCCURRENCES_TO_PLOT=1
#max_last_occurrences=[NUMBER_OF_LAST_OCCURRENCES_TO_PLOT][4][5]
last_occurrence=[]

robot_pos_x=3
robot_pos_y=0

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
5:"BOMB",
6:"BOMB",
7:"OK",
8:"BOMB",
9:"OK",
10:"OK",
11:"OK",
12:"BOMB",
13:"OK",
14:"BOMB",
15:"OK",
16:"GOAL",
17:"OK",
18:"OK",
19:"OK",
20:"BOMB"
}
INITIAL_CELL=1

ACTIONS={0:"RIGHT", 1:"LEFT", 2:"UP", 3:"DOWN"}

runs_rewards=[]
steps_per_run=[]

class QSolver:

    def __init__(self, action_space):
        self.exploration_rate = EXPLORATION_MAX
        self.q_values = [ [ 1 for i in range(4) ] for j in range(20) ]
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
    global robot_pos_x
    global robot_pos_y

    state_next=state+NUM_OF_Y_BLOCKS
    robot_pos_y=robot_pos_y+1
    return state_next


def two_action(state):
    global robot_pos_x
    global robot_pos_y

    state_next=state-NUM_OF_Y_BLOCKS
    robot_pos_y=robot_pos_y-1
    return state_next


def three_action(state):
    global robot_pos_x
    global robot_pos_y

    state_next = state+1
    robot_pos_x=robot_pos_x-1
    return state_next


def four_action(state):
    global robot_pos_x
    global robot_pos_y

    state_next = state-1
    robot_pos_x=robot_pos_x+1
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
    if state not in BLOCKS or BLOCKS[state]=="BOMB":
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


def get_last_successfull_paths_figure():

    data = last_occurrence

    cmap = colors.ListedColormap(['black', 'white', 'green', 'blue', 'yellow', "cyan", "magenta"])
    bounds = [-1,-0.5,1,2,3,6,7]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    fig, ax = plt.subplots()
    ax.imshow(data, cmap=cmap, norm=norm)

    # draw gridlines
    ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
    ax.set_xticks(np.arange(-.5, NUM_OF_X_BLOCKS, 1))
    ax.set_yticks(np.arange(-.5, NUM_OF_Y_BLOCKS, 1))
    ax.grid(True)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    return fig


def get_stats_figure():
    fig, axes = plt.subplots(nrows=1, ncols=2)
    fig.set_size_inches(12, 4)
    plot_steps_per_run(axes)
    plot_rewards_per_run(axes)
    return fig

def show_in_same_window(stats, succesfull_paths):
    app = QApplication.instance()
    ui = QWidget()
    vbox = QVBoxLayout()
    vbox.addWidget(stats.canvas)
    vbox.addWidget(succesfull_paths.canvas)
    ui.setLayout(vbox)
    ui.show()
    sys.exit(app.exec_())

def store_stats_and_reset(solver, steps, run_reward):
    global robot_pos_x
    global robot_pos_y
    global last_occurrence

    runs_rewards.append(run_reward)
    steps_per_run.append(steps)
    if solver.exploration_rate==EXPLORATION_MIN:
        stats_figure=get_stats_figure()
        last_succesfull_paths_figure=get_last_successfull_paths_figure()
        show_in_same_window(stats_figure, last_succesfull_paths_figure)
    robot_pos_x=3
    robot_pos_y=0
    init_last_occurrence_matrix()
    return INITIAL_CELL


def init_last_occurrence_matrix():
    global last_occurrence
    global robot_pos_x
    global robot_pos_y

    last_occurrence = [[0 for x in range(NUM_OF_X_BLOCKS)] for y in range(NUM_OF_Y_BLOCKS)]
    last_occurrence[robot_pos_x][robot_pos_y]=6
    for  y in range(NUM_OF_Y_BLOCKS):
        for x in range(NUM_OF_X_BLOCKS):
            if BLOCKS[(x*NUM_OF_Y_BLOCKS)+(NUM_OF_Y_BLOCKS-y)] == "GOAL":
                last_occurrence[y][x] = 7
            elif BLOCKS[(x*NUM_OF_Y_BLOCKS)+(NUM_OF_Y_BLOCKS-y)] == "BOMB":
                last_occurrence[y][x] = -1




def robot():
    init_last_occurrence_matrix()
    action_space = ACTIONS
    q_solver = QSolver(action_space)
    run = 0
    while True:
        run_reward=0
        run += 1
        state = INITIAL_CELL
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
            if state not in BLOCKS:
                logging.info("OUT OF LIMITS!!! run: " + str(run) + ", exploration: " + str(q_solver.exploration_rate) + ", step: " + str(step))
                state=store_stats_and_reset(q_solver, step, run_reward)
                break
            elif BLOCKS[state] == "GOAL" or BLOCKS[state] == "BOMB":
                logging.info(BLOCKS[state] + "!!! run: " + str(run) + ", exploration: " + str(q_solver.exploration_rate) + ", step: " + str(step))
                state=store_stats_and_reset(q_solver, step, run_reward)
                break
            else:
                logging.info("pos robot ->" + str(robot_pos_x) + ", " + str(robot_pos_y))
                last_occurrence[robot_pos_x][robot_pos_y]=last_occurrence[robot_pos_x][robot_pos_y]+1

if __name__ == "__main__":
    logging.basicConfig(filename='app.log', filemode='w', level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
    robot()
