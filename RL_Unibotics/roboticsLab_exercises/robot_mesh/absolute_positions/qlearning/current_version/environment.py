import sys
import logging
import random
import numpy as np
from collections import deque




class meshEnvironment():

    NUM_OF_X_BLOCKS=10
    NUM_OF_Y_BLOCKS=10

    INIT_Y=9
    INIT_X=0

    robot_pos_y=INIT_Y
    robot_pos_x=INIT_X

    #NOTE THAT blocks must be numbered considering that this will be the layout:
    # For self.NUM_OF_X_BLOCKS=5 and  self.NUM_OF_Y_BLOCKS=4
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

    def __init__(self):
        print("Environment initialized")

    def act(self, q_solver, state):
        if np.random.rand() < q_solver.exploration_rate:
            logging.warning("executed random action")
            return random.choice(list(self.ACTIONS.keys()))
        logging.info(q_solver.q_values[state-1])
        return np.argmax(q_solver.q_values[state-1])

    def one_action(self, state):
        state_next=state+self.NUM_OF_Y_BLOCKS
        self.robot_pos_x=self.robot_pos_x+1
        return state_next


    def two_action(self, state):
        state_next=state-self.NUM_OF_Y_BLOCKS
        self.robot_pos_x=self.robot_pos_x-1
        return state_next


    def three_action(self, state):
        state_next = state+1
        self.robot_pos_y=self.robot_pos_y-1
        return state_next


    def four_action(self,state):
        state_next = state-1
        self.robot_pos_y=self.robot_pos_y+1
        return state_next


    def undefined_action(self,state):
        logging.info("invalid action ")
        return 0

    def execute_step(self, state, action):
        switcher = {
                0: self.one_action, #RIGHT
                1: self.two_action, #LEFT
                2: self.three_action, #UP
                3: self.four_action #DOWN
        }
        logging.info("action " + self.ACTIONS[action])
        state_next=switcher.get(action, self.undefined_action)(state)
        logging.info("state_next " +str(state_next))
        return state_next

    def reset(self):
        self.robot_pos_y=self.INIT_Y
        self.robot_pos_x=self.INIT_X

    def get_state(self):
        return (self.robot_pos_x*self.NUM_OF_Y_BLOCKS)+(self.NUM_OF_Y_BLOCKS-self.robot_pos_y)
