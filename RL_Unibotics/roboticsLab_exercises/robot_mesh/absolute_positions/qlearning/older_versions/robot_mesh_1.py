import time
import random
import numpy as np
from collections import deque

GAMMA = 0.95

LEARNING_RATE = 0.001

MEMORY_SIZE = 1000000
BATCH_SIZE = 20

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.995


class QSolver:

    def __init__(self, action_space):
        self.exploration_rate = EXPLORATION_MAX
        self.q_values = [ [ 1 for i in range(4) ] for j in range(20) ]
        print("q_values init")
        print(str(self.q_values))
        self.action_space = action_space
        self.memory = deque(maxlen=MEMORY_SIZE)

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            print("executed random action")
            return random.randrange(self.action_space[0], self.action_space[-1])
        return np.argmax(self.q_values[state-1])

    def calculate_quality(self, state, action, reward, state_next):
        if state_next==16 or state_next<1 or state_next>20 or state_next==5 or state_next==6 or state_next==8 or state_next==12 or state_next ==14 or state_next==20 or state_next >20 or state_next <1:
            q_update=reward
        else:    
            q_update = (reward + GAMMA * np.amax(self.q_values[state_next-1]))
        
        self.q_values[state-1][action] = q_update
        print("updated state " + str(state) + " action " + str(action) + " to " + str(q_update)) 
        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)

def get_actions():
    m = []
    for i in range(4):
        m.append(i)
    return m

def one_action(state):
    state_next=state+4
    print("right")
    return state_next


def two_action(state):  
    state_next=state-4
    print("left")
    return state_next


def three_action(state):
    state_next = state+1
    print ("up")
    return state_next


def four_action(state):
    state_next = state-1
    print ("down")
    return state_next

def undefined_action(state):
    print("invalid action "+str(action))
    return 0
        
def execute_step(state, action):
    switcher = {
            0: one_action,
            1: two_action,
            2: three_action,
            3: four_action
    }
    print("action " + str(action))
    state_next=switcher.get(action, undefined_action)(state)
    print("state_next " +str(state_next))
    return state_next


def get_reward(state):
    
    if state<1 or state >20:
        reward = -10
    elif state == 5 or state==6 or state==8 or state==14 or state==12 or state==20:
        reward = -10
    elif state == 16:
        reward = 10
    else:
        reward = 0
    
    return reward


def robot():
    action_space = get_actions()
    q_solver = QSolver(action_space)
    run = 0
    while True:
        run += 1
        state = 1
        step = 0
        while True:
            step += 1
            print("STEP " + str(step) + " RUN " + str(run) + " -------------------------")
            action = q_solver.act(state)
            print("executing step")
            state_next= execute_step(state, action)
            reward = get_reward(state_next)
            print("reward " + str(reward))
            q_solver.calculate_quality(state, action, reward, state_next)
            state = state_next
            if state==16:
                print("GOAL!!! run: " + str(run) + ", exploration: " + str(q_solver.exploration_rate) + ", step: " + str(step))
                state=1
                time.sleep(5)
                if q_solver.exploration_rate==EXPLORATION_MIN:
                    exit()
                break
            elif state<1 or state >20:
                print("OUT OF LIMITS!!! run: " + str(run) + ", exploration: " + str(q_solver.exploration_rate) + ", step: " + str(step))
                state=1
                time.sleep(0.2)
                break
            elif state==5 or state==6 or state==8 or state==12 or state==14 or state==20:
                print("BOMB!!! run: " + str(run) + ", exploration: " + str(q_solver.exploration_rate) + ", step: " + str(step))
                state=1
                time.sleep(0.2)
                break
                            




if __name__ == "__main__":
    robot()
