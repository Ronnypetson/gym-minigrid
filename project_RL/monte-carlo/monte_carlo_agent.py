import random
import numpy as np
from collections import defaultdict as dd

class MonteCarlo:
    DIRECTIONS = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    def __init__(self, n_zero=0.9):
        """The directions represent the possible direction(x, y) the agent could go
        NORTH (0, 1), SOUTH (0, -1), EAST(1, 0) or WEST (-1, 0)"""
        self.n_zero = n_zero
        self.greedy_policy = lambda x: self.n_zero/(self.n_zero + x)
        self.init_q_value_table()
        self.init_visited_states()

    def init_q_value_table(self):
        """Initialise q value table with zeros.
        Its first dimension is the state size and the second dimension is the action size.
        """
        self.q_value_table = dd(lambda: dd(float))

    def init_visited_states(self):
        """**Initialise visited states table with zeros.
        Its first dimension is the state itself and the second dimension is the quantity of times that state has been visited"""
        # Some help here
        self.visited_states = dd(int)

    def init_eligibility_table(self):
        """Initialise eligibility trace table with zeros. Must be invoked before each episode.
        Its first dimension is the state size and the second dimension is the action size.
        """
        self.eligibility_table = dd(lambda: dd(int))

    def get_epsilon(self, state):
        _state = tuple(state)
        return 1 if (not self.visited_states[_state] == True) else self.n_zero/(self.n_zero + self.visited_states[_state])

    def get_new_action_e_greedly(self, state):
        """With probability 1 - epsilon choose the greedy action.
        With probability epsilon choose random action.
        """
        reward_state = self.q_value_table[tuple(state)]

        if random.random() < self.get_epsilon(state):
            return random.choice(self.DIRECTIONS)
        else:
            max_action = 0
            max_val = -float('inf')
            for action in reward_state.keys():
                if reward_state[action] > max_val:
                    max_val = reward_state[action]
                    max_action = action
            return max_action

    def update(self, states, actions, rewards,):
        # Perhaps I should use zip(states, actions,rewards)
        for state, action, reward in states,actions,rewards:
            _state = tuple(state)
            _action = tuple(action)
            
            self.q_value_table[_state][_action] = reward
