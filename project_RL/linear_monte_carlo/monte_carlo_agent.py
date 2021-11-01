import random
import numpy as np
from collections import defaultdict as dd
from project_RL.parsing import linear_parse_observation_to_state


class LinearMonteCarlo:
    """*-visit Monte-Carlo"""
    def __init__(self,
                 env,
                 learning_rate=1e-3,
                 n_zero=7,
                 gamma=0.9):
        self.action_size = env.action_space.n
        self.learning_rate = learning_rate
        observation = env.reset()
        state = linear_parse_observation_to_state(observation)
        self.num_features = len(state)
        self.gamma = gamma
        self.n_zero = n_zero
        self.init_q_value_table()
        self.init_visited_states()
        self.init_visited_state_action()

    def init_q_value_table(self):
        """Creates q_value_table as a dictionary.
        Its first dimension is the state size and the second dimension is the action size.
        The q_value_table is initially zero. It increases as new states are observed.
        Zero is the default value for actions.
        """
        self.q_value_table = np.zeros((self.action_size, self.num_features))

    def init_visited_state_action(self):
        """Initialise eligibility trace table with zeros.
        Its first dimension is the state size and the second dimension is the action size.
        """
        self.visited_state_action = dd(lambda: dd(int))

    def init_visited_states(self):
        """**Initialise visited states table with zeros.
        Its first dimension is the state itself and the second dimension is the quantity of times that state has been visited"""
        self.visited_states = dd(int)

    def get_epsilon(self, state):
        state = tuple(state.tolist())
        return 1 if not self.visited_states[state] else self.n_zero/(self.n_zero + self.visited_states[state])

    def get_new_action_e_greedly(self, state):
        """With probability 1 - epsilon choose the greedy action.
        With probability epsilon choose random action.
        """
        eps = self.get_epsilon(state)
        rnd = random.random()
        if rnd < eps:
            action = random.choice(range(self.action_size))
            return random.choice(range(self.action_size))
        else:
            action = self.get_new_action_greedly(state)
            return action
    
    def get_new_action_greedly(self, state):
        """With probability 1.0 choose the greedy action.
        With probability 0.0 choose random action.
        Uses a random selection of actions whenever you have a draw among actions.
        """
        q_state = self.q_value_table @ state
        max_value = max(q_state)
        max_actions = [i for i, x in enumerate(q_state) if x == max_value]
        return random.choice(max_actions)

    def update(self, states, actions, rewards):
        ''' Diga podi crê omi
        '''
        states = states[::-1]
        actions = actions[::-1]
        rewards = rewards[::-1]
        g = 0.0
        for state, action, reward in zip(states, actions, rewards):
            g = g * self.gamma + reward
            _state = tuple(state.tolist())
            self.visited_state_action[_state][action] += 1
            self.visited_states[_state] += 1
            lr = self.learning_rate * (1 / self.visited_state_action[_state][action])
            self.q_value_table[action] += lr * (g - self.q_value_table[action] @ state) * state
