import random
from collections import defaultdict as dd

class MonteCarlo:
    """*-visit Monte-Carlo"""
    def __init__(self, env, n_zero=7, gamma=0.9):
        self.action_size = env.action_space.n
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
        self.q_value_table = dd(lambda: {i: 0 for i in range(0, self.action_size)})

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
        max_value = self.q_value_table[state][max(self.q_value_table[state], key=self.q_value_table[state].get)]
        max_actions = [i for i, x in self.q_value_table[state].items() if x == max_value]
        return random.choice(max_actions)

    def update(self, states, actions, rewards):
        g = 0
        for state, action, reward in zip(states[::-1], actions[::-1], rewards[::-1]):
            g = g * self.gamma + reward
            self.visited_state_action[state][action] +=1
            self.visited_states[state] +=1
            self.q_value_table[state][action] += (1/self.visited_state_action[state][action]) * (g - self.q_value_table[state][action])
