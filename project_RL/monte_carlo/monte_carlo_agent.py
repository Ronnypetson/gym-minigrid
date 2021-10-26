import random
from collections import defaultdict as dd

class MonteCarlo:

    def __init__(self, env, n_zero=7):
        self.action_size = env.action_space.n
        self.n_zero = n_zero
        self.greedy_policy = lambda x: self.n_zero/(self.n_zero + x)
        self.init_q_value_table()
        self.init_visited_states()

    def init_q_value_table(self):
        """Creates q_value_table as a dictionary.
        Its first dimension is the state size and the second dimension is the action size.
        The q_value_table is initially zero. It increases as new states are observed.
        Zero is the default value for actions.
        """
        self.q_value_table = dd(lambda: {i: 0 for i in range(0, self.action_size)})

    def init_eligibility_table(self):
        """Initialise eligibility trace table with zeros. Must be invoked before each episode.
        Its first dimension is the state size and the second dimension is the action size.
        """
        self.eligibility_table = dd(lambda: dd(int))

    def init_visited_states(self):
        """**Initialise visited states table with zeros.
        Its first dimension is the state itself and the second dimension is the quantity of times that state has been visited"""
        self.visited_states = dd(int)

    def get_epsilon(self, state):
        return 1 if not self.visited_states[state] else self.greedy_policy(self.visited_states[state])

    def get_new_action_e_greedly(self, state):
        """With probability 1 - epsilon choose the greedy action.
        With probability epsilon choose random action.
        """
        eps = self.get_epsilon(state)
        rnd = random.random()
        if rnd < eps:
            action = random.choice(range(self.action_size))
            # print(f'random action: {action}')
            return random.choice(range(self.action_size))
        else:
            action = self.get_new_action_greedly(state)
            return action
    
    def get_new_action_greedly(self, state):
        """With probability 1.0 choose the greedy action.
        With probability 0.0 choose random action.
        Uses a random selection of actions whenever you have a draw among actions.
        """
        # max_action = 0
        # max_val = -float('inf')
        # reward_state = self.q_value_table[state]
        # for action in reward_state.keys():
        #     if reward_state[action] > max_val:
        #         max_val = reward_state[action]
        #         max_action = action
        
        # if(max_action > 0):
        #     print(f'reward {reward_state[max_action]}')
        #     print(f'MAX ACTION {max_action} AND TEST {max(self.q_value_table[state], key=self.q_value_table[state].get)}')
        # return max_action

        max_value = self.q_value_table[state][max(self.q_value_table[state], key=self.q_value_table[state].get)]
        max_actions = [i for i, x in self.q_value_table[state].items() if x == max_value]
        return random.choice(max_actions)

    def update(self, states, actions, rewards,):
        for state, action, reward in zip(states, actions, rewards):
            self.eligibility_table[state][action] +=1
            self.visited_states[state] +=1
        
        for state, action, reward in zip(states, actions, rewards):
            self.q_value_table[state][action] += reward/self.eligibility_table[state][action]
