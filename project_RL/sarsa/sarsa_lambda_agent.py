import random
import numpy as np
from collections import defaultdict as dd


class SarsaLambda:
    """Sarsa Lambda Algorithm exploits and explores an environment.
    It learns the state action value of each state and action pair.
    It converges to the optimal action-value function.
    """

    def __init__(self, env, discount_rate=0.9, learning_rate=0.1, epsilon=0.5, lambda_param=0.8):
        self.action_size = env.action_space.n
        self.epsilon = epsilon
        self.discount_rate = discount_rate
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.__init_q_value_table()

    def __init_q_value_table(self):
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
        self.eligibility_table = dd(lambda: dd(float))

    def get_new_action_e_greedly(self, state):
        """With probability 1 - epsilon choose the greedy action.
        With probability epsilon choose random action.

        """
        if random.random() < self.epsilon:
            return random.choice(range(self.action_size))
        else:
            return self.get_new_action_greedly(state)

    def get_new_action_greedly(self, state):
        """With probability 1.0 choose the greedy action.
        With probability 0.0 choose random action.
        Uses a random selection of actions whenever you have a draw among actions.

        """
        q_state = self.q_value_table[state]
        max_value = max(q_state.items(), key=lambda x: x[1])
        list_of_max_actions = list()
        # Iterate over all the items in dictionary to find keys with max value
        for key, value in q_state.items():
            if value == max_value[1]:
                list_of_max_actions.append(key)
        return random.choice(list_of_max_actions)

    # def _get_goal_position(self, state):
    #     grid = state['image']
    #     rows = len(grid)
    #     columns = len(grid[0])
    #     for row in range(rows):
    #         for column in range(columns):
    #             if grid[row][column][0] == 8:
    #                 return 10 * row + column
    #     return 0

    def update(self, state, action, reward, new_state, new_action, done):
        """Updates the state action value for every pair state and action
        in proportion to TD-error and eligibility trace

        """
        q_value_state_s = self.q_value_table[state]
        q_value_new_state = self.q_value_table[new_state]
        td_error = reward + self.discount_rate * q_value_new_state[new_action] - q_value_state_s[action]

        self.eligibility_table[state][action] += 1.0
        if np.abs(td_error) > 1e-4:
            for state in self.eligibility_table.keys():
                for action in self.eligibility_table[state].keys():
                    self.q_value_table[state][action] += self.learning_rate * td_error * self.eligibility_table[state][action]
                    self.eligibility_table[state][action] = self.discount_rate * self.lambda_param * self.eligibility_table[state][action]
