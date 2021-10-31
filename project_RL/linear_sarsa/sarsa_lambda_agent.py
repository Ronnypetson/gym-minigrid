import random
import numpy as np
from collections import defaultdict as dd
from project_RL.linear_sarsa.parsing import parse_observation_to_state


class LinearSarsaLambda:
    """Sarsa Lambda Algorithm exploits and explores an environment.
    It learns the state action value of each state and action pair.
    It converges to the optimal action-value function.
    """
    def __init__(self,
                 env,
                 discount_rate=0.9,
                 initial_learning_rate=1e-2,
                 epsilon=0.5,
                 lambda_param=0.8,
                 n0=None):
        self.action_size = env.action_space.n
        observation = env.reset()
        state = parse_observation_to_state(observation)
        self.num_features = len(state)
        self.epsilon = epsilon
        self.discount_rate = discount_rate
        self.initial_learning_rate = initial_learning_rate
        self.lambda_param = lambda_param
        self.n0 = n0
        self.__init_q_value_table()
        self.__init_state_visits_table()

    def __init_q_value_table(self):
        """Creates q_value_table as a dictionary.
        Its first dimension is the state size and the second dimension is the action size.
        The q_value_table is initially zero. It increases as new states are observed.
        Zero is the default value for actions.

        """
        self.q_value_table = np.zeros((self.action_size, self.num_features))

    def __init_state_visits_table(self):
        """Initialise state visits table with zeros.
        It stores how many times each state was visited while the agent is trained.
        """
        self.state_visits = dd(lambda: 0)

    def init_eligibility_table(self):
        """Initialise eligibility trace table with zeros. Must be invoked before each episode.
        Its first dimension is the state size and the second dimension is the action size.
        """
        self.eligibility_table = np.zeros((self.action_size, self.num_features))

    def get_new_action_e_greedly(self, state):
        """With probability 1 - epsilon choose the greedy action.
        With probability epsilon choose random action.

        """
        if self.n0:
            eps = self.n0 / (self.n0 + self.state_visits[tuple(state)])
            eps = max(self.epsilon, eps)
        else:
            eps = self.epsilon
        if random.random() < eps:
            return random.choice(range(self.action_size))
        else:
            return self.get_new_action_greedly(state)

    def get_new_action_greedly(self, state):
        """With probability 1.0 choose the greedy action.
        With probability 0.0 choose random action.
        Uses a random selection of actions whenever you have a draw among actions.

        """
        q_state = self.q_value_table @ state
        max_value = max(q_state)
        list_of_max_actions = list()
        # Iterate over all the items in dictionary to find keys with max value
        for key, value in enumerate(q_state):
            if value == max_value:
                list_of_max_actions.append(key)
        action = random.choice(list_of_max_actions)
        return action

    def update(self, state, action, reward, new_state, new_action, done):
        """Updates the state action value for every pair state and action
        in proportion to TD-error and eligibility trace.
        """
        q_value_state_s = self.q_value_table @ state
        q_value_new_state = self.q_value_table @ new_state
        td_error = reward + self.discount_rate * q_value_new_state[new_action] - q_value_state_s[action]
        self.eligibility_table[action] = self.discount_rate\
                                         * self.lambda_param\
                                         * self.eligibility_table[action]\
                                         + state
        # self.state_visits[tuple(state)] += 1
        # learning_rate = self.initial_learning_rate / self.state_visits[tuple(state)]
        self.q_value_table[action] += self.initial_learning_rate * td_error * self.eligibility_table[action]
