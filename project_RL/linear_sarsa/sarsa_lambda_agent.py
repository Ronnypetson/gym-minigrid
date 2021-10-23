import random
import numpy as np
from project_RL.linear_sarsa.parsing import parse_observation_to_state


class LinearSarsaLambda:
    """Sarsa Lambda Algorithm exploits and explores an environment.
    It learns the state action value of each state and action pair.
    It converges to the optimal action-value function.
    """

    def __init__(self,
                 env,
                 discount_rate=0.9,
                 learning_rate=0.1,
                 epsilon=0.5,
                 lambda_param=0.8):
        self.action_size = env.action_space.n
        observation = env.reset()
        state = parse_observation_to_state(observation)
        self.num_features = len(state)
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
        # self.q_value_table = dd(lambda: {i: 0 for i in range(0, self.action_size)})
        self.q_value_table = np.zeros((self.action_size, self.num_features))

    def init_eligibility_table(self):
        """Initialise eligibility trace table with zeros. Must be invoked before each episode.
        Its first dimension is the state size and the second dimension is the action size.
        """
        # self.eligibility_table = dd(lambda: dd(float))
        self.eligibility_table = np.zeros((self.action_size, self.num_features))

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
        # q_state = self.q_value_table[state]
        q_state = self.q_value_table @ state
        # max_value = max(q_state.items(), key=lambda x: x[1])
        max_value = max(q_state)
        list_of_max_actions = list()
        # Iterate over all the items in dictionary to find keys with max value
        for key, value in enumerate(q_state):
            # if value == max_value[1]:
            if value == max_value:
                list_of_max_actions.append(key)
        # try:
        #     action = random.choice(list_of_max_actions)
        # except:
        #     import pdb; pdb.set_trace()
        action = random.choice(list_of_max_actions)
        return action

    def update(self, state, action, reward, new_state, new_action, done):
        """Updates the state action value for every pair state and action
        in proportion to TD-error and eligibility trace

        """
        q_value_state_s = self.q_value_table @ state
        q_value_new_state = self.q_value_table @ new_state
        td_error = reward + self.discount_rate * q_value_new_state[new_action] - q_value_state_s[action]
        self.eligibility_table[action] = self.discount_rate\
                                         * self.lambda_param\
                                         * self.eligibility_table[action]\
                                         + state
        # self.eligibility_table[action] = self.discount_rate * self.lambda_param * state
        self.q_value_table[action] += self.learning_rate * td_error * self.eligibility_table[action]
        # print(td_error, np.linalg.norm(self.q_value_table), np.linalg.norm(self.eligibility_table))
