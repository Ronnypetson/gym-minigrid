import random
import numpy as np
from collections import defaultdict as dd
from project_RL.parsing import linear_parse_observation_to_state


class LinearQLearning:
    """Linear Approximator Q-Learning Algorithm exploits and explores an environment.
    It learns the state action value of each state and action pair.
    It converges to the optimal action-value function.
    """

    def __init__(self,
                 env,
                 discount_rate=0.9,
                 learning_rate=1e-3,
                 n0=3,
                 min_eps=0.3):
        self.action_size = env.action_space.n
        observation = env.reset()
        state = linear_parse_observation_to_state(observation)
        self.num_features = len(state)
        self.learning_rate = learning_rate
        self.n0 = n0
        self.discount_rate = discount_rate
        self.min_eps = min_eps
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

    def get_new_action_e_greedly(self, state):
        """With probability 1 - epsilon choose the greedy action.
        With probability epsilon choose random action.

        """
        _state = tuple(state.tolist())
        eps = self.n0 / (self.n0 + self.state_visits[_state])
        eps = max(eps, self.min_eps)
        if random.random() < eps:
            return random.choice(range(self.action_size))
        else:
            return self.get_new_action_greedly(state)

    def get_new_action_greedly(self, state):
        """With probability 1.0 choose the greedy action.
        With probability 0.0 choose random action.
        Uses a random selection of actions whenever you have a draw among actions.

        """
        ''' TODO: depurar os NaNs na q_value_table
        '''
        state_q = self.q_value_table @ state
        max_value = max(state_q)
        max_actions = [i for i, x in enumerate(state_q) if x == max_value]
        return random.choice(max_actions)

    def update(self, state, action, reward, new_state, new_action, done):
        """Updates the state action value for every pair state and action
        in proportion to TD-error and eligibility trace
        """
        # pre update
        _state = tuple(state.tolist())

        # import pdb; pdb.set_trace()
        q_value_state_s = self.q_value_table @ state
        q_value_new_state = self.q_value_table @ new_state
        td_error = reward + self.discount_rate * max(q_value_new_state) - q_value_state_s[action]

        ''' TODO: Checar essa condição.
        '''
        if np.abs(td_error) > 1e-4:
            self.q_value_table[action] += self.learning_rate * td_error * state

        # post update
        self.state_visits[_state] += 1
