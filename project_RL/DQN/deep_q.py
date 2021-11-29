import random
import numpy as np
from collections import defaultdict as dd
from project_RL.agent import BaseAgent


class DeepQLearning(BaseAgent):
    """ Deep Neural Network approximator of the Q value function.
    """
    def __init__(
        self,
        env,
        discount_rate=0.9,
        learning_rate=1e-3,
        n0=3,
        min_eps=0.3
    ):
        self.action_size = env.action_space.n
        observation = env.reset()
        self.learning_rate = learning_rate
        self.n0 = n0
        self.discount_rate = discount_rate
        self.min_eps = min_eps

    def update(self, state, action, reward, new_state, new_action, done):
        """Updates the state action value for every pair state and action
        in proportion to TD-error and eligibility trace
        """
        pass
