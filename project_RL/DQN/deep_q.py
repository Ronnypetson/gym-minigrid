from typing import List, Tuple
import random
import numpy as np
import torch
from torch.optim import Adam
from collections import defaultdict as dd
from project_RL.agent import BaseAgent
from project_RL.DQN.q_network import QNet
from project_RL.batch import (
    np2tensor,
    resize_range,
    experience2batches,
    NumpyArrayTransform
)


class DQNAgent(BaseAgent):
    """ Deep Neural Network approximator of the Q value function.
    """
    def __init__(
        self,
        env,
        discount_rate=0.9,
        learning_rate=1e-3,
        n0=3,
        min_eps=0.3,
        min_replay_size=512,
        max_replay_size=1024,
        batch_size=8
    ):
        self.action_size = env.action_space.n
        # observation = env.reset()
        self.learning_rate = learning_rate
        self.n0 = n0
        self.discount_rate = discount_rate
        self.min_eps = min_eps
        aspect_reduction = 2
        self.q_net = QNet(
            num_actions=self.action_size,
            h = 112 // aspect_reduction,
            w = 112 // aspect_reduction
        )
        self._transforms = NumpyArrayTransform(
            [
                resize_range,
                lambda x: x[::aspect_reduction, ::aspect_reduction],
                lambda x: np2tensor(x, True),
                lambda x: x.permute(0, 3, 1, 2)
            ]
        )
        self.min_replay_size = min_replay_size
        self.max_replay_size = max_replay_size
        self.batch_size = min(batch_size, self.min_replay_size)
        self._replay_buffer = []
        self.opt = Adam(self.q_net.parameters(), lr=3e-4)

    def get_new_action_greedly(
        self,
        state: np.ndarray
        ):
        """ Picks a new action with maximum value.
        """
        state = self._transforms(state)
        self.q_net.eval()
        state_q = self.q_net(state)[0].detach().numpy()
        max_value = max(state_q)
        max_actions = [i for i, x in enumerate(state_q) if x == max_value]
        return random.choice(max_actions)

    def get_new_action_e_greedly(
        self,
        state: np.ndarray
        ):
        ''' Chooses epsilon-greedy action from the current velue function.
        '''
        if random.random() < self.min_eps:
            return random.choice(range(self.action_size))
        else:
            return self.get_new_action_greedly(state)

    def _update(self):
        ''' Runs one mini-batch inference and gradient descent update.
        '''
        self.q_net.train()
        batch_ids = np.random.choice(list(range(len(self._replay_buffer))), self.batch_size)
        batch_exp = [self._replay_buffer[idx] for idx in batch_ids]
        state, action, reward, new_state = experience2batches(batch_exp)
        # Run one backpropagation step
        self.opt.zero_grad()
        q_value = self.q_net(state)
        new_q_value = self.q_net(new_state)
        loss = reward + self.discount_rate * torch.max(new_q_value, dim=1)[0]\
            - q_value[torch.arange(q_value.size(0)), action]
        loss = torch.mean(torch.abs(loss))
        loss.backward()
        self.opt.step()

    def update(self, state, action, reward, new_state, new_action, done):
        """ Updates the state action value for every pair state and action
        in proportion to TD-error and eligibility trace
        """
        experience = (
            self._transforms(state),
            action,
            torch.tensor(reward).float().unsqueeze(0),
            self._transforms(new_state)
        )
        self._replay_buffer.append(experience)
        buffer_len = len(self._replay_buffer)
        if buffer_len < self.min_replay_size:
            return
        if buffer_len == self.max_replay_size:
            self._replay_buffer.pop(0)
        self._update()
