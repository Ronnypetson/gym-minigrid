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
        min_replay_size=32,
        max_replay_size=1024,
        batch_size=32
    ):
        self.action_size = env.action_space.n
        observation = env.reset()
        self.learning_rate = learning_rate
        self.n0 = n0
        self.discount_rate = discount_rate
        self.min_eps = min_eps
        self.q_net = QNet()
        self._transforms = NumpyArrayTransform(
            [
                lambda x: np2tensor(x, True),
                resize_range,
                lambda x: x.mean(axis=-1)
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
        state_q = self.q_net(state)
        max_value = max(state_q[0])
        max_actions = [i for i, x in enumerate(state_q) if x == max_value]
        return random.choice(max_actions)

    def get_new_action_e_greedly(
        self,
        state: np.ndarray
        ):
        ''' Chooses epsilon-greedy action from the current velue function.
        '''
        return np.random.randint(0, self.action_size)

    def _experience2batches(
        self,
        exp: List[Tuple]
        ):
        ''' Unpacks a experiences into batch tensors.
        '''
        state = []
        action = []
        reward = []
        new_state = []
        for s, a, r, s_ in zip(exp):
            state.append(s)
            action.append(a)
            reward.append(r)
            new_state.append(s_)
        import pdb; pdb.set_trace()
        state = torch.cat(state, dim=0)
        new_state = torch.cat(new_state, dim=0)
        action = torch.tensor(action)
        reward = torch.cat(reward, dim=0)
        assert len(state) == len(new_state) == len(reward) == len(action)
        return state, action, reward, new_state

    def _update(self):
        ''' Runs one mini-batch inference and gradient descent update.
        '''
        self.q_net.train()
        batch_ids = np.random.choice(list(range(len(self._replay_buffer))), self.batch_size)
        batch_exp = [self._replay_buffer[idx] for idx in batch_ids]
        state, action, reward, new_state = self._experience2batches(batch_exp)
        self.opt.zero_grad()
        q_value = self.q_net(state)
        new_q_value = self.q_net(new_state)
        loss = reward + self.discount_rate * torch.max(new_q_value, dim=1)[0]\
            - q_value[torch.arange(q_value.size(0)), action]
        loss = torch.abs(loss)
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
