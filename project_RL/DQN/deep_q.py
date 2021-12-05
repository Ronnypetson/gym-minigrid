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
        discount_rate: float=0.9,
        learning_rate: float=1e-3,
        n0: int=3,
        min_eps: float=0.3,
        min_replay_size: int=512,
        max_replay_size: int=1024,
        batch_size: int=8
    ):
        self.action_size = env.action_space.n
        # observation = env.reset()
        self.learning_rate = learning_rate
        self.n0 = n0
        self.discount_rate = discount_rate
        self.min_eps = min_eps
        self.visited_states = dd(int)
        aspect_reduction = 2
        original_size = 112
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.q_net = QNet(
            num_actions=self.action_size,
            h = original_size // aspect_reduction,
            w = original_size // aspect_reduction
        ).to(self.device)
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
        self.opt = Adam(self.q_net.parameters(), lr=learning_rate)

    def get_new_action_greedly(
        self,
        state: np.ndarray
        ):
        """ Picks a new action with maximum value.
        """
        state = self._transforms(state).to(self.device)
        self.q_net.eval()
        state_q = self.q_net(state)[0].cpu().detach().numpy()
        max_value = max(state_q)
        max_actions = [i for i, x in enumerate(state_q) if x == max_value]
        return random.choice(max_actions)

    def _get_state_eps(
        self,
        state
        ) -> float:
        ''' Returns the eps of the state for epsilon-greedy policy.
        '''
        state.flags.writeable = False
        state = hash(state.data.tobytes())
        if self.visited_states[state] == 0:
            return 1
        else:
            return self.n0 / (self.n0 + self.visited_states[state])

    def get_new_action_e_greedly(
        self,
        state: np.ndarray
        ):
        ''' Chooses epsilon-greedy action from the current velue function.
        '''
        eps = self._get_state_eps(state)
        eps = min(self.min_eps, eps)
        if random.random() < eps:
            return random.choice(range(self.action_size))
        else:
            return self.get_new_action_greedly(state)

    def _update(self):
        ''' Runs one mini-batch inference and gradient descent update.
        '''
        self.q_net.train()
        batch_ids = np.random.choice(list(range(len(self._replay_buffer))), self.batch_size)
        batch_exp = [self._replay_buffer[idx] for idx in batch_ids]
        state, action, reward, new_state, done = experience2batches(batch_exp, device=self.device)
        # Run one backpropagation step
        self.opt.zero_grad()
        q_value = self.q_net(state)
        new_q_value = self.q_net(new_state)
        loss = reward + self.discount_rate * (1.0 - done) * torch.max(new_q_value, dim=1)[0]\
            - q_value[torch.arange(q_value.size(0)), action]
        loss = torch.mean(loss ** 2.0)
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
            self._transforms(new_state),
            torch.tensor(float(done)).float().unsqueeze(0)
        )
        self._replay_buffer.append(experience)
        buffer_len = len(self._replay_buffer)
        if buffer_len < self.min_replay_size:
            return
        if buffer_len == self.max_replay_size:
            self._replay_buffer.pop(0)
        self._update()
