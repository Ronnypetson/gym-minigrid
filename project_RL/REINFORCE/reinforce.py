import random
import numpy as np
import torch
from torch.optim import Adam
from project_RL.agent import BaseAgent
from project_RL.REINFORCE.q_network import QNet
from project_RL.batch import (
    np2tensor,
    resize_range,
    NumpyArrayTransform
)

from torch.distributions import Categorical


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
    ):
        self.action_size = env.action_space.n
        self.log_probs= []
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

        self.opt = Adam(self.q_net.parameters(), lr=learning_rate)

    def get_new_action_greedly(
        self,
        state: np.ndarray
        ):
        """ Picks a new action with maximum value.
        """
        state = self._transforms(state)
        # self.q_net.eval()
        # state_q = self.q_net(state)[0].detach().numpy()
        # max_value = max(state_q)
        # max_actions = [i for i, x in enumerate(state_q) if x == max_value]
        # chosen_action = random.choice(max_actions)
        
        # self.log_probs.append(m.log_prob(chosen_action))
        # print(self.log_probs)
        # return chosen_action
        # state = torch.from_numpy(state_q).float().unsqueeze(0)

        probs = self.q_net(state)
        m = Categorical(probs)
        action = m.sample()
        self.log_probs.append(m.log_prob(action))
        return action.item()

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

    def update(self, rewards):
        """ Updates the state action value for every pair state and action
        in proportion to TD-error and eligibility trace
        """
        if len(self.log_probs) < 2:
            return

        R = 0
        policy_loss = []
        returns = []

        self.q_net.train()
        # batch_ids = np.random.choice(list(range(len(self._replay_buffer))), self.batch_size)
        # batch_exp = [self._replay_buffer[idx] for idx in batch_ids]
        # states, actions, rewards, new_state = experience2batches(batch_exp)

        for r in rewards[::-1]:
            R = r + self.discount_rate * R
            returns.insert(0, R)

        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-2) ###

        for log_prob, R in zip(self.log_probs, returns):
            policy_loss.append(-log_prob * R)

        self.opt.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        for param in self.q_net.parameters():
            if not param.grad.isfinite().all():
                import pdb; pdb.set_trace()
        self.opt.step()
        self.log_probs = [] ### 
