from abc import ABC, abstractmethod


class BaseAgent(ABC):
    """ Abstract class for all agents.
    """
    @abstractmethod
    def update(
        self,
        state,
        action,
        reward,
        new_state,
        **args
    ):
        """ Method for the update rule of the agent.
        """


class BaseTabularAgent(BaseAgent):
    """ Abstract class for all tabular agents.
    """
    @abstractmethod
    def __init_q_value_table(self):
        """Creates q_value_table as a dictionary.
        Its first dimension is the state size and the second dimension is the action size.
        The q_value_table is initially zero. It increases as new states are observed.
        Zero is the default value for actions.
        """

    @abstractmethod
    def __init_state_visits_table(self):
        """Initialise state visits table with zeros.
        It stores how many times each state was visited while the agent is trained.
        """

    @abstractmethod
    def get_new_action_e_greedly(self, state):
        """With probability 1 - epsilon choose the greedy action.
        With probability epsilon choose random action.

        """

    @abstractmethod
    def get_new_action_greedly(self, state):
        """With probability 1.0 choose the greedy action.
        With probability 0.0 choose random action.
        Uses a random selection of actions whenever you have a draw among actions.
        """
