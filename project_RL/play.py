import os
import dill
from gym_minigrid.wrappers import *


def load_agent(agent_filename, env_name):
    with open(agent_filename, 'rb') as f:
        agent = dill.load(f)
        print('Play with trained agent')
        env = gym.make(env_name)
        play(env, agent, 100)


def play(env, agent, observation_parser, episodes=1):
    for episode in range(episodes):
        # reset environment before each episode
        observation = env.reset()
        state = observation_parser(observation)
        action = agent.get_new_action_greedly(state)
        done = False
        total_reward = 0

        env.render()
        while not done:
            observation, reward, done, info = env.step(action)
            env.render()
            next_state = observation_parser(observation)
            total_reward += reward
            action = agent.get_new_action_greedly(next_state)


if __name__ == '__main__':
    agent_relative_path = 'sarsa/trained_agent/agent_log_MiniGrid-DoorKeyObst-7x7-v0_21-10-26-14-37-21.pickle'
    agent_filename = os.path.join(os.path.dirname(__file__), agent_relative_path)
    env_name = 'MiniGrid-DoorKeyObst-7x7-v0'
    load_agent(agent_filename, env_name)
