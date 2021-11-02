from project_RL import plot
from project_RL.linear_q_learning.q_learning_agent import LinearQLearning
from gym_minigrid.wrappers import *
from datetime import datetime as dt
from project_RL.play import play
from project_RL.parsing import linear_parse_observation_to_state
import dill
import pickle


def train(env, hyperparameters):
    """ Train a sarsa lambda agent in the requested environment

    Arguments:
        hyperparameters dictionary containing:
            - env_name
            - discount_rate
            - learning_rate
            - lambda
            - n0 (initial exploration rate, it decays as the number of visits to a state increases)
    """
    agent = LinearQLearning(env,
                            hyperparameters['discount_rate'],
                            hyperparameters['learning_rate'],
                            hyperparameters['n0'])

    # create log file, add hyperparameters into it
    env_name = hyperparameters['env_name']

    log_filename = f'log_{env_name}_{dt.now().strftime("%y-%m-%d-%H-%M-%S")}.csv'
    with open(log_filename, 'a') as f:
        f.write(f'hyperparameters_size,{hyperparameters.__len__()}\n')
        f.write('\n'.join(map(','.join, {str(key): str(value) for key, value in hyperparameters.items()}.items())))
        f.write('\n')
        # write csv header
        f.write('Episode,Step,Total Reward,q_value_table_length\n')

    # initialise variables for plotting purpose
    step = 0

    for episode in range(int(1e4)):
        # reset environment before each episode
        total_reward = 0.0

        observation = env.reset()
        state = linear_parse_observation_to_state(observation)
        action = agent.get_new_action_e_greedly(state)
        done = False

        while not done:
            observation, reward, done, info = env.step(action)
            next_state = linear_parse_observation_to_state(observation)
            total_reward += reward
            agent.update(state, action, reward, next_state, None, done)

            state = next_state
            action = agent.get_new_action_e_greedly(next_state)
        
            if done:
                with open(log_filename, 'a') as f:
                    f.write(f'{episode},{step},{total_reward},{agent.q_value_table.__len__()}\n')
                if episode % 100 == 0:
                    play(env, agent, linear_parse_observation_to_state)
            step += 1
    env.close()

    with open(f'agent_{log_filename[:-4]}.pickle', 'wb') as f:
        dill.dump(agent, f, pickle.HIGHEST_PROTOCOL)

    plot.plot(log_filename[:-4])  # filename without extension

    return agent


if __name__ == '__main__':
    hyperparameters = {
        # 'env_name': 'MiniGrid-Empty-5x5-v0',
        # 'env_name': 'MiniGrid-DoorKey-8x8-v0',
        # 'env_name': 'MiniGrid-Empty-Random-6x6-v0',
        # 'env_name': 'MiniGrid-Empty-16x16-v0',
        # 'env_name': 'MiniGrid-DistShift1-v0',
        # 'env_name': 'MiniGrid-LavaGapS5-v0',
        # 'env_name': 'MiniGrid-SimpleCrossingS9N1-v0',
        # 'env_name': 'MiniGrid-Dynamic-Obstacles-5x5-v0',
        # 'env_name': 'MiniGrid-Dynamic-Obstacles-Random-6x6-v0',
        'env_name': 'MiniGrid-DoorKeyObst-7x7-v0',
        'discount_rate': 0.99,
        'learning_rate': 1e-3,
        'n0': 20
    }

    env = gym.make(hyperparameters['env_name'])
    agent = train(env, hyperparameters)
