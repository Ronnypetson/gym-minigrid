from project_RL.linear_monte_carlo.monte_carlo_agent import LinearMonteCarlo
from project_RL.parsing import linear_parse_observation_to_state
from project_RL.play import play
from gym_minigrid.wrappers import *
from time import time


def train(env, hyperparameters):
    """ Train a sarsa lambda agent in the requested environment

    Arguments:
        hyperparameters dictionary containing:
            - env_name
            - n zero value
    """
    agent = LinearMonteCarlo(env,
                             hyperparameters['learning_rate'],
                             hyperparameters['n_zero'])

    # create log file, add hyperparameters into it
    env_name = hyperparameters['env_name']
    log_filename = f'log_{env_name}_{time()}.csv'
    with open(log_filename, 'a') as f:
        f.write('\n'.join(map(','.join, {str(key): str(value) for key, value in hyperparameters.items()}.items())))
        f.write('\n')
        # write csv header
        f.write('Episode,Step,Total Reward,q_value_table_length\n')

    
    # initialise variables for plotting purpose
    step = 0

    for episode in range(int(1e4)):
        # reset environment before each episode
        total_reward = 0.0

        # to update our agent after each episode
        states, actions, rewards = [], [], []

        observation = env.reset()
        state = linear_parse_observation_to_state(observation)
        action = agent.get_new_action_e_greedly(state)
        done = False

        while not done:
            observation, reward, done, info = env.step(action)
            next_state = linear_parse_observation_to_state(observation)
            total_reward += reward
            next_action = agent.get_new_action_e_greedly(next_state)

            states.append(state)
            actions.append(action)
            rewards.append(reward)

            state = next_state
            action = next_action

            if done:
                with open(log_filename, 'a') as f:
                    f.write(f'{episode},{step},{total_reward},{agent.q_value_table.__len__()}\n')
                if episode % 500 == 0 and episode != 0:
                    print(f'episode {episode}')
                    play(env, agent, linear_parse_observation_to_state)
            step += 1

        agent.update(states, actions, rewards)
    env.close()
    return agent


if __name__ == '__main__':
    hyperparameters = {
        # 'env_name': 'MiniGrid-Empty-8x8-v0',
        # 'env_name': 'MiniGrid-DoorKey-5x5-v0',
        # 'env_name': 'MiniGrid-Empty-Random-6x6-v0',
        # 'env_name': 'MiniGrid-Empty-16x16-v0',
        'env_name': 'MiniGrid-DistShift1-v0',
        # 'env_name': 'MiniGrid-LavaGapS5-v0',
        # 'env_name': 'MiniGrid-SimpleCrossingS9N1-v0',
        # 'env_name': 'MiniGrid-Dynamic-Obstacles-5x5-v0',
        # 'env_name': 'MiniGrid-Dynamic-Obstacles-6x6-v0',
        'n_zero': 30,
        'learning_rate': 1e-3,
    }

    env = gym.make(hyperparameters['env_name'])
    env = ReseedWrapper(env)
    agent = train(env, hyperparameters)
