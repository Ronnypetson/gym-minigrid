from project_RL.sarsa.sarsa_lambda_agent import SarsaLambda
from gym_minigrid.wrappers import *
from time import time


def train(env, hyperparameters):
    """ Train a sarsa lambda agent in the requested environment

    Arguments:
        hyperparameters dictionary containing:
            - env_name
            - discount_rate
            - learning_rate
            - epsilon
            - lambda
    """
    agent = SarsaLambda(env, hyperparameters['discount_rate'],
                        hyperparameters['learning_rate'],
                        hyperparameters['epsilon'],
                        hyperparameters['lambda'])

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

        agent.init_eligibility_table()
        observation = env.reset()
        state = parse_observation_to_state(observation)
        action = agent.get_new_action_e_greedly(state)
        done = False

        while not done:
            observation, reward, done, info = env.step(action)
            next_state = parse_observation_to_state(observation)
            total_reward += reward
            next_action = agent.get_new_action_e_greedly(next_state)
            agent.update(state, action, reward, next_state, next_action, done)

            state = next_state
            action = next_action

            if done:
                with open(log_filename, 'a') as f:
                    f.write(f'{episode},{step},{total_reward},{agent.q_value_table.__len__()}\n')
                if episode % 100 == 0:
                    play(env, agent, log_filename)
            step += 1
    env.close()
    return agent


def parse_observation_to_state(observation):
    return tuple([tuple(observation["image"].flatten()),
                  observation["direction"]])


def play(env, agent, log_filename, episodes=1):
    for episode in range(episodes):
        # reset environment before each episode
        observation = env.reset()
        state = parse_observation_to_state(observation)
        action = agent.get_new_action_greedly(state)
        done = False
        total_reward = 0

        env.render()
        while not done:
            observation, reward, done, info = env.step(action)
            env.render()
            next_state = parse_observation_to_state(observation)
            total_reward += reward
            action = agent.get_new_action_greedly(next_state)

        # write result to csv log
        with open(log_filename, 'a') as f:
            f.write(f'-1,-1,{total_reward},{agent.q_value_table.__len__()}\n')


if __name__ == '__main__':
    hyperparameters = {
        # 'env_name': 'MiniGrid-Empty-5x5-v0',
        # 'env_name': 'MiniGrid-DoorKey-8x8-v0',
        'env_name': 'MiniGrid-Empty-Random-6x6-v0',
        # 'env_name': 'MiniGrid-Empty-16x16-v0',
        # 'env_name': 'MiniGrid-DistShift1-v0',
        # 'env_name': 'MiniGrid-LavaGapS5-v0',
        # 'env_name': 'MiniGrid-SimpleCrossingS9N1-v0',
        # 'env_name': 'MiniGrid-Dynamic-Obstacles-5x5-v0',
        # 'env_name': 'MiniGrid-Dynamic-Obstacles-Random-6x6-v0',
        'discount_rate': 0.9,
        'learning_rate': 0.01,
        'lambda': 0.9,
        'epsilon': 0.3
    }

    env = gym.make(hyperparameters['env_name'])
    agent = train(env, hyperparameters)
