from project_RL.linear_sarsa.sarsa_lambda_agent import LinearSarsaLambda
from gym_minigrid.wrappers import *
from time import time
from project_RL.linear_sarsa.parsing import parse_observation_to_state
from matplotlib import pyplot as plt


def train(env, hyperparameters, num_episodes=int(1e2)):
    """ Train a sarsa lambda agent in the requested environment

    Arguments:
        hyperparameters dictionary containing:
            - env_name
            - discount_rate
            - learning_rate
            - epsilon
            - lambda
    """
    agent = LinearSarsaLambda(env,
                              hyperparameters['discount_rate'],
                              hyperparameters['initial_learning_rate'],
                              hyperparameters['epsilon'],
                              hyperparameters['lambda'],
                              hyperparameters['n0'])

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
    all_rewards = []
    all_means = []
    for episode in range(num_episodes):
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
                all_rewards.append(total_reward)
                all_means.append(np.mean(all_rewards[-50:]))

                if episode % 50 == 0:
                    plt.plot(all_means, 'b.')
                    plt.draw()
                    plt.pause(0.0001)
                    plt.clf()

                with open(log_filename, 'a') as f:
                    f.write(f'{episode},{step},{total_reward},{agent.q_value_table.__len__()}\n')
                # if episode % 100 == 99:
                #     play(env, agent, log_filename)
            step += 1
    env.close()
    return agent


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
        'env_name': 'MiniGrid-DoorKey-5x5-v0',
        # 'env_name': 'MiniGrid-Empty-Random-6x6-v0',
        # 'env_name': 'MiniGrid-Empty-16x16-v0',
        # 'env_name': 'MiniGrid-DistShift1-v0',
        # 'env_name': 'MiniGrid-LavaGapS5-v0',
        # 'env_name': 'MiniGrid-SimpleCrossingS9N1-v0',
        # 'env_name': 'MiniGrid-Dynamic-Obstacles-5x5-v0',
        # 'env_name': 'MiniGrid-Dynamic-Obstacles-Random-6x6-v0',
        # 'env_name': 'Pong-ram-v0',
        'discount_rate': 0.99,
        'initial_learning_rate': 1e-2,
        'lambda': 0.9,
        'epsilon': 0.3,
        'n0': 10
    }

    env = ReseedWrapper(gym.make(hyperparameters['env_name']))
    plt.ion()
    agent = train(env, hyperparameters, num_episodes=int(2e3))
    input()
    play(env, agent, '')
