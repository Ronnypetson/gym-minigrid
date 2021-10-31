from project_RL.plot import plot
from project_RL.linear_sarsa.sarsa_lambda_agent import LinearSarsaLambda
from gym_minigrid.wrappers import *
from datetime import datetime as dt
from project_RL.play import play
from project_RL.linear_parsing import parse_observation_to_state
import dill
import pickle


def train(env, hyperparameters, num_episodes=int(1e2)):
    """ Train a sarsa lambda agent in the requested environment

    Arguments:
        hyperparameters dictionary containing:
            - env_name
            - discount_rate
            - learning_rate
            - lambda
            - n0 (initial exploration rate, it decays as the number of visits to a state increases)
    """
    agent = LinearSarsaLambda(env, hyperparameters['discount_rate'],
                            hyperparameters['learning_rate'],
                            hyperparameters['epsilon'],
                            hyperparameters['lambda'],
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
                with open(log_filename, 'a') as f:
                    f.write(f'{episode},{step},{total_reward},{agent.q_value_table.__len__()}\n')
                if episode % 500 == 0:
                    play(env, agent)
            step += 1
    env.close()

    with open(f'agent_{log_filename[:-4]}.pickle', 'wb') as f:
        dill.dump(agent, f, pickle.HIGHEST_PROTOCOL)

    plot(log_filename[:-4])  # filename without extension

    return agent


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
        # 'env_name': 'MiniGrid-DoorKeyObst-7x7-v0',
        'discount_rate': 0.9,
        'learning_rate': 1e-3,
        'lambda': 0.9,
        'epsilon': 0.3,
        'n0': None
    }

    env = ReseedWrapper(gym.make(hyperparameters['env_name']))
    agent = train(env, hyperparameters, num_episodes=int(2e3))
    # It looks like env.close() inside "train" frees all resources,
    # requiring us to re-create the environment. Not closing it though 
    # makes the subsequent "plot" call unusable.
    env = ReseedWrapper(gym.make(hyperparameters['env_name']))
    play(env, agent, episodes=1)
