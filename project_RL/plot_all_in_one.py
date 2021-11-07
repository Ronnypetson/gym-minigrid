import csv
import matplotlib.pyplot as plt
import numpy as np

# download results-for-plotting from driving if you want to generate new images
def plot_all_in_one(results, block_size=10, title='plot_all_in_one', truncate_rewards_len_at=int(2e4)):
    plt.figure(figsize=(10, 5))

    rewards_results = {}
    for algorithm, log_file in results.items():
        rewards_results[algorithm] = process_file(log_file, block_size, truncate_rewards_len_at)
        plot(algorithm, rewards_results[algorithm], block_size)

    # additional plot config
    img_name = title.replace(' ', '_')
    plt.title(title)
    plt.savefig(f'{img_name}.png')
    plt.show()


def process_file(log_filename, block_size, truncate_rewards_len_at):
    """Plots rewards from csv of train data"""
    with open(log_filename, 'r') as file:
        # extract hyperparameters from first rows
        firstrow = file.readline().replace('\n', '').split(',')
        hyperparameters_size = int(firstrow[1])
        hyperparameters = ""
        for i in range(hyperparameters_size):
            hyperparameters += file.readline().replace(',', ':')

        # process the remaining lines, plot reward
        csv_file = csv.DictReader(file)

        rewards = [float(row['Total Reward']) for row in csv_file]
        rewards = rewards[:truncate_rewards_len_at]
        # rewards length must be multiple of block size,
        if len(rewards) % block_size != 0:
            # discard last incomplete block
            rewards = rewards[:-(len(rewards) % block_size)]

        # reshape array in block size defined above
        average_in_blocks = np.mean(np.array(rewards).reshape(-1, block_size), axis=1)
        return average_in_blocks


def plot(label, average_in_blocks, block_size):
    plt.plot(average_in_blocks, alpha=.9, label=label)
    plt.ylabel('avg reward')
    plt.xlabel(f'batch {block_size} episodes')
    plt.legend(loc="lower right")


def plot_tabular_deterministic_results():
    tabular_results = {
        # label: path
        'Monte Carlo': 'results/Monte-Carlo/deterministic/log_MiniGrid-DoorKeyNoObst-6x6-v0_1636138012.1325028.csv',
        'Q-Learning': 'results/Q-Learning/deterministic/log_MiniGrid-DoorKeyNoObst-6x6-v0_21-11-06-18-11-02.csv',
        'Sarsa λ 1': 'results/Sarsa-lambda/deterministic/lambda1/log_MiniGrid-DoorKeyNoObst-6x6-v0_21-11-06-14-19-13.csv',
        'Sarsa λ 0.8': 'results/Sarsa-lambda/deterministic/lambda08/log_MiniGrid-DoorKeyNoObst-6x6-v0_21-11-04-12-18-51.csv',
        'Sarsa λ 0.6': 'results/Sarsa-lambda/deterministic/lambda06/log_MiniGrid-DoorKeyNoObst-6x6-v0_21-11-04-12-19-25.csv',
        'Sarsa λ 0.4': 'results/Sarsa-lambda/deterministic/lambda04/log_MiniGrid-DoorKeyNoObst-6x6-v0_21-11-04-12-20-09.csv',
        'Sarsa λ 0.2': 'results/Sarsa-lambda/deterministic/lambda02/log_MiniGrid-DoorKeyNoObst-6x6-v0_21-11-06-14-18-59.csv',
        'Sarsa λ 0': 'results/Sarsa-lambda/deterministic/lambda0/log_MiniGrid-DoorKeyNoObst-6x6-v0_21-11-04-12-21-15.csv',
    }

    plot_all_in_one(tabular_results,
                    block_size=100,
                    title='Tabular RL algorithms on deterministic environment',
                    truncate_rewards_len_at=int(2e4))


def plot_linear_deterministic_results():
    linear_results = {
        # label: path
        'Monte Carlo': 'results/Monte-Carlo-Linear/deterministic/log_MiniGrid-DoorKeyNoObst-6x6-v0_1635909286.187907.csv',
        'Q-Learning': 'results/Q-Learning-linear/deterministic/log_MiniGrid-DoorKeyNoObst-6x6-v0_21-11-06-18-17-43.csv',
        'Sarsa λ 1': 'results/Sarsa-lambda-linear/deterministic/lambda1/log_MiniGrid-DoorKeyNoObst-6x6-v0_21-11-05-20-12-33.csv',
        'Sarsa λ 0.8': 'results/Sarsa-lambda-linear/deterministic/lambda08/log_MiniGrid-DoorKeyNoObst-6x6-v0_21-11-05-20-12-02.csv',
        'Sarsa λ 0.6': 'results/Sarsa-lambda-linear/deterministic/lambda06/log_MiniGrid-DoorKeyNoObst-6x6-v0_21-11-05-20-11-53.csv',
        'Sarsa λ 0.4': 'results/Sarsa-lambda-linear/deterministic/lambda04/log_MiniGrid-DoorKeyNoObst-6x6-v0_21-11-05-20-11-47.csv',
        'Sarsa λ 0.2': 'results/Sarsa-lambda-linear/deterministic/lambda02/log_MiniGrid-DoorKeyNoObst-6x6-v0_21-11-05-20-11-40.csv',
        'Sarsa λ 0': 'results/Sarsa-lambda-linear/deterministic/lambda0/log_MiniGrid-DoorKeyNoObst-6x6-v0_21-11-05-20-11-34.csv',

    }
    plot_all_in_one(linear_results,
                    block_size=100,
                    title='Linear Function Approximator RL algorithms on deterministic environment',
                    truncate_rewards_len_at=int(2e4))


def plot_tabular_stochastic_results():
    tabular_results = {
        # label: path
        'Monte Carlo': 'results/Monte-Carlo/stochastic/log_MiniGrid-DoorKeyObst-6x6-v0_1636123388.2562478.csv',
        'Q-Learning': 'results/Q-Learning/stochastic/log_MiniGrid-DoorKeyObst-6x6-v0_21-11-01-23-36-01.csv',
        'Sarsa λ 1': 'results/Sarsa-lambda/stochastic/lambda1/log_MiniGrid-DoorKeyObst-6x6-v0_21-11-04-22-13-18.csv',
        'Sarsa λ 0.8': 'results/Sarsa-lambda/stochastic/lambda08/log_MiniGrid-DoorKeyObst-6x6-v0_21-11-03-17-47-27.csv',
        'Sarsa λ 0.6': 'results/Sarsa-lambda/stochastic/lambda06/log_MiniGrid-DoorKeyObst-6x6-v0_21-11-06-08-23-03.csv',
        'Sarsa λ 0.4': 'results/Sarsa-lambda/stochastic/lambda04/log_MiniGrid-DoorKeyObst-6x6-v0_21-11-05-20-46-19.csv',
        'Sarsa λ 0.2': 'results/Sarsa-lambda/stochastic/lambda02/log_MiniGrid-DoorKeyObst-6x6-v0_21-11-05-10-48-41.csv',
        'Sarsa λ 0': 'results/Sarsa-lambda/stochastic/lambda0/log_MiniGrid-DoorKeyObst-6x6-v0_21-11-04-20-09-00.csv',

    }
    plot_all_in_one(tabular_results,
                    block_size=1000,
                    title='Tabular RL algorithms on stochastic environment',
                    truncate_rewards_len_at=int(1e5))


def plot_linear_stochastic_results():
    linear_results = {
        # label: path
        'Monte Carlo': 'results/Monte-Carlo-Linear/stochastic/log_MiniGrid-DoorKeyObst-6x6-v0_1635872479.1784768.csv',
        'Q-Learning': 'results/Q-Learning-linear/stochastic/log_MiniGrid-DoorKeyObst-6x6-v0_21-11-02-12-25-45.csv',
        'Sarsa λ 1': 'results/Sarsa-lambda-linear/stochastic/lambda1/log_MiniGrid-DoorKeyObst-6x6-v0_21-11-05-02-39-11.csv',
        'Sarsa λ 0.8': 'results/Sarsa-lambda-linear/stochastic/lambda08/log_MiniGrid-DoorKeyObst-6x6-v0_21-11-02-19-13-00.csv',
        'Sarsa λ 0.6': 'results/Sarsa-lambda-linear/stochastic/lambda06/log_MiniGrid-DoorKeyObst-6x6-v0_21-11-02-23-10-52.csv',
        'Sarsa λ 0.4': 'results/Sarsa-lambda-linear/stochastic/lambda04/log_MiniGrid-DoorKeyObst-6x6-v0_21-11-03-09-30-02.csv',
        'Sarsa λ 0.2': 'results/Sarsa-lambda-linear/stochastic/lambda02/log_MiniGrid-DoorKeyObst-6x6-v0_21-11-03-21-24-25.csv',
        'Sarsa λ 0': 'results/Sarsa-lambda-linear/stochastic/lambda0/log_MiniGrid-DoorKeyObst-6x6-v0_21-11-04-12-40-22.csv',

    }
    plot_all_in_one(linear_results,
                    block_size=1000,
                    title='Linear Function Approximator RL algorithms on stochastic environment',
                    truncate_rewards_len_at=int(1e5))


if __name__ == '__main__':
    plot_tabular_deterministic_results()
    plot_linear_deterministic_results()
    plot_tabular_stochastic_results()
    plot_linear_stochastic_results()
