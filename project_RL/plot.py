import os
import csv
import matplotlib.pyplot as plt
import numpy as np


def plot(log_filename, block_size=10):
    """Plots rewards from csv of train data"""
    with open(f'{log_filename}.csv', 'r') as file:
        # process hyperparameters from first rows
        firstrow = file.readline().replace('\n', '').split(',')
        hyperparameters_size = int(firstrow[1])
        hyperparameters = ""
        for i in range(hyperparameters_size):
            # row = file.readline().replace('\n', '').split(',')
            # hyperparameters[row[0]] = row[1]
            hyperparameters += file.readline().replace(',', ':')

        # process the remaining lines, plot reward
        csv_file = csv.DictReader(file)

        rewards = [float(row['Total Reward']) for row in csv_file]
        # rewards length must be multiple of block size,
        if len(rewards) % block_size != 0:
            # discard last incomplete block
            rewards = rewards[:-(len(rewards) % block_size)]

        # reshape array in block size defined above
        average_in_blocks = np.mean(np.array(rewards).reshape(-1, block_size), axis=1)
        plt.plot(average_in_blocks, '.', alpha=.5)
        plt.ylabel('avg reward')
        plt.xlabel(f'batch {block_size} episode')
        plt.gcf().text(0.02, 0.0,  hyperparameters, fontsize=8)
        plt.subplots_adjust(bottom=0.3)  # make room for text
        plt.savefig(f'{log_filename}.png')
        plt.show()


if __name__ == '__main__':
    plot(os.path.join(os.path.dirname(__file__), 'sarsa/trained_agent/log_MiniGrid-DoorKeyObst-7x7-v0_21-10-26-14-37-21'), 100)

