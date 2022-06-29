import numpy as np
import matplotlib.pyplot as plt

def plot_time_series(dataset):

    # TODO vlines uses the same colors for different data, colors therefore not to be manually assigned
    # Not high priority, probably not even needed

    dataset = np.array(dataset)

    for data in dataset:
        plt.vlines(data[0], 0, data[1])

    plt.xlabel("t")
    plt.ylabel(r"$s/s_{max}$")

    max_n_steps = np.amax(dataset[:, 0])
    plt.xlim(0, max_n_steps)
    plt.ylim(0, 1)

    plt.show()

def plot_size_probability(dataset, labels=[]):
    """
    Plots the size probability of multiple models.

    Arguments:
        dataset (list of tuples): Each tuple holds a x and y value.
    """
    
    dataset = np.array(dataset, dtype=object)

    for i, data in enumerate(dataset):

        if len(labels) == 0:
            label = None
        else:
            label = labels[i]

        plt.plot(data[0], data[1], label=label)

    plt.xscale('log')
    plt.yscale('log')
    
    plt.xlabel("s")
    plt.ylabel("P(s)")
    
    if len(labels) != 0:
        plt.legend()

    plt.show()

