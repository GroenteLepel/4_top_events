import modify_data as md
import numpy as np
import config as conf
import pandas as pd
import matplotlib.pyplot as plt


def read_in():
    em_flat = np.loadtxt("data/event_map_filtered.txt", delimiter=' ')
    n_data_points = len(em_flat) / conf.SIZE_2D
    event_map = em_flat.reshape(
        (n_data_points, conf.N_PARTICLES + 1, conf.N_BINS, conf.LEN_VECTOR))
    return event_map


def calc_stats(event_map):
    lengths = np.sqrt((event_map ** 2).sum(axis=3))
    stds, means = lengths.std(axis=0), lengths.mean(axis=0)
    return lengths, stds, means


def plot_histograms(length_vectors, stds, means, n_bins=100):
    fig, ax = plt.subplots(6, 8, figsize=(20, 10))
    bins = np.zeros((6, 8, n_bins))
    for i in range(6):
        for j in range(8):
            if i == 0:
                ax[i][j].set_title(j)
            if j == 0:
                ax[i][j].set_ylabel(i).set_rotation(0)
            ax[i][j].set_xticks([])
            ax[i][j].set_yticks([])
            if stds[i, j] == 0 and means[i, j] == 0:
                ax[i][j].annotate("Empty.", (0, 0), va='center', ha='center',
                                  fontsize=20)
                ax[i][j].set_xlim(-1, 1)
                ax[i][j].set_ylim(-1, 1)
                continue
            non_zero = length_vectors[:, i, j] != 0
            bins[i, j] = \
                ax[i][j].hist(length_vectors[non_zero, i, j], bins=n_bins,
                              color='black')[0]
            ax[i, j].set_title("n = {}".format(bins[i, j].sum()))

    fig.tight_layout()
    fig.show()


def find_outliers(event_map, at_particle: int, at_bin: int, distance_filter: int):
    length_values, std, mean = calc_stats(event_map)
    return np.where(
        length_values[:, at_particle, at_bin] > mean[at_particle, at_bin] + \
        distance_filter * std[at_particle, at_bin]
    )

