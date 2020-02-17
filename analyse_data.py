import numpy as np
import config as conf
import pandas as pd


def read_in():
    """
    Reads in the datafile delivered in the config and reshapes it to the
    desired tensor shape of (n_points, n_particles + 1, n_bins, len_vector),
    where all these variables are defined in the config file.
    :return:
    """
    em_flat_df = pd.read_csv("{}{}".format(conf.DATA_FOLDER, conf.DATA_FILE),
                             header=None, sep=' ')
    n_data_points = len(em_flat_df.index)
    event_map = em_flat_df.to_numpy().reshape(
        (n_data_points, conf.N_PARTICLES + 1, conf.N_BINS, conf.LEN_VECTOR))
    return event_map


def calc_stats(event_map):
    """
    Calculates the lengths of the five-vectors, the standard deviation and means
    of the event map bins.
    :param event_map: matrix of shape (n, 6, 8, 5) containing the data.
    :return: list of three matrices of shape (n, 6, 8) in order lengths, stds,
    means.
    """
    lengths = np.sqrt((event_map ** 2).sum(axis=3))
    stds, means = lengths.std(axis=0), lengths.mean(axis=0)
    return lengths, stds, means


def find_outliers(event_map, distance_filter: int = 5):
    """
    Locates data that lies outside mean > distance_filter * sigma and returns
    the indices.
    :param event_map: Tensor of shape n x n_particles x n_bins x len_vector
    :param distance_filter: Integer indicating how far away the data point has
    to be located.
    :return: ndarray
    """
    length_values, std, mean = calc_stats(event_map)
    return np.where(length_values > mean + distance_filter * std)

