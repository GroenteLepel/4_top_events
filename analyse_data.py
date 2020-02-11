import modify_data as md
import numpy as np
import config as conf
import pandas as pd


def read_in():
    em_flat = np.loadtxt("data/event_map2d_flattened.txt", delimiter=' ')
    event_map = em_flat.reshape(
        (100000, conf.N_PARTICLES + 1, conf.N_BINS, conf.LEN_VECTOR))
    return event_map


event_map = read_in()
lengths = np.sqrt((event_map ** 2).sum(axis=3))
stds, means = lengths.std(axis=0), lengths.mean(axis=0)

(lengths[:][1][0] < means[1][0] - 3 * stds[1][0]).sum()
(lengths[:][1][0] > means[1][0] + 3 * stds[1][0]).sum()