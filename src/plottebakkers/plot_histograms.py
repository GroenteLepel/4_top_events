import matplotlib.pyplot as plt
import numpy as np
from src import config as conf


def plot(length_vectors, stds, means, n_bins: int = 50,
         save: bool = False, file_name: str = None):
    plt.rcParams.update({'font.size': 18})
    fig, ax = plt.subplots(conf.N_PARTICLES + 1, conf.N_BINS, figsize=(20, 10))
    bins = np.zeros((conf.N_PARTICLES + 1, conf.N_BINS, n_bins))
    for i in range(conf.N_PARTICLES + 1):
        for j in range(conf.N_BINS):
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
            ax[i, j].set_title("n = {0:d}".format(int(bins[i, j].sum())))

    fig.tight_layout()
    if save:
        if file_name is None:
            file_name = 'histograms.png'
        fig.savefig('{}{}'.format(conf.DATA_FOLDER, file_name))
    else:
        fig.show()
