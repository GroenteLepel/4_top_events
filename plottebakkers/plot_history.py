import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import History
import config as conf


def plot(hist: History,
         fig_title: str = None, save: bool = False, file_name: str = None):

    if fig_title is None:
        fig_title = 'loss per epoch'

    fig = plt.figure(figsize=(3, 3))
    ax = fig.add_subplot()
    ax.set_yscale('log')
    ax.plot(hist.history['loss'], label='training', c='black', ls='-')
    ax.plot(hist.history['val_loss'], label='validation', c='black', ls='--')
    ax.legend()

    # setting labels
    ax.set_xlabel('# epochs')
    # ax.set_ylabel('binary cross-entropy loss')
    ax.set_title(fig_title)

    fig.tight_layout()
    if save:
        if file_name is None:
            file_name = 'history.png'
        fig.savefig('{}{}'.format(conf.DATA_FOLDER, file_name))
    else:
        fig.show()
