import pandas as pd
import modify_data as md
import tensorflow as tf
from tensorflow.keras import layers


def init_model():
    model = tf.keras.Sequential()
    model.add(layers.Reshape((162, 1), input_shape=(162,)
                             ))
    # model.add(layers.Conv1D(128, 48,
    #                         activation='relu',
    #                         padding='same',
    #                         ))
    # model.add(layers.MaxPool1D(2
    #                            ))
    model.add(layers.Conv1D(64, 24,
                            activation='relu'
                            # data_format='channels_first',
                            # input_shape=(1, 162),
                            ))
    # model.add(layers.MaxPool1D(2
    #                            ))
    # model.add(layers.Dropout(0.2))
    model.add(layers.Conv1D(32, 12,
                            activation='tanh'
                            ))
    model.add(layers.Flatten())
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='nadam',
                  loss='mean_squared_error',
                  metrics=['accuracy'])
    return model

def set_gpu_growth():
    """
    If you are using the GPU that also runs the system, you don't want
    tensorflow to commandeer the entire memory, so this should prevent that.
    Fixes the error "Could not create cudnn handle: CUDNN_STATUS_ALLOC_FAILED"
    """
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)

