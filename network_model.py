import pandas as pd
import modify_data as md
import tensorflow as tf
from tensorflow.keras import layers
import config as conf
from tensorflow.keras import models
import ipykernel  # for the verbose output of tensorflow


def init_model_1d():
    model = tf.keras.Sequential()
    model.add(layers.Reshape((conf.SIZE_1D, 1), input_shape=(conf.SIZE_1D,)
                             ))
    model.add(layers.Conv1D(64, 24,
                            activation=tf.nn.leaky_relu
                            ))
    model.add(layers.Conv1D(32, 12,
                            activation='tanh'
                            ))
    model.add(layers.Flatten())
    model.add(layers.Dense(16,
                           activation=tf.nn.leaky_relu
                           ))
    model.add(layers.Dense(1,
                           activation='sigmoid'
                           ))
    model.compile(optimizer='nadam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def init_model_2d():
    model = tf.keras.Sequential()
    model.add(
        layers.Reshape((conf.N_PARTICLES + 1, conf.N_BINS, conf.LEN_VECTOR),
                       input_shape=(conf.SIZE_2D,)
                       ))
    model.add(layers.Conv2D(64, (1, 1),
                            activation=tf.nn.leaky_relu
                            ))
    model.add(layers.Conv2D(64, (1, 1),
                            activation=tf.nn.leaky_relu
                            ))
    model.add(layers.Conv2D(128, (2, 2)))
    model.add(layers.Conv2D(128, (1, 1),
                            activation=tf.nn.leaky_relu
                            ))
    model.add(layers.Conv2D(128, (1, 1),
                            activation=tf.nn.leaky_relu
                            ))
    model.add(layers.MaxPool2D())
    model.add(layers.Conv2D(56, (1, 1),
                            activation=tf.nn.leaky_relu
                            ))
    model.add(layers.Conv2D(56, (1, 1),
                            activation=tf.nn.leaky_relu
                            ))
    model.add(layers.Flatten())
    model.add(layers.Dense(100,
                           activation=tf.nn.leaky_relu
                           ))
    model.add(layers.Dense(1,
                           activation='sigmoid'
                           ))
    model.compile(optimizer='nadam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.summary()

    return model


def init_concatenated_model():
    # generating inputs
    input = layers.Input(shape=(conf.SIZE_2D,))
    reshaped = layers.Reshape(
        (conf.N_PARTICLES + 1, conf.N_BINS, conf.LEN_VECTOR),
        )(input)

    # channel 1
    conv11 = layers.Conv2D(64, (1, 1), activation=tf.nn.leaky_relu)(reshaped)
    conv12 = layers.Conv2D(64, (1, 1), activation=tf.nn.leaky_relu)(conv11)
    drop1 = layers.Dropout(0.5)(conv12)
    pool1 = layers.MaxPool2D()(drop1)
    flat1 = layers.Flatten()(pool1)

    # channel 2
    conv21 = layers.Conv2D(32, (conf.N_PARTICLES + 1, 1), activation=tf.nn.leaky_relu)(
        reshaped)
    drop2 = layers.Dropout(0.5)(conv21)
    pool2 = layers.MaxPool2D(pool_size=(1, 2))(drop2)
    flat2 = layers.Flatten()(pool2)

    # channel 3
    conv31 = layers.Conv2D(20, (1, conf.N_BINS), activation=tf.nn.leaky_relu)(reshaped)
    drop3 = layers.Dropout(0.5)(conv31)
    pool3 = layers.MaxPool2D(pool_size=(2, 1))(drop3)
    flat3 = layers.Flatten()(pool3)

    # channel 4
    conv41 = layers.Conv2D(24, (3, 3), activation=tf.nn.leaky_relu)(drop1)
    pool4 = layers.MaxPool2D()(conv41)
    flat4 = layers.Flatten()(pool4)

    # merge
    merged = layers.concatenate([flat1, flat2, flat3, flat4])

    # interpretation
    dense1 = layers.Dense(16, activation=tf.nn.leaky_relu)(merged)
    outputs = layers.Dense(1, activation='sigmoid')(dense1)
    model = models.Model(inputs=input, outputs=outputs)

    # compile
    model.compile(loss='binary_crossentropy', optimizer='nadam',
                  metrics=['accuracy'])

    # summarize
    print(model.summary())
    return model


def set_gpu_growth():
    """
    If you are using the GPU that also runs the system, you don't want
    tensorflow to commandeer the entire memory, so this should prevent that.
    Fixes the error "Could not create cudnn handle: CUDNN_STATUS_ALLOC_FAILED"
    """
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
