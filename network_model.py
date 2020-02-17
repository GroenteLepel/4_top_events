import tensorflow as tf
from tensorflow.keras import layers
import config as conf
from tensorflow.keras import models
import ipykernel  # for the verbose output of tensorflow


def init_sequential_model():
    model = tf.keras.Sequential()
    model.add(
        layers.Reshape((conf.N_PARTICLES + 1, conf.N_BINS, conf.LEN_VECTOR),
                       input_shape=(conf.SIZE_2D,)
                       ))
    model.add(layers.Conv2D(64, (1, 1),
                            activation=tf.nn.leaky_relu,
                            name='1st_feature_enhancer_64x1x1'
                            ))
    model.add(layers.Conv2D(64, (1, 1),
                            activation=tf.nn.leaky_relu,
                            name='2nd_feature_enhancer_64x1x1'
                            ))
    model.add(layers.Conv2D(128, (2, 2),
                            activation=tf.nn.leaky_relu,
                            name='2x2_features'))
    model.add(layers.Conv2D(128, (1, 1),
                            activation=tf.nn.leaky_relu,
                            name='1st_hidden_128x1x1'
                            ))
    model.add(layers.Conv2D(128, (1, 1),
                            activation=tf.nn.leaky_relu,
                            name='2nd_hidden_128x1x1'
                            ))
    model.add(layers.MaxPool2D(
        name='max_2x2'
    ))
    model.add(layers.Conv2D(56, (1, 1),
                            activation=tf.nn.leaky_relu,
                            name='3rd_hidden_56x1x1'
                            ))
    model.add(layers.Conv2D(56, (1, 1),
                            activation=tf.nn.leaky_relu,
                            name='4th_hidden_56x1x1'
                            ))
    model.add(layers.Flatten(
        name='flatten'
    ))
    model.add(layers.Dense(100,
                           activation=tf.nn.leaky_relu,
                           name='leaky_relu'
                           ))
    model.add(layers.Dense(1,
                           activation='sigmoid',
                           name='sigmoid_classification'
                           ))
    model.compile(optimizer='nadam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.summary()

    return model


def init_concatenated_model():
    # pre-channels
    # generating inputs
    input = layers.Input(shape=(conf.SIZE_2D,),
                         name='flattened_input')
    reshaped = layers.Reshape(
        (conf.N_PARTICLES + 1, conf.N_BINS, conf.LEN_VECTOR),
        name='main_reshape'
        )(input)

    # enhancing features
    enhance_1 = layers.Conv2D(48, (1, 1),
                           activation=tf.nn.leaky_relu,
                           name='1st_FE_48x1x1'
                           )(reshaped)
    enhance_2 = layers.Conv2D(48, (1, 1),
                           activation=tf.nn.leaky_relu,
                           name='2nd_FE_48x1x1'
                           )(enhance_1)

    # channel 1
    drop1 = layers.Dropout(0.5,
                           name='0.5_drop_1'
                           )(enhance_2)
    conv1 = layers.Conv2D(32, (conf.N_PARTICLES + 1, 1),
                           activation=tf.nn.leaky_relu,
                           name='CF_32x6x1'
                           )(drop1)
    pool2 = layers.MaxPool2D(pool_size=(1, 2),
                             name='max_1x2'
                             )(conv1)
    flat1 = layers.Flatten(name='flat_1'
                           )(pool2)

    # channel 2
    drop2 = layers.Dropout(0.5,
                           name='0.5_drop_2'
                           )(enhance_2)
    conv2 = layers.Conv2D(20, (1, conf.N_BINS),
                           activation=tf.nn.leaky_relu,
                           name='RF_20x1x8'
                           )(drop2)
    pool3 = layers.MaxPool2D(pool_size=(2, 1),
                             name='max_2x1'
                             )(conv2)
    flat2 = layers.Flatten(name='flat_2'
                           )(pool3)

    # channel 3
    drop3 = layers.Dropout(0.5,
                           name='0.5_drop_3'
                           )(enhance_2)
    conv3 = layers.Conv2D(24, (3, 3),
                           activation=tf.nn.leaky_relu,
                           name='GF_3x3'
                           )(drop3)
    pool3 = layers.MaxPool2D(name='max_2x2'
                             )(conv3)
    flat3 = layers.Flatten(name='flat_3'
                           )(pool3)

    # merge
    merged = layers.concatenate([flat1, flat2, flat3])

    # interpretation
    dense1 = layers.Dense(16,
                          activation=tf.nn.leaky_relu,
                          name='l_relu'
                          )(merged)
    outputs = layers.Dense(1,
                           activation='sigmoid',
                           name='sigmoid_class'
                           )(dense1)
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
