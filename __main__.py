import modify_data as md
from sklearn.model_selection import train_test_split
import network_model as nm
import matplotlib.pyplot as plt
import time

batch_size = 50000
epochs = 40


def run_training():
    nm.set_gpu_growth()
    ds, ls = md.load_data()

    train_ds, val_ds, train_ls, val_ls = \
        train_test_split(ds[:batch_size], ls[:batch_size], train_size=0.8)

    test_model = nm.init_model_2d()

    history = test_model.fit(train_ds, train_ls, epochs=epochs,
                             validation_data=(val_ds, val_ls))

    current_time = time.gmtime()
    plt.title("{}:{},{}".format(current_time.tm_hour, current_time.tm_min,
                                current_time.tm_sec))
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='val')
    plt.legend()
    plt.show()


# md.modify_data(label_set=False)
run_training()
