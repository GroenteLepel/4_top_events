import modify_data as md
from sklearn.model_selection import train_test_split
import network_model as nm
import matplotlib.pyplot as plt

batch_size = 50000
epochs = 90

md.modify_data(label_set=False)
nm.set_gpu_growth()
ds, ls = md.load_data()

train_ds, val_ds, train_ls, val_ls = \
    train_test_split(ds[:batch_size], ls[:batch_size], train_size=0.8)

test_model = nm.init_model()
test_model.summary()

history = test_model.fit(train_ds, train_ls, epochs=epochs, validation_data=(val_ds, val_ls))

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.legend()
plt.show()
