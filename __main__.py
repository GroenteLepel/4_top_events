import modify_data as md
import config as conf
import tensorflow as tf
from tensorflow.keras.models import load_model

# This script assumes that the data to test the trained model on looks similar
#  to the data provided in the exercise.

# use md.read_in() to read in the .csv file
unmodified_data = md.read_in()

# writes the modified data and labels to two separate files
md.modify_data(unmodified_data)

# after modify_data() has run, the load_data() command loads in the saved files
data, labels = md.load_data()

# load in the desired model
# to_load = "sequential_model.h5"
to_load = "concatenated_model.h5"
model = load_model("{}{}".format(conf.DATA_FOLDER, to_load),
                   custom_objects={'leaky_relu': tf.nn.leaky_relu})
model.summary()

# chose whether to evaluate or predict
results = model.evaluate(data, labels)
# predictions = model.predict(data)
