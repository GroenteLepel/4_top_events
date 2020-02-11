import pandas as pd
import numpy as np
import pickle
import config as conf
import progress_bar.progress_bar as pb


def read_in(pickle_result: bool = False):
    # Running the read_csv command once with parameter error_bad_names=False
    #  resulted in the command skipping certain lines, because it found some
    #  that had length up to 18. This helped me figuring out what the max
    #  length of the csv file was, so I could use the names=list(range(n))
    #  parameter to forcibly read the file, filling the shorter rows with NaN.
    header = ['event ID', 'process ID', 'event weight', 'MET', 'METphi']
    objects = list(range(1, 14))

    df = pd.read_csv("data/TrainingValidationData.csv", sep=';', header=None,
                     names=header + objects)
    df = df.fillna(0)  # zero-padding the empty columns
    change_labels(df)  # change the labels from ttbar and 4top to 0 and 1
    df = df.drop(columns=['event ID', 'event weight'])

    if pickle_result:
        pickle_object(df, "df_dataset.pkl")

    return df


def change_labels(df):
    """
    Change the labels in the second column of the dataframe from ttbar and 4top
    to respectively 0 and 1.
    :param df:
    :return:
    """
    # df[1].unique() looks for all possible values in that column, this returns
    #  array(['ttbar', '4top'], dtype=object), meaning that I only have to
    #  ttbar with 0 and 4top with 1.
    df['process ID'] = df['process ID'].replace('ttbar', 0)
    df['process ID'] = df['process ID'].replace('4top', 1)


def pickle_object(obj, filename: str):
    with open("{}{}".format(conf.PICKLEJAR, filename), 'wb') as out_file:
        pickle.dump(obj, out_file)


def normalise(data: pd.DataFrame):
    """
    Normalise all the data. This replaces the old data.
    """
    print("Normalising data.")
    # Define att for readability
    normalised = \
        (data - data.min()) / \
        (data.max() - data.min())
    data = normalised.fillna(0)


def standardise(data: pd.DataFrame):
    """
    Standardise all the data. This replaces the old data.
    """
    print("Standardising data.")
    standardised = \
        (data - data.mean()) / \
        data.std()

    data = standardised.fillna(0)


def load_pickle(n_objects: int, filename: str):
    """
    Loads in objects from the picklejar with the desired filename.
    :param n_objects: amount of objects in the file to load.
    :param filename: filename of the file containing the object in the
    picklejar.
    :return: the unpickled objects.
    """
    objects = [None] * n_objects
    with open("{}{}".format(conf.PICKLEJAR, filename), 'rb') as in_file:
        for n in range(n_objects):
            objects[n] = pickle.load(in_file)

    if n_objects == 1:
        return objects[0]

    return objects


def to_four_vector(obj: str):
    """
    Makes a 4-vector of the given string object, where each value is separated
    by a comma, in the form of:
    obj, E, pt, eta, phi,
    and returns a 5D array containing the charge at the first element, then the
    four vector.
    :param obj:
    :return: 5D array
    """
    # look at the second character in the string if the object contains a
    #  charge
    if obj[1] == '+':
        charge = np.array([1])
    elif obj[1] == '-':
        charge = np.array([-1])
    else:
        charge = np.array([0])
    four_vector = np.asarray(obj.split(",")[-4:], dtype=float)

    return np.concatenate((charge, four_vector))


def generate_input_map(event):
    """
    Generates an input map based on the given event. All possible objects are:
    j, b, m, e, p (jet, b-jet, muon, electron, photon) and will coincide with
    rows 1 to 5 of the map respectively. Each row can have 8 options max.
    :return: array containing the first two values MET and METPHI, and flattened
    5x8x4 array containing the four-vectors of each type.
    """
    # the event map to fill up
    event_map = np.zeros((conf.N_PARTICLES, conf.N_BINS, conf.LEN_VECTOR))
    return_array = np.zeros(conf.SIZE_2D)
    types = ['j', 'b', 'm', 'e', 'g']  # type list to use as filter

    # generate Series object containing all the object types as characters.
    met, metphi = event[['MET', 'METphi']]
    if event[1] != 0:
        type_list = event[3:17].str.slice(stop=1)

        for i in range(len(types)):
            # find all the indices where a certain type of object is located
            type_locations = type_list[type_list == types[i]].index

            # store all found objects as four vector in event_map
            for j, at_loc in enumerate(type_locations):
                if j == conf.N_BINS:
                    break
                event_map[i][j] = to_four_vector(event[at_loc])

    return_array[:2] = np.array([met, metphi])
    return_array[conf.N_PARAMS * conf.LEN_VECTOR:] = event_map.flatten()
    return return_array


def generate_map_set(df: pd.DataFrame, save: bool = False):
    filename = "event_map2d_flattened.txt"
    dataset = \
        np.zeros((len(df), conf.SIZE_2D))
    print("Converting dataframe to acceptable array.")
    for index, row in df.iterrows():
        bar = pb.percentage_to_bar(index / len(df) * 100)
        print(bar, end='\r')
        dataset[index] = generate_input_map(row)

    if save:
        print("Saving array to file.")
        np.savetxt("data/{}".format(filename), dataset, fmt='%4e')
    else:
        return dataset


def generate_label_set(df: pd.DataFrame, save: bool = False):
    labelset = np.asarray(df['process ID'].to_numpy())
    if save:
        np.savetxt("data/labelset.txt", labelset, fmt='%1.0d')
    else:
        return labelset


def modify_data(map_set: bool = True, label_set: bool = True):
    tvd = read_in()

    if map_set:
        generate_map_set(tvd, save=True)
    if label_set:
        generate_label_set(tvd, save=True)


def load_data():
    dataset = pd.read_csv("data/event_map2d_flattened.txt", header=None, sep=' ')
    labelset = pd.read_csv("data/labelset.txt", header=None, sep=' ')
    return dataset, labelset
