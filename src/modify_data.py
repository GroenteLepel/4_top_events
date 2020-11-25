import pandas as pd
import numpy as np
import pickle
from src import analyse_data as ad, config as conf, progress_bar as pb


def read_in(file_to_read: str = None, pickle_result: bool = False):
    """
    This function reads in the .csv data file provided for the exercise.
    :param file_to_read: string indicating the file located in the data folder
    from the config file which needs to be read.
    :param pickle_result: boolean indicated if the resulting modified DataFrame
    must be pickled.
    :return: DataFrame object containing the labels, missing energies and
    observed objects.
    """
    # Running the read_csv command once with parameter error_bad_names=False
    #  resulted in the command skipping certain lines, because it found some
    #  that had length up to 18. This helped me figuring out what the max
    #  length of the csv file was, so I could use the names=list(range(n))
    #  parameter to forcibly read the file, filling the shorter rows with NaN.
    header = ['event ID', 'process ID', 'event weight', 'MET', 'METphi']
    objects = list(range(1, 14))

    if file_to_read is None:
        file_to_read = "TrainingValidationData.csv"

    df = pd.read_csv("{}{}".format(conf.DATA_FOLDER, file_to_read),
                     sep=';', header=None,
                     names=header + objects)
    df = df.fillna(0)  # zero-padding the empty columns, replacing NaN's with 0
    change_labels(df)  # change the labels from ttbar and 4top to 0 and 1
    df = df.drop(columns=['event ID', 'event weight'])

    if pickle_result:
        pickle_object(df, "df_dataset.pkl")

    return df


def change_labels(df):
    """
    Change the labels in the second column of the dataframe from ttbar and 4top
    to respectively 0 and 1.
    :param df: pandas.DataFrame object which contains the labels defined by
    'process ID'. Locates the 'ttbar' and '4top' events and replaces these with
    0s and 1s, respectively.
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
    j, b, m, e, g (jet, b-jet, muon, electron, photon) and will coincide with
    rows 1 to 5 of the map respectively. Each row can have 8 options max.
    :return: array containing the first two values MET and METPHI, and flattened
    5x8x4 array containing the four-vectors of each type.
    """
    # the event map to fill up
    event_map = np.zeros((conf.N_PARTICLES+1, conf.N_BINS, conf.LEN_VECTOR))
    types = ['j', 'b', 'm', 'e', 'g']  # type list to use as filter

    # store the missing energies in the first two categories
    event_map[0, 0, 0], event_map[0, 1, 0] = event[['MET', 'METphi']]

    # generate Series object containing all the object types as characters, and
    #  check if there is any object observed, otherwise the observation is empty
    if event[1] != 0:
        type_list = event[3:17].str.slice(stop=1)

        for i in range(len(types)):
            # find all the indices where a certain type of object is located
            type_locations = type_list[type_list == types[i]].index

            # store all found objects as five vector in event_map
            for j, at_loc in enumerate(type_locations):
                if j == conf.N_BINS:
                    # throw away items if bin capacity has been reached
                    break
                event_map[i+1][j] = to_four_vector(event[at_loc])

    return event_map.flatten()


def generate_map_part(df: pd.DataFrame):
    data_set = np.zeros((len(df), conf.SIZE_2D))

    for index, row in df.iterrows():
        bar = pb.percentage_to_bar(index / len(df) * 100)
        print(bar, end='\r')
        data_set[index] = generate_input_map(row)
    return data_set


def generate_map_set(df: pd.DataFrame,
                     save: bool = False, file_name: str = None):
    print("Converting dataframe to acceptable array.")
    dataset = generate_map_part(df)

    if save:
        print("Saving array to file.")
        if file_name is None:
            path = conf.DATA_FILE
        else:
            path = "{}{}".format(conf.DATA_FOLDER, file_name)
        np.savetxt("{}".format(path), dataset, fmt='%4e')
        print("Saved {}".format(path))
    else:
        return dataset


def generate_label_set(df: pd.DataFrame, save: bool = False):
    labelset = np.asarray(df['process ID'].to_numpy())
    if save:
        np.savetxt("{}".format(conf.LABEL_FILE), labelset, fmt='%1.0d')
        print("Saved {}".format(conf.LABEL_FILE))
    else:
        return labelset


def remove_outliers():
    event_map = ad.read_in()

    to_remove = ad.find_outliers(event_map)
    ds, ls = load_data()
    ds_filtered = np.delete(ds.to_numpy(), to_remove[0], axis=0)
    ls_filtered = np.delete(ls.to_numpy(), to_remove[0], axis=0)
    np.savetxt("{}_filtered.txt".format(conf.DATA_FILE),
               ds_filtered, fmt='%4e')
    np.savetxt("{}_filtered.txt".format(conf.LABEL_FILE),
               ls_filtered, fmt='%1.0d')


def modify_data(tvd: pd.DataFrame = None,
                map_set: bool = True, label_set: bool = True):
    if tvd is None:
        tvd = read_in()

    if map_set:
        generate_map_set(tvd, save=True)
    if label_set:
        generate_label_set(tvd, save=True)


def load_data():
    dataset = pd.read_csv(conf.DATA_FILE, header=None, sep=' ')
    labelset = pd.read_csv(conf.LABEL_FILE, header=None, sep=' ')
    return dataset, labelset
