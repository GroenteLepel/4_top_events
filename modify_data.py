import pandas as pd
import numpy as np
import pickle
from config import PICKLEJAR
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
                      names=header+objects)
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
    return df


def pickle_object(obj, filename: str):
    with open("{}{}".format(PICKLEJAR, filename), 'wb') as out_file:
        pickle.dump(obj, out_file)


def load_pickle(n_objects: int, filename: str):
    """
    Loads in objects from the picklejar with the desired filename.
    :param n_objects: amount of objects in the file to load.
    :param filename: filename of the file containing the object in the
    picklejar.
    :return: the unpickled objects.
    """
    objects = [None] * n_objects
    with open("{}{}".format(PICKLEJAR, filename), 'rb') as in_file:
        for n in range(n_objects):
            objects[n] = pickle.load(in_file)

    if n_objects == 1:
        return objects[0]

    return objects


def to_four_vector(obj: str):
    """
    Makes a 4-vector of the given string object, where each value is separated
    by a comma, in the form of:
    obj, E, pt, eta, phi
    :param obj:
    :return: 4D array
    """
    return np.asarray(obj.split(",")[-4:], dtype=float)


def generate_input_map(event):
    """
    Generates an input map based on the given event. All possible objects are:
    j, b, m, e, g (jet, b-jet, muon, electron, gluon) and will coincide with
    rows 1 to 5 of the map respectively. Each row can have 8 options max.
    :return: array containing the first two values MET and METPHI, and flattened
    5x8x4 array containing the four-vectors of each type.
    """
    event_map = np.zeros((5, 8, 4))  # the event map to fill up
    return_array = np.zeros(2 + 5 * 8 * 4)
    # TODO: does not distinguish e- and e+, m- and m+ etc. to do?
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
                if j == 8:
                    break
                event_map[i][j] = to_four_vector(event[at_loc])

    return_array[:2] = np.array([met, metphi])
    return_array[2:] = event_map.flatten()
    return return_array


def generate_map_set(df: pd.DataFrame, save: bool = False):
    dataset = np.zeros((len(df), (2 + 5 * 8 * 4)))
    print("Converting dataframe to acceptable array.")
    for index, row in df.iterrows():
        bar = pb.percentage_to_bar(index / len(df) * 100)
        print(bar, end='\r')
        dataset[index] = generate_input_map(row)

    if save:
        print("Saving array to file.")
        np.savetxt("data/event_map_flattened.txt", dataset, fmt='%4e')
    else:
        return dataset


def generate_label_set(df: pd.DataFrame, save: bool = False):
    labelset = df['process ID'].as_matrix()
    if save:
        np.savetxt("data/labelset.txt", labelset, fmt='%1.0d')
    else:
        return labelset
