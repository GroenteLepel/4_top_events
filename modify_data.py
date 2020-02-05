import pandas as pd
import pickle
from config import PICKLEJAR


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

