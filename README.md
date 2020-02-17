# 4_top_events
Designing a machine learning model which can separate background from 4 top event signals from simulated data from the Large Hadron Collider.

# How to use
1. (optional) open the config.py file to adjust the folders to liking, and rename the __files if you like. If you adjust this to either event_map_from_exercise_filtered.txt and labelset_from_exercise_filtered.txt one gets the files used for training and one can use the ad.read_in() function to check the tensor shape.
2. Then, open the __main__ file and replace the file_to_read with the filename of testing.
3. Run the md.modify_data() method to generate a data file and label file named according to the __files in config.py, located in the DATA_FOLDER.
4. Read in the data and label files using the md.load_data() function.
5. Define the model to load, either concatenated_model.h5 (default, and the one desired of use) or sequential_model.h5
6. Choose whether to evaluate or predict using the loaded model.
