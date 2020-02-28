from s04_building_ann.cx01_data_processing import load_dataset, encode, split, scaling

# -------------------------------------------------------------------------------------------------
# data pre-processing
# -------------------------------------------------------------------------------------------------

x, y = load_dataset()
x = encode(x, 1)  # encode country
x = encode(x, 2)  # encode gender

x_train, x_test, y_train, y_test = split(x, y)  # split data for test and train
x_train, x_test = scaling(x_train, x_test)  # scaling the data
