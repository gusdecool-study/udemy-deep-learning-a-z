from s04_building_ann.data_processing import DataProcessing

data_processing = DataProcessing()
x, y = data_processing.load_dataset()

x = DataProcessing.encode(x, 1)  # encode country
x = DataProcessing.encode(x, 2)  # encode gender

x_train, x_test, y_train, y_test = DataProcessing.split(x, y)  # split data for test and train

# scaling the data
data_processing.fit(x_train)
x_train = data_processing.scale(x_train)
x_test = data_processing.scale(x_test)
