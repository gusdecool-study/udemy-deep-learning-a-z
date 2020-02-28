from sklearn.metrics import confusion_matrix
from s04_building_ann.cx01_data_processing import DataProcessing
from s04_building_ann.cx02_ann import Ann

# -------------------------------------------------------------------------------------------------
# data pre-processing
# -------------------------------------------------------------------------------------------------

data_processing = DataProcessing()
x, y = data_processing.load_dataset()

x = DataProcessing.encode(x, 1)  # encode country
x = DataProcessing.encode(x, 2)  # encode gender

x_train, x_test, y_train, y_test = DataProcessing.split(x, y)  # split data for test and train

# scaling the data
data_processing.fit(x_train)
x_train = data_processing.scale(x_train)
x_test = data_processing.scale(x_test)

# -------------------------------------------------------------------------------------------------
# Artificial Neural Networks (ANN)
# -------------------------------------------------------------------------------------------------

ann = Ann()
ann.add_layer(6)  # first hidden layer
ann.add_layer(6)  # second hidden layer
ann.add_layer(1, 'sigmoid')  # output layer, 1 output using Sigmoid
ann.compile()  # configure the model for training
ann.train(x_train, y_train)  # train the model

y_prediction = ann.predict(x_test)
y_prediction = (y_prediction > 0.5)  # Change prediction to Boolean

cm = confusion_matrix(y_test, y_prediction)  # Making the confusion matrix

# Examine the var cm above
