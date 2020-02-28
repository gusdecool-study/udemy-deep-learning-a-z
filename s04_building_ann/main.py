from sklearn.metrics import confusion_matrix
from s04_building_ann.ann import Ann
from s04_building_ann.main_data_pre_processing import x_train, x_test, y_train, y_test

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
