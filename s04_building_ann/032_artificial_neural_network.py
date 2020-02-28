from sklearn.metrics import confusion_matrix

# import keras
from keras.models import Sequential
from keras.layers import Dense

from s04_building_ann.cx01_data_processing import load_dataset, encode, split, scaling

x, y = load_dataset()

x = encode(x, 1)
x = encode(x, 2)

x_train, x_test, y_train, y_test = split(x, y)
x_train, x_test = scaling(x_train, x_test)

# -------------------------------------------------------------------
# Part 2 - Now let's make the ANN
# -------------------------------------------------------------------

# Initializing the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(6, kernel_initializer='uniform', activation='relu'))

# Adding second layer
classifier.add(Dense(6, kernel_initializer='uniform', activation='relu'))

# Adding the output layer
classifier.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))

# Compiling the ANN
classifier.compile('adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fitting the ANN to the training set
classifier.fit(x_train, y_train, batch_size=10, epochs=30)

# Predicting the test set results
y_pred = classifier.predict(x_test)

# Change prediction to Boolean
y_pred = (y_pred > 0.5)

# Making the confusion matrix

cm = confusion_matrix(y_test, y_pred)

# Examine the var cm above
