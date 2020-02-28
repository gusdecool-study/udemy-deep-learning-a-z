# Artificial Neural Network

# Importing the libraries
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# import keras
from keras.models import Sequential
from keras.layers import Dense

# -------------------------------------------------------------------
# Part 1 - Data Precessing
# -------------------------------------------------------------------

file_location = os.getcwd()
file_location += '/s04_building_ann/churn_modelling.csv'
dataset = pd.read_csv(file_location)

X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# -------------------------------------------------------------------
# Encoding categorical data
# -------------------------------------------------------------------

# Encode country
label_encoder_X_1 = LabelEncoder()
X[:, 1] = label_encoder_X_1.fit_transform(X[:, 1])

# Encode gender
label_encoder_X_2 = LabelEncoder()
X[:, 2] = label_encoder_X_2.fit_transform(X[:, 2])

# Causing error in Keras classifier fit
# ct = ColumnTransformer(
#     [('Country', OneHotEncoder(categories='auto'), [0])],
#     remainder='passthrough')
#
# X = ct.fit_transform(X)

# Splitting the dataset into the Training set and Test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Feature scaling
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler(with_mean=False)
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

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
classifier.fit(X_train, y_train, batch_size=10, epochs=30)

# Predicting the test set results
y_pred = classifier.predict(X_test)

# Change prediction to Boolean
y_pred = (y_pred > 0.5)

# Making the confusion matrix

cm = confusion_matrix(y_test, y_pred)

# Examine the var cm above
