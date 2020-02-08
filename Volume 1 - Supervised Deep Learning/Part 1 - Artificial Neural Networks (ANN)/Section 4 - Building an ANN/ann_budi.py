# Artificial Neural Network

# -------------------------------------------------------------------
# Part 1 - Data Precessing
# -------------------------------------------------------------------

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
from docutils.nodes import classifier

dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# -------------------------------------------------------------------
# Encoding categorical data
# -------------------------------------------------------------------

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Encode country
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

# Encode gender
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

ct = ColumnTransformer(
    [('Country', OneHotEncoder(categories='auto'), [0])], 
    remainder='passthrough')

X = ct.fit_transform(X)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler(with_mean=False)
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# -------------------------------------------------------------------
# Part 2 - Now let's make the ANN
# -------------------------------------------------------------------

# Importing Keras Libraries and Packages
import keras
from keras.models import Sequential
from keras.layers import Dense

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

# -------------------------------------------------------------------
# Clean code below
# -------------------------------------------------------------------

# Fitting logistic refresstion to the training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the result
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)