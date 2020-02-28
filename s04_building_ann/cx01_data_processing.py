import pandas as pd
import pathlib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_dataset():
    """
    Load dataset from CSV

    :return: Array of: independent variables, dependant variables
    :rtype: (numpy.ndarray, numpy.ndarray)
    """
    file_location = str(pathlib.Path(__file__).parent.absolute()) + '/churn_modelling.csv'
    dataset = pd.read_csv(file_location)

    independent_variables = dataset.iloc[:, 3:13].values
    dependant_variables = dataset.iloc[:, 13].values

    return independent_variables, dependant_variables


def encode(data, index):
    """
    Encode categorical data into numbers

    :param data: data to be encoded
    :type data: numpy.ndarray
    :param index: the index that will be encoded
    :type index: int

    :return: encoded data
    :rtype: numpy.ndarray
    """

    encoder = LabelEncoder()
    data[:, index] = encoder.fit_transform(data[:, index])

    return data


def split(x, y):
    """
    Split the data that we want to train and test

    :param x: x data
    :type x: numpy.ndarray
    :param y: y data
    :type y: numpy.ndarray

    :return: (x_train, x_test, y_train, y_test)
    :rtype: (numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray)
    """

    return train_test_split(x, y, test_size=0.25, random_state=0)


def scaling(train_data, test_data):
    """
    Scaling the train and test data

    :param train_data: train data
    :type train_data: numpy.ndarray
    :param test_data: test data
    :type test_data: numpy.ndarray
    :return: list of scaled train and test data
    :rtype: (numpy.ndarray, numpy.ndarray)
    """

    scale = StandardScaler(with_mean=False)
    train_data = scale.fit_transform(train_data)
    test_data = scale.fit(test_data)

    return train_data, test_data
