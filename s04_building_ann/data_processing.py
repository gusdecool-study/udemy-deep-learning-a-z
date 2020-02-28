import pandas as pd
import pathlib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


class DataProcessing:
    """
    Data processing utility
    """

    __dataset: None

    def __init__(self):
        self.__scale = StandardScaler(with_mean=False)

    def load_dataset(self):
        """
        Load dataset from CSV

        :return: Array of: independent variables, dependant variables
        :rtype: (numpy.ndarray, numpy.ndarray)
        """
        file_location = str(pathlib.Path(__file__).parent.absolute()) + '/churn_modelling.csv'
        self.__dataset = pd.read_csv(file_location)

        independent_variables = self.__dataset.iloc[:, 3:13].values
        dependant_variables = self.__dataset.iloc[:, 13].values

        return independent_variables, dependant_variables

    @staticmethod
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

    @staticmethod
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

    def fit(self, data):
        """
        Add sample data to fit

        :param data: data
        :type data: numpy.ndarray
        """

        self.__scale.fit(data)

    def scale(self, data):
        """
        Scaling the data. WARNING: you must run fit() before.

        :param data: train data
        :type data: numpy.ndarray
        :return: scaled data
        :rtype: numpy.ndarray
        """

        scaled_data = self.__scale.transform(data)

        return scaled_data
