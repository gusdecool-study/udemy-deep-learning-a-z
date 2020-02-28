import pandas as pd
import pathlib


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
