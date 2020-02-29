from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

from s04_building_ann.main_data_pre_processing import x_train, y_train
from s06_tuning_ann.build import build_classifier


def execute():
    """
    tuning ANN so we know which parameters work best

    :return: (best parameters, best accuracy)
    :rtype: (dictionary, float)
    """

    classifier = KerasClassifier(build_classifier)
    parameters = {'batch_size': [25, 32], 'nb_epoch': [100, 500], 'optimizer': ['adam', 'rmsprop']}

    grid_search = GridSearchCV(classifier, parameters, 'accuracy', cv=10)
    grid_search = grid_search.fit(x_train, y_train)

    best_parameters = grid_search.best_params_
    best_accuracy = grid_search.best_score_

    return best_parameters, best_accuracy


execute()
