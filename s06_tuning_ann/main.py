from keras import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from s04_building_ann.ann import Ann
from s04_building_ann.main_data_pre_processing import x_train, y_train


# Evaluating the ANN
def build_classifier():
    """
    build classifier

    :return: classifier
    :rtype: Sequential
    """

    ann = Ann()
    ann.add_layer(6)
    ann.add_layer(6)
    ann.add_layer(1, 'sigmoid')
    ann.compile()

    return ann.get_classifier()


classifier = KerasClassifier(build_classifier, batch_size=10, nb_epoch=30)
accuracies = cross_val_score(classifier, x_train, y_train, cv=10, n_jobs=-1)
mean = accuracies.mean()
variance = accuracies.std()
