from keras import Sequential
from keras.wrappers.scikit_learn import KerasClassifier

from s04_building_ann.ann import Ann


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


classifier = KerasClassifier(build_classifier, batch_size=10, nb_epoch=100)
