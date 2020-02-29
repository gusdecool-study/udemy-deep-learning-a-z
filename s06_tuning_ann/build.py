from keras import Sequential

from s04_building_ann.ann import Ann


def build_classifier(optimizer: str = 'adam'):
    """
    build classifier

    :return: classifier
    :rtype: Sequential
    """

    ann = Ann()
    ann.add_layer(6)
    ann.add_drop(0.1)
    ann.add_layer(6)
    ann.add_drop(0.1)
    ann.add_layer(1, 'sigmoid')
    ann.compile(optimizer)

    return ann.get_classifier()
