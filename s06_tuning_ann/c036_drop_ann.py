from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from s04_building_ann.main_data_pre_processing import x_train, y_train
from s06_tuning_ann.build import build_classifier


def execute():
    classifier = KerasClassifier(build_classifier, batch_size=10, nb_epoch=30)
    accuracies = cross_val_score(classifier, x_train, y_train, cv=10, n_jobs=-1)
    mean = accuracies.mean()
    variance = accuracies.std()
