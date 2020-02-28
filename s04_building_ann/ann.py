from keras import Sequential
from keras.layers import Dense


class Ann:
    """
    Artificial Neural Networks (ANN) with Sequential classifier
    """

    __classifier = Sequential()

    def add_layer(self, node, activation='relu'):
        """
        add ANN layer

        :type activation: str
        :type node: int
        :param node: How may nodes will be creating
        :param activation: activation name, default 'relu'
        """

        self.__classifier.add(Dense(node, kernel_initializer='uniform', activation=activation))

    def compile(self):
        """
        Configures the model for training.
        """

        self.__classifier.compile('adam', loss='binary_crossentropy', metrics=['accuracy'])

    def train(self, x, y):
        """
        Train the data set

        :type y: numpy.ndarray
        :type x: numpy.ndarray
        :param x: x data
        :param y: y data
        """

        self.__classifier.fit(x, y, batch_size=10, epochs=30)

    def predict(self, x):
        """
        predict the independent variables

        :type x: numpy.ndarray
        :param x: x data to test
        :return: return list of prediction result
        :rtype: numpy.ndarray
        """

        return self.__classifier.predict(x)

    def get_classifier(self):
        """
        get classifier instance

        :return: classifier instance
        :rtype: Sequential
        """

        return self.__classifier
