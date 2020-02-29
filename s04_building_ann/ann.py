from keras import Sequential
from keras.layers import Dense, Dropout


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

    def add_drop(self, fraction):
        """
        add dropout in hidden layer

        :param fraction: fraction number between 0 and 1.
        :type fraction: float
        """

        self.__classifier.add(Dropout(fraction))

    def compile(self, optimizer: str = 'adam'):
        """
        Configures the model for training.
        """

        self.__classifier.compile(optimizer, loss='binary_crossentropy', metrics=['accuracy'])

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
