import numpy as np
import time
np.random.seed(123)  # for reproducibility

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
from keras.layers.normalization import BatchNormalization


def CNN():
    def ConvBNLayer(num_filters):
        model.add(Convolution2D(num_filters, kernel_size=(3, 3), activation='relu'))
        model.add(BatchNormalization(axis=3))
        model.add(Dropout(p))

    p = .5

    model = Sequential()

    model.add(Convolution2D(64, kernel_size=(3, 3), activation='relu', input_shape=(50, 50, 3)))
    model.add(BatchNormalization(axis=1))
    model.add(Dropout(p))

    ConvBNLayer(128)
    model.add(MaxPooling2D(pool_size=(2, 2)))

    ConvBNLayer(256)
    ConvBNLayer(256)
    model.add(MaxPooling2D(pool_size=(2, 2)))

    ConvBNLayer(512)
    ConvBNLayer(512)
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization(axis=1))
    model.add(Dropout(p))
    model.add(Dense(1, activation='sigmoid'))

    start = time.time()
    print(model.predict(np.ones((64, 50, 50, 3))).shape)
    print(time.time() - start)

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['binary_accuracy'])

    return model