import numpy as np
import time
np.random.seed(123)  # for reproducibility

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
from keras.layers.normalization import BatchNormalization
from keras_model import *




model = CNN()

model.fit(X_train, Y_train,
          batch_size=32, nb_epoch=10, verbose=1)

score = model.evaluate(X_test, Y_test, verbose=0)

model.save_weights("model.h5")
