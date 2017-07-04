import numpy as np
import time
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
from keras.layers.normalization import BatchNormalization
from keras_model import *
from utils import *


game_dir= '/scratch/zz1409/old_data/games/games/'

com_dir ='/scratch/zz1409/old_data/coms/commericals/'
X_train, X_test, y_train, y_test  = load_data(game_dir,com_dir)

model = CNN()
#model.load_weights("nba_model.h5")

model.fit(X_train, y_train,batch_size=64, nb_epoch=12, verbose=1, validation_data = (X_test,y_test))

score = model.evaluate(X_test, y_test, verbose=0)

print(score)

model.save_weights("nba_model.h5")
