


from sklearn.model_selection import train_test_split
from model import *
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from utils import *
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression


model = LogisticRegression(max_iter=10)

X_train, X_test, y_train, y_test = load_audio(1)

X_train = X_train.reshape(X_train.shape[0] , -1)

X_test = X_test.reshape(X_test.shape[0],-1)

model.fit(X_train,y_train)

print(model.score(X_test,y_test))


