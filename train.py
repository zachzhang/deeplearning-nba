import torch.utils.data as data
import pickle
import numpy as np
import argparse
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from model import *
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve

X = np.load('X.npy') / 255.
X = np.transpose(X, (0,3,1,2))

y = np.load('y.npy')

#CHANGE TO CHECK FOR NOVEL COMMERICALS
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_loader = data.TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
test_loader = data.TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

train_loader = data.DataLoader(train_loader, batch_size=64, shuffle=True, pin_memory=False)
test_loader = data.DataLoader(test_loader, batch_size=64, shuffle=False, pin_memory=False)

model = ConvNet()
#model = torch.load(open('model.p','rb'))



opt = optim.Adam(model.parameters(), lr=0.001)

bce = nn.BCELoss()


def train():
    avg_loss = 0
    model.train()
    start = time.time()

    for data, target in train_loader:
        data = Variable(data).float()
        target = Variable(target).float()

        opt.zero_grad()

        y_hat = model(data)

        loss = bce(y_hat, target)
        loss.backward()

        opt.step()

        avg_loss += loss.data[0]

    print("TRAINING loss: ", (avg_loss / len(train_loader)), ' time: ', time.time() - start)


def test():
    avg_loss = 0
    acc = 0
    model.eval()
    start = time.time()

    y_hat_all = np.zeros(y_test.shape)
    i = 0
    for data, target in test_loader:

        data = Variable(data).float()
        target = Variable(target).float()

        y_hat = model(data)

        y_hat_all[i*data.size()[0]:(i+1)*data.size()[0]] = y_hat.data.numpy()

        loss = bce(y_hat, target)

        avg_loss += loss.data[0]
        i+=1
    print("TESTING loss: ", (avg_loss / len(test_loader)), ' time: ', time.time() - start)


    #Pick a threshold according to f score
    precision, recall, thresholds = precision_recall_curve(y_test, y_hat_all)
    f_score = 2* precision * recall / (precision + recall)
    i_max = np.nanargmax(f_score)
    f_max = f_score[i_max]
    max_thresh = thresholds[i_max]
    
    #Convert prob to a decision
    y_pred = (y_hat_all > max_thresh)

    #accuracy
    acc = ( y_pred == y_test).mean()

    #confusion matrix
    cm = confusion_matrix(y_test, y_pred)    

    print('Accuracy: ' , acc, ' F-score: ', f_max )

    print('Confusion Matrix')
    print(cm)

    print(max_thresh)

    np.save('a.npy',y_test)
    np.save('b.npy',y_pred)

#test()

for i in range(1):

    train()
    test()
    torch.save(model,open('model.p','wb'))


