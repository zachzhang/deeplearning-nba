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
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from utils import *
import time

from model import *

start = time.time()

X_train = np.load('X_train_norm.npy').astype(np.float32)
X_test = np.load('X_test_norm.npy').astype(np.float32)
y_train = np.load('y_train.npy').astype(np.float32)
y_test = np.load('y_test.npy').astype(np.float32)

#mean = X_train.mean(axis=(0,2)).reshape((1,4*13,1))
#var = X_train.var(axis=(0,2)).reshape((1,4*13,1))

#X_train = (X_train - mean)/var
#X_test = (X_test - mean)/var

print('Finished Loading' , time.time() - start)

#X_train, X_test, y_train, y_test = load_audio(1)

gpu = True


train_loader = data.TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
test_loader = data.TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

if gpu:
    train_loader = data.DataLoader(train_loader, batch_size=64, shuffle=True, pin_memory=True)
    test_loader = data.DataLoader(test_loader, batch_size=64, shuffle=False, pin_memory=True)
else:
    train_loader = data.DataLoader(train_loader, batch_size=64, shuffle=True, pin_memory=False)
    test_loader = data.DataLoader(test_loader, batch_size=64, shuffle=False, pin_memory=False)

print('Creating Model', time.time() - start)

model = AudioConvNet()

#model = torch.load('model.p')

if gpu:
    model = model.cuda()


opt = optim.Adam(model.parameters(), lr=0.001)

bce = nn.BCELoss()


def train():
    avg_loss = 0
    model.train()
    start = time.time()

    for data, target in train_loader:
        data = Variable(data).float()
        target = Variable(target).float()

        if gpu:
            data,target = data.cuda(),target.cuda()

        opt.zero_grad()

        y_hat = model(data)

        loss = bce(y_hat, target)
        loss.backward()

        opt.step()

        avg_loss += loss.data.cpu()[0]

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

        if gpu:
            data,target = data.cuda(),target.cuda()

        y_hat = model(data)

        y_hat_all[i * data.size()[0]:(i + 1) * data.size()[0]] = y_hat.unsqueeze(1).data.cpu().numpy()

        loss = bce(y_hat, target)

        avg_loss += loss.data.cpu()[0]
        i += 1
    print("TESTING loss: ", (avg_loss / len(test_loader)), ' time: ', time.time() - start)

    # Pick a threshold according to f score
    precision, recall, thresholds = precision_recall_curve(y_test, y_hat_all)
    f_score = 2 * precision * recall / (precision + recall)
    i_max = np.nanargmax(f_score)
    f_max = f_score[i_max]
    max_thresh = thresholds[i_max]

    # Convert prob to a decision
    y_pred = (y_hat_all > max_thresh)

    # accuracy
    acc = (y_pred == y_test).mean()

    # confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    print('Accuracy: ', acc, ' F-score: ', f_max)

    print('Confusion Matrix')
    print(cm)

    print(max_thresh)


#test()

for i in range(10):
    train()
    test()
    torch.save(model, open('model.p', 'wb'))

