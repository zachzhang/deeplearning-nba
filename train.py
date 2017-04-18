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

X_train = np.random.randn(1000, 3, 50, 50)
y_train = np.ones(1000)

X_test = np.random.randn(100, 3, 50, 50)
y_test = np.ones(100)

train_loader = data.TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
test_loader = data.TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

train_loader = data.DataLoader(train_loader, batch_size=64, shuffle=True, pin_memory=False)
test_loader = data.DataLoader(test_loader, batch_size=64, shuffle=False, pin_memory=False)

model = ConvNet()

opt = optim.Adam(model.parameters(), lr=0.001)

bce = nn.BCELoss()


def train():
    avg_loss = 0
    model.train()
    start = time.time()

    for data, target in train_loader:
        data = Variable(data).float()
        target = Variable(target).float()

        # print(data)
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

    for data, target in test_loader:
        data = Variable(data)
        target = Variable(target)

        y_hat = model(data)

        loss = bce(y_hat, target)

        acc += ((y_hat.data > .5) == target.data).mean()

        avg_loss += loss.data[0]

    print("TRAINING loss: ", (avg_loss / len(train_loader)), ' time: ', time.time() - start)


train()