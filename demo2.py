# LIVE DEMO

from scipy.misc import imresize
import torch.utils.data as data
import pickle
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import pickle
import skvideo.io
from scipy.misc import imresize
import cv2

model = torch.load(open('model.p', 'rb'))

cap = cv2.VideoCapture(0)

fps = int(cap.get(cv2.CAP_PROP_FPS))
fps = 32

size = (50, 50)

i = 0

label = 'GAME'

while (True):
    # Capture frame-by-frame
    ret, frame = cap.read()


    if i % (1 * fps) == 0:
        x = np.zeros((1, size[0], size[1], 3))
        x[0] = imresize(frame, size)
        x = x.transpose((0, 3, 1, 2))

        x = Variable(torch.from_numpy(x)).float()

        p = model(x)

        print(p.data.numpy()[0])

        # y_hat = (y_hat.data.numpy()[0]  > 1e-16) * 1

        y_hat = (p.data.numpy()[0] > .7) * 1

        label = 'GAME' if y_hat > 0 else 'COMMERICAL'


    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, label +' : ' + str(p.data.numpy()[0]), (100, 100), font, 2, (255, 255, 255), 2, cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow('frame', frame)

    i += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

        # if i > 100:

        #    break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

