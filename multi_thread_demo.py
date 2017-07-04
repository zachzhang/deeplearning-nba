# LIVE DEMO
import time

import numpy as np

from scipy.misc import imresize
import cv2
from keras_model import *

## model = torch.load(open('model.p', 'rb'))

model = CNN()
model.load_weights('nba_model_small.h5')

cap = cv2.VideoCapture('demo.mp4')

fps = int(cap.get(cv2.CAP_PROP_FPS))

fps = 64

size = (50, 50)

i = 0

while (True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    if i % (1 * fps) == 0:
        start = time.time()

        x = np.zeros((1, size[0], size[1], 3))
        x[0] = imresize(frame, size)

        y_hat = model.predict(x)[0]

        print(y_hat[0])

        y_hat = (y_hat.data.numpy()[0] > .7) * 1

        label = 'GAME' if y_hat > 0 else 'COMMERICAL'

        print(time.time() - start)

    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(frame, label, (100, 100), font, 2, (255, 255, 255), 2, cv2.LINE_AA)

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

