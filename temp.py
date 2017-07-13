# LIVE DEMO


import numpy as np
import cv2
from scipy.misc import imresize


def cap_video():

    cap = cv2.VideoCapture('./commerical.mp4')

    i = 0

    while (True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        frame = imresize(frame, (100,100))
        cv2.imshow('frame', frame)
        i += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

cap_video()

'''

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
'''