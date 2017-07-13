# LIVE DEMO
import time

import numpy as np

from scipy.misc import imresize
import cv2
from keras_model import *
from multiprocessing import Process
import threading
from multiprocessing import Queue
import time

q = Queue()

global LABEL
LABEL = 'GAME'

global CURRENT_FRAME
CURRENT_FRAME = np.zeros((1,50,50,3))

def img_processing(q):

    model = CNN()
    model.load_weights('nba_model_small.h5')
    size = (50, 50)
    while True:


        frame = CURRENT_FRAME
        X = np.expand_dims(imresize(frame, size), 0)
        y_hat = model.predict(X)[0]

        print(y_hat)

        global LABEL

        if y_hat > .7:
            LABEL = 'GAME'
        else:
            LABEL = 'COMMERICAL'

        time.sleep(1)

        '''
        if ~q.empty():

            frame = q.get()
            X = np.expand_dims(imresize(frame, size),0)
            y_hat = model.predict(X)[0]

            print(y_hat)

            global LABEL

            if y_hat > .7:
                LABEL = 'GAME'
            else:
                LABEL = 'COMMERICAL'

        else:
            time.sleep(1)

        '''

def cap_video(q):

    #cap = cv2.VideoCapture('demo.mp4')
    cap = cv2.VideoCapture('./commerical.mp4')

    fps = 32

    i = 0
    global CURRENT_FRAME

    while (True):
        # Capture frame-by-frame
        ret, frame = cap.read()


        #if i % (1 * fps) == 0 and ~q.full():
        #    CURRENT_FRAME = frame
            #q.put(frame)


        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, LABEL, (100, 100), font, 2, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('frame', frame)
        i += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()



#cv2.namedWindow("Right View")

#img_proc = threading.Thread(target=img_processing, args = (q,))
#img_proc.start()

cap_video(q)


#vid_cap_thread = threading.Thread(target=cap_video, args = (q,))
#vid_cap_thread.start()