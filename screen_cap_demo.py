# LIVE DEMO
import time

import os
import numpy as np
import pyscreenshot as ImageGrab
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
CURRENT_FRAME = np.zeros((50,50,3))

def mute():
    os.system('amixer cset numid=3 1')

def unmute():
    os.system('amixer cset numid=3 2')

def img_processing(q):


    model = CNN()
    model.load_weights('nba_model_small.h5')
    size = (50, 50)

    prev = 0 

    while True:

        im=ImageGrab.grab()
        frame = np.array(im)

        X = np.expand_dims(imresize(frame, size), 0)

        y_hat = model.predict(X)[0]

        print(y_hat)

        global LABEL

        if y_hat > .5:
            if prev ==0:
                unmute()
            prev = 0 
        else:
            if prev == 1:
                mute()
            prev = 1

        time.sleep(1)


def cap_video(q):


    fps = 32

    i = 0
    global CURRENT_FRAME


    while (True):

        im=ImageGrab.grab()
        im.show()
        frame = np.array(im)
        time.sleep(1)
        print(frame.shape)
        '''
        if i % (1 * fps) == 0 and ~q.full():
            CURRENT_FRAME = frame
            #q.put(frame)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, LABEL, (100, 100), font, 2, (255, 255, 255), 2, cv2.LINE_AA)
        '''
        '''
        cv2.imshow('frame', frame)
        i += 1

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        '''
    cap.release()
    cv2.destroyAllWindows()




img_processing(q)

#cv2.namedWindow("Right View")
'''
img_proc = threading.Thread(target=img_processing, args = (q,))
img_proc.start()
'''

'''
cap_video(q)

vid_cap_thread = threading.Thread(target=cap_video, args = (q,))
vid_cap_thread.start()
'''
