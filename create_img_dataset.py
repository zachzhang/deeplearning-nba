import skvideo.io
import os
from scipy.misc import imresize
import numpy as np
import cv2 as cv

def check_size(file):

    cap = cv.VideoCapture(file)
    ret, frame = cap.read()

    print(frame.shape)

def read_vid(file,size, sample_rate):

    cap = cv.VideoCapture(file)

    fps= int( cap.get(cv.cv.CV_CAP_PROP_FPS) )
    length = int(cap.get(cv.cv.CV_CAP_PROP_FRAME_COUNT))


    freq = fps // sample_rate

    X = np.zeros((length // freq ,size[0] , size[1] ,3 ))

    i = 0
    j = 0
    while(cap.isOpened()):

        ret, frame = cap.read()

        if ret==True:

            if i % freq ==0:

                X[j]  = imresize(frame,size)

                j+=1
                
                if j >= X.shape[0]:
                    break

            i+=1
        else:
            break


    return(X)

IMG_SIZE = (50,50)
SAMPLE_RATE = 1

game_videos = os.listdir('./games')
com_videos = os.listdir('./commericals')


X = []
y = []

for game in game_videos:
    
    
    try:
        scaled = read_vid('./games/' + game, IMG_SIZE, SAMPLE_RATE )
        X.append(scaled)
        y.append(np.ones((scaled.shape[0],1)))
    except:
        pass

for com in com_videos:

    try:
        scaled = read_vid('./commericals/' + com, IMG_SIZE, SAMPLE_RATE)
        X.append(scaled)
        y.append(np.zeros((scaled.shape[0],1)))
    except:
        pass

X = np.vstack(X)
y = np.vstack(y)

print(X.shape,y.shape)
print(y.mean())

np.save('X.npy',X)
np.save('y.npy',y)


