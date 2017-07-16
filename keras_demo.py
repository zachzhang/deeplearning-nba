#LIVE DEMO
import pickle 
import numpy as np
import pickle
from scipy.misc import imresize
import cv2 
from keras_model import *

model = CNN()
model.load_weights('./nba_model_small.h5')

cap = cv2.VideoCapture('demo.mp4')

fps= int( cap.get(cv2.CAP_PROP_FPS ))

size= (50,50)

i = 0

label = 'GAME'

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if i % (1*fps) == 0:

        x = np.zeros((1,size[0],size[1],3))
        x[0] = imresize(frame,size)

        y_hat = model.predict(x)[0]

        print(y_hat)

        y_hat = (y_hat  > .7) * 1

        label = 'GAME' if y_hat > 0 else 'COMMERICAL'


    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame,label,(100,100), font, 2,(255,255,255),2,cv2.LINE_AA)
    '''
    cv2.namedWindow('window',cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('window',cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    '''
    # Display the resulting frame
    cv2.imshow('window',frame)
    
    i +=1
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

        
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

