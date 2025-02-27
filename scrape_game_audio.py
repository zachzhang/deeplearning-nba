import sys

from pytube import YouTube
from pprint import pprint
from pytube.exceptions import MultipleObjectsReturned
import cv2 as cv
import os
from scipy.misc import imresize
import numpy as np
import requests
from python_speech_features import mfcc
import scipy.io.wavfile as wav


'''
Get The Actual MP4s For NBA GAMES
'''


def extract_features(file, size, sample_rate):
    cap = cv.VideoCapture(file)
    fps = int(cap.get(cv.cv.CV_CAP_PROP_FPS))
    length = int(cap.get(cv.cv.CV_CAP_PROP_FRAME_COUNT))
    freq = fps // sample_rate

    X = np.zeros((length // freq, size[0], size[1], 3))
    i = 0
    j = 0

    while (cap.isOpened()):

        ret, frame = cap.read()

        if ret == True:
            
            if i % freq == 0:

                X[j] = imresize(frame, size)

                j += 1

                if j >= X.shape[0]:
                    break

            i += 1

        else:
            break

    X = X.astype(np.float32)

    return (X)


nba_replays = 'https://www.youtube.com/channel/UC0rDNVMafPWtpY63vFbxC3A/videos'
html = requests.get(nba_replays)
games = [ x.split('"')[0] for x in html.text.split('"/watch?v=')  ][1:]
games = ['https://www.youtube.com/watch?v=' + x for x in games ]

size = (50,50)

cmd= 'ffmpeg -ac 1 -i {} {}'
rm = 'rm {}'
j= 0

for i,game in enumerate(games):

    written = True
   
    if j > 40:
        break

    try:
        yt = YouTube(game)

        yt.set_filename( 'game_'+str(i))

        video = yt.filter('mp4')[0]
    
        print(video)

        video.download('./games')


        fn = './games/game_'+str(i) +'.mp4'
        new_fn = 'game_'+str(i)+'.wav'

        # get .wav
        os.system(cmd.format(fn,'./games_audio/' +new_fn))

        # delete video file
        os.system(rm.format('./games/game_'+str(i) +'.mp4'))

        # read wav
        fs,x = wav.read('./games_audio/' +new_fn)

        #mfcc coefs
        mel= mfcc(x[:,0],fs)

        #save mfcc
        np.save('./games_audio/game_'+str(i)+'.npy' , mel.astype(np.float32))

        #remove .wav
        os.system(rm.format( './games_audio/' +new_fn ) )

        j+=1

    except:
        pass
