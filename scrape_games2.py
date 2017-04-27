import sys

from pytube import YouTube
from pprint import pprint
from pytube.exceptions import MultipleObjectsReturned
import cv2 as cv
import os
from scipy.misc import imresize
import numpy as np
import requests

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


games = [
    'https://www.youtube.com/watch?v=WcvWtA1FP34',
    'https://www.youtube.com/watch?v=9i0qBIqJeNs',
    'https://www.youtube.com/watch?v=912zHQs1hAU',
    'https://www.youtube.com/watch?v=I8hmXSVtnyc',
    'https://www.youtube.com/watch?v=FLCTPqtqLYA',
    'https://www.youtube.com/watch?v=WS0_Fr-o3bY',
    'https://www.youtube.com/watch?v=SsmaRj4KQcs'
]


nba_replays = 'https://www.youtube.com/channel/UC0rDNVMafPWtpY63vFbxC3A/videos'
html = requests.get(nba_replays)
games = [ x.split('"')[0] for x in html.text.split('"/watch?v=')  ][1:]
games = ['https://www.youtube.com/watch?v=' + x for x in games ]

games= [games[1]]

size = (50,50)

rm = 'rm {}'

for i,game in enumerate(games):

    written = True
    yt = YouTube(game)

    yt.set_filename('game_' + str(i))

    print(yt.get_videos())
    # yt.filter('mp4')[-1]
    yt.filter('mp4')[0]

    video = yt.get('mp4','360p')
    video.download('./games')
    print('here')

    '''
    try:
        yt = YouTube(game)

        yt.set_filename( 'game_'+str(i))

        print(yt.get_videos())
        #yt.filter('mp4')[-1]
        yt.filter('mp4')[0]

        video = yt.get('mp4')
        video.download('./games')
        print('here')
    except:
        written = False
    '''

    if written:
        X = extract_features('./games/game_'+str(i) +'.mp4', size, 1)

        # save
        np.save('./games/game_'+str(i) + '.npy', X)

        # delete file
        os.system(rm.format('./games/game_'+str(i) +'.mp4'))