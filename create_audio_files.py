import skvideo.io
import os
from scipy.misc import imresize
import numpy as np
import cv2 as cv

game_videos = os.listdir('./games')
com_videos = os.listdir('./commericals')


ffmpeg = 'ffmpeg -ac 1 -i {} {}'

for game in game_videos:
    
    try:

        in_file = './games/' + game
        out_file = './games_audio/' + game.split('.mp4')[0] + '.mp3'

        os.system(ffmpeg.format(in_file,out_file))

    except:
        pass

for com in com_videos:

    try:
        in_file = './commericals/' + com              
        out_file = './commerical_audio/' + com.split('.mp4')[0] + '.mp3'

        os.system(ffmpeg.format(in_file,out_file))

    except:
        pass



