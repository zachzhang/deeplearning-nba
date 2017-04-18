import skvideo.io
import os
from scipy.misc import imresize
import numpy as np
import cv2 as cv
from ffmpy import FFmpeg

IMG_SIZE = (50,50)
SAMPLE_RATE = 1

game_videos = os.listdir('./games')
com_videos = os.listdir('./commericals')


command = 'ffmpeg -y -i {} -vf scale=iw*.5:ih*.5 {}'
rm = 'rm {}'

#for game in game_videos:

#    os.system("some_command with args")

for com in com_videos:

    try:
        in_file = './commericals/' +com
        out_file = './commericals/' +com.split('.mp4')[0] + '_' + '.mp4'

        print(out_file)
        #ff = FFmpeg(inputs={'test.mp4': None},outputs={'output.mp4': ['-vf', 'scale=iw*.5:ih*.5']})
        #ff.run()
        
        os.system(command.format(in_file,out_file))
        os.system(rm.format(in_file))

    except:
        pass
    #print(command.format(file,file))

