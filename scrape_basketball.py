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


youtube_page= 'https://www.youtube.com/channel/UC8PmmDVbF7UC3gbFDnLiB7w/videos'

html = requests.get(youtube_page)
games = [ x.split('"')[0] for x in html.text.split('"/watch?v=')  ][1:]
games = ['https://www.youtube.com/watch?v=' + x for x in games ]

base_dir = '/scratch/zz1409/basketball/'
cmd = 'ffmpeg -y -i {} -vf scale=200:200 -r 1 {}'
rm = 'rm {}'
j= 0


for i,game in enumerate(games):

    written = True
   
    try:
        yt = YouTube(game)

        fn = game.split('/watch?v=')[1]
        yt.set_filename(fn)

        video = yt.filter('mp4')[0]
    
        print(video)

        video.download(base_dir)

    except:
        written = False

    if written:

        os.system(cmd.format(base_dir + fn + '.mp4', base_dir + fn+'.mp4'))
