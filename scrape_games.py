import sys

from pytube import YouTube
from pprint import pprint
from pytube.exceptions import MultipleObjectsReturned

'''
Get The Actual MP4s For NBA GAMES
'''

games = [
    'https://www.youtube.com/watch?v=WcvWtA1FP34',
    'https://www.youtube.com/watch?v=9i0qBIqJeNs',
    'https://www.youtube.com/watch?v=912zHQs1hAU',
    'https://www.youtube.com/watch?v=I8hmXSVtnyc',
    'https://www.youtube.com/watch?v=FLCTPqtqLYA',
    'https://www.youtube.com/watch?v=WS0_Fr-o3bY',
    'https://www.youtube.com/watch?v=SsmaRj4KQcs'
]



for i,game in enumerate(games):

    try:
        yt = YouTube(game)

        yt.set_filename( 'game_'+str(i))

        print(yt.get_videos())
        #yt.filter('mp4')[-1]

        video = yt.get('mp4')
        video.download('./games')
    except:
        pass
