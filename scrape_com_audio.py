from bs4 import BeautifulSoup
import requests
import time
from fake_useragent import UserAgent
import pickle
import skvideo.io
import os
from scipy.misc import imresize
import numpy as np
import cv2 as cv
from python_speech_features import mfcc
import scipy.io.wavfile as wav

def download_video(url):

    fn = url.split('/')[-1].split('mp4')[0]
    fn = './commericals/' + fn + '.mp4'

    written = True

    r = requests.get(url)

    with open(fn, 'wb') as f:

        f.write(r.content)

        time.sleep(18)

    return fn,written


def extract_features(file,size, sample_rate):

    cap = cv.VideoCapture(file)
    fps= int( cap.get(cv.cv.CV_CAP_PROP_FPS) )
    length = int(cap.get(cv.cv.CV_CAP_PROP_FRAME_COUNT))
    freq = fps // sample_rate

    if freq ==0:
        return np.zeros((1 ,size[0] , size[1] ,3 ))

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

    X = X.astype(np.float32)

    return(X)

proxies = [
    '138.128.225.235:80:zachproxy:zachariah36',
    '154.16.33.132:80:zachproxy:zachariah36',
    '107.175.34.13:80:zachproxy:zachariah36',
    '104.168.68.223:80:zachproxy:zachariah36',
    '107.175.32.115:80:zachproxy:zachariah36',
    '138.128.225.216:80:zachproxy:zachariah36',
    '138.128.225.190:80:zachproxy:zachariah36',
    '107.175.34.7:80:zachproxy:zachariah36',
    '154.16.27.97:80:zachproxy:zachariah36',
    '155.94.218.68:80:zachproxy:zachariah36'
]

ua = UserAgent()
video_urls = []

home_url = 'https://adland.tv/commercials'
url_prefix = 'https://adland.tv/'

pointer = 0

IMG_SIZE= (50,50)

cmd= 'ffmpeg -ac 1 -i {} {}'
rm = 'rm {}'


'''
Get the urls of the right files first
'''

for i in range(60,120):

    if i == 0:
        base_url = home_url
    else:
        base_url = home_url + '?page=' + str(i)

    headers = {"Connection": "close", "User-Agent": ua.random}
    proxy = {"http": proxies[pointer % len(proxies)]}
    pointer += 1

    rsp = requests.get(base_url, headers=headers)
    time.sleep(6)

    soup = BeautifulSoup(rsp.content)

    videos = soup.find_all('div', {'class': 'post-title'})

    for vid in videos:
        vid_link = vid.find_all('a')[0].attrs['href']

        headers = {"Connection": "close", "User-Agent": ua.random}
        proxy = {"http": proxies[pointer % len(proxies)]}
        pointer += 1

        r = requests.get(url_prefix + vid_link, headers=headers)

        vid_page = BeautifulSoup(r.content)

        source = vid_page.find_all('source')

        vid_file = [s.attrs['src'] for s in source if '.mp4' in s.attrs['src']]

        video_urls += vid_file

        pickle.dump(video_urls , open('video_urls.p','wb'))

        print(vid_file)

        if len(vid_file) == 0:
            continue

        #DOWNLOAD THE VIDEO
        fn , written = download_video(vid_file[0])

        if written:

            try:

                base_fn =fn.split('/')[-1].split('.mp4')[0]
                new_fn = base_fn+'.wav'

                os.system(cmd.format(fn,'./commerical_audio/' +new_fn))

                os.system(rm.format(fn))

                fs,x = wav.read('./commerical_audio/' +new_fn)            

                if len(x.shape)==2:
                    mel= mfcc(x[:,0],fs)
                else:
                    mel= mfcc(x,fs)

                np.save( './commerical_audio/' + base_fn +'.npy' , mel.astype(np.float32))

                os.system(rm.format( './commerical_audio/' + +new_fn))
            except:
                pass

        time.sleep(6)
