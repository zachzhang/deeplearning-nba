from bs4 import BeautifulSoup
import requests
import time
from fake_useragent import UserAgent
import pickle

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
video_urls = pickle.load(open('video_urls.p','rb'))

'''
Get The Actual MP4s For Commericals
'''

pointer = 0

for url in video_urls:

    headers = {"Connection": "close", "User-Agent": ua.random}
    proxy = {"http": proxies[pointer % len(proxies)]}
    pointer += 1

    fn = url.split('/')[-1].split('mp4')[0]

    print(fn)

    fn = './commericals/' + fn

    try:

        r = requests.get(url)

        with open(fn + '.mp4', 'wb') as f:
            f.write(r.content)

        command = 'ffmpeg -y -i {} -vf scale=iw*.5:ih*.5 {}'
        rm = 'rm {}'

        time.sleep(18)

    except:
        pass
