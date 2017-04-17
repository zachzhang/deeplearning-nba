# get ads

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
video_urls = []

home_url = 'https://adland.tv/commercials'
url_prefix = 'https://adland.tv/'

pointer = 0

'''
Get the urls of the right files first
'''

for i in range(10):

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
        time.sleep(6)


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

    r = requests.get(url)

    with open(fn + '.mp4', 'wb') as f:
        f.write(r.content)

    time.sleep(6)




