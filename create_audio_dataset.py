import skvideo.io
import os
from scipy.misc import imresize
import numpy as np
import cv2 as cv
import scipy.io.wavfile as wav
from scipy.signal import spectrogram

def audio_features(in_file, window = 1):

    print(in_file)
    fs,x = wav.read(in_file)
    print(fs,len(x))

    num_windows =  x.shape[0] // (fs*window)

    window_width = fs*window

    _, _, Sxx = spectrogram(x,fs)

    print(Sxx.shape)

    return(Sxx)

    '''
    _, _, Sxx = spectrogram(x[ 0:window_width],fs)

    
    audio_feat = np.zeros( num_windows , Sxx.shape[0] , Sxx.shape[1]  )

    for i in range(num_windows):

        _, _, Sxx = spectrogram(x[ i* window_width: (i+1)*window_width],fs)

        audio_feat[i] = Sxx

    return(audio_feat)
    '''

SAMPLE_RATE = 1

game_videos = os.listdir('./games_audio')
com_videos = os.listdir('./commerical_audio')


feat = audio_features('./games_audio/' + game_videos[0], window = 1)

print(feat.shape)

'''
X = []
y = []

for game in game_videos:
    
    try:
        feat = audio_features('./games/' + game, window = 1)
        X.append(feat)
        y.append(np.ones(scaled.shape[0]))
    except:
        pass

for com in com_videos:

    try:
        feat = audio_features('./commericals/' + com, window = 1)
        X.append(feat)
        y.append(np.zeros(scaled.shape[0]))
    except:
        pass

X = np.vstack(X)
y = np.vstack(y)

print(X.shape,y.shape)
print(y.mean())

np.save('X.npy',X)
np.save('y.npy',y)

'''
