import numpy as np
import time 

import os

def load_data():

    games = os.listdir('./games')
    coms = os.listdir('./commericals')

    game_split = int(np.floor( len(games)*.8 ))
    com_split = int(np.floor( len(coms)*.8 ))

    X_train = []
    y_train = []

    X_test = []
    y_test = []

    start =time.time()

    for i,game in enumerate(games):

        try:

            x = np.load('./games/'+game)
            
            if i < game_split:
                X_train.append(x)
                y_train.append(np.ones((x.shape[0],1)))

            else:
                X_test.append(x)
                y_test.append(np.ones((x.shape[0],1)))
        except:
            pass

    for i,com in enumerate(coms):
        
        try:
            x  = np.load('./commericals/'+com)
            if i < com_split:
                X_train.append(x)
                y_train.append(np.zeros((x.shape[0],1)))
            else:
                X_test.append(x)
                y_test.append(np.zeros((x.shape[0],1)))
        except:
            pass

    X_train = np.vstack(X_train)
    X_test = np.vstack(X_test)
    
    X_train = np.transpose(X_train, (0,3,1,2))
    X_test = np.transpose(X_test, (0,3,1,2)) 

    y_train = np.vstack(y_train)
    y_test = np.vstack(y_test)

    train_perm = np.random.permutation(X_train.shape[0])
    test_perm =  np.random.permutation(X_test.shape[0])

    return X_train[train_perm], X_test[test_perm], y_train[train_perm] , y_test[test_perm]


def split_audio(mel,l=1):

    frame_len = l * 100
    num_frames = mel.shape[0] // frame_len

    flatten = np.zeros((num_frames, mel.shape[1], frame_len ))

    for i in range(num_frames):
        flatten[i] = np.transpose(mel[i * (frame_len): (i + 1) * frame_len])

    return flatten

def load_audio(length= 1):
    games = os.listdir('./games_audio')
    coms = os.listdir('./commerical_audio')

    game_split = int(np.floor(len(games) * .8))
    com_split = int(np.floor(len(coms) * .8))

    X_train ,y_train , X_test, y_test = [],[],[],[]

    start = time.time()

    for i, game in enumerate(games):

        try:

            x = np.load('./games_audio/' + game)
            x = split_audio(x,l=length)

            if i < game_split:
                X_train.append(x)
                y_train.append(np.ones((x.shape[0], 1)))

            else:
                X_test.append(x)
                y_test.append(np.ones((x.shape[0], 1)))
        except:
            pass

    for i, com in enumerate(coms):

        try:
            x = np.load('./commerical_audio/' + com)
            x = split_audio(x,l=length)

            if i < com_split:
                X_train.append(x)
                y_train.append(np.zeros((x.shape[0], 1)))
            else:
                X_test.append(x)
                y_test.append(np.zeros((x.shape[0], 1)))
        except:
            pass

    X_train = np.vstack(X_train)
    X_test = np.vstack(X_test)

    y_train = np.vstack(y_train)
    y_test = np.vstack(y_test)

    train_perm = np.random.permutation(X_train.shape[0])
    test_perm = np.random.permutation(X_test.shape[0])

    return X_train[train_perm], X_test[test_perm], y_train[train_perm], y_test[test_perm]

