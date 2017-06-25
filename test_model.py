import numpy as np
import time 

from torch.autograd import Variable
import torch
import os



def split_audio(mel,l=1):

    frame_len = l * 100
    num_frames = mel.shape[0] // frame_len

    flatten = np.zeros((num_frames, mel.shape[1], frame_len ))

    for i in range(num_frames):
        flatten[i] = np.transpose(mel[i * (frame_len): (i + 1) * frame_len])

    return flatten

def test_model():

    length = 1
    model = torch.load('model.p')

    games = os.listdir('./games_audio')
    coms = os.listdir('./commerical_audio')

    game_split = int(np.floor(len(games) * .8))
    com_split = int(np.floor(len(coms) * .8))

    
    x = np.load('test_game.npy')
    x = split_audio(x,l=length)
    y = np.ones(x.shape[0])
    
    y_hat = np.ones(x.shape[0])
                
    for i in range( x.shape[0] // 64 ):
                
        y_hat[i*64:(i+1)*64] = model(Variable( torch.from_numpy(x[i*64:(i+1)*64]) ).float() ).data.numpy()

    
    errors = ((y_hat > .7) != y).flatten()
    
    errors = np.nonzero(errors)[0]



    print( len(errors) , x.shape[0] , errors  )


    '''
    for i, game in  enumerate(games):


        #try:


            if i > game_split:
                
                x = np.load('./games_audio/' + game)
                x = split_audio(x,l=length)
                
                y = np.ones(x.shape[0])

                y_hat = np.ones(x.shape[0])

                for i in range( x.shape[0] // 64 ):

                    y_hat[i*64:(i+1)*64] = model(Variable( torch.from_numpy(x[i*64:(i+1)*64]) ).float() ).data.numpy()

                errors = ((y_hat > .7) != y).flatten()
                errors = np.nonzero(errors)[0]

                print(game, len(errors) , x.shape[0] , errors  )

        #except:
        #    pass

    for i, com in enumerate(coms):

        try:
            x = np.load('./commerical_audio/' + com)
            x = split_audio(x,l=length)

            if i > com_split:
                x = np.load('./commerical_audio/' + com)
                x = split_audio(x,l=length)
                
                y = np.zeros(x.shape[0])

                y_hat = model(Variable( torch.from_numpy(x).float() ))
                
                errors = ((y_hat.data.numpy() > .7) != y).flatten()
                errors = np.nonzero(errors)[0]

                if len(errors) / float( x.shape[0]) > .4:
                    print(com, len(errors), x.shape[0]  , errors  )


                
        except:
            pass

    '''


test_model()
