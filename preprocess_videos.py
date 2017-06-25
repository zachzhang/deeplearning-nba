
import os

#Reduce videos to a standard size

cmd = 'ffmpeg -y -i {} -vf scale=200:200 -r 8 {}'

base_dir = '/scratch/zz1409/commericals/'

files = os.listdir(base_dir)

for f in files:

    os.system(cmd.format(base_dir+f,base_dir+f))
