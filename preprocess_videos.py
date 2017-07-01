
import os

#Reduce videos to a standard size

cmd = 'ffmpeg -y -i {} -vf scale=200:200 -r 1 {}'

base_dir = '/scratch/zz1409/football/'
base_dir2 = '/scratch/zz1409/football2/'

files = os.listdir(base_dir)

for f in files:
    #new_f = f.split('.mp4')[0]+'_.mp4'
    os.system(cmd.format(base_dir+f,base_dir2+f))
