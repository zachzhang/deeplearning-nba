#!/bin/bash
#SBATCH --nodes=1
#SBATCH --mem=128GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=nba
#SBATCH --time=8:00:00


module purge
module load python/intel/2.7.12
module load ffmpeg/intel/3.2.2
module load opencv/intel/2.4.13.2 
module load scikit-learn/intel/0.18.1
module load pytorch/intel/20170125

cd /home/zz1409/deeplearning-nba

python train.py
