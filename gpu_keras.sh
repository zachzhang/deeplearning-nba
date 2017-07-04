#!/bin/bash
#SBATCH --nodes=1
#SBATCH --mem=50GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=deep_nba
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:2

module purge
module load python/intel/2.7.12
module load tensorflow/python2.7/20170201
module load opencv/intel/2.4.13.2 
module load keras/2.0.2
module load h5py/intel/2.7.0rc2

cd /home/zz1409/deeplearning-nba

python keras_train.py

