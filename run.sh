#!/bin/bash
#SBATCH --nodes=1
#SBATCH --mem=80GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=deep_nba
#SBATCH --time=168:00:00

module purge
module load python/intel/2.7.12
module load ffmpeg/intel/3.2.2
module load opencv/intel/2.4.13.2 

cd /home/zz1409/deeplearning-nba

#python utils.py
#python scrape_com_audio.py 
python scrape_commerical2.py

