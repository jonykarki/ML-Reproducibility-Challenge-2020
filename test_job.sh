#!/bin/bash

#SBATCH -J MLR
#SBATCH -p gtx
#SBATCH -o ./logs/log_1018_o.%j
#SBATCH -e ./logs/log_1018_e.%j
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 24:00:00
#SBATCH --mail-user=jkarki@patriots.uttyler.edu
#SBATCH -A COSC4381Spring2020
#SBATCH --mail-type=all

module load python3/3.7.0

python3 train_color_denoising.py --train_batch 2 --noise_level 5 --device_id 0  &

python3 train_color_denoising.py --train_batch 4 --noise_level 5 --device_id 1  &

python3 train_color_denoising.py --train_batch 8 --noise_level 5 --device_id 2  


