#!/bin/bash

#SBATCH -J MLR_3
#SBATCH -p gtx
#SBATCH -o ./logs/1020_CD_sc_25_16_o.%j
#SBATCH -e ./logs/1020_CD_sc_25_16_e.%j
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 24:00:00
#SBATCH --mail-user=jkarki@patriots.uttyler.edu
#SBATCH -A COSC4381Spring2020
#SBATCH --mail-type=all

module load python3/3.7.0

python3 train_color_denoising.py --train_batch 16 --noise_level 25 --mode sc --device_id 0  &

python3 train_color_denoising.py --train_batch 16 --noise_level 30 --mode sc --device_id 1 &

python3 train_color_denoising.py --train_batch 16 --noise_level 50 --mode sc --device_id 2 &

python3 train_gray_denoising.py --train_batch 16 --noise_level 5 --mode sc --device_id 3   

