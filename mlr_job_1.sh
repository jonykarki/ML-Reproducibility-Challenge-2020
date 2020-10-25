#!/bin/bash

#SBATCH -J MLR_1
#SBATCH -p p100
#SBATCH -o ./logs/1024_GD_sc_5025_16_o.%j
#SBATCH -e ./logs/1024_GD_sc_5025_16_e.%j
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 24:00:00
#SBATCH --mail-user=jkarki@patriots.uttyler.edu
#SBATCH -A COSC4381Spring2020
#SBATCH --mail-type=all

module load python3/3.7.0

python3 train_color_denoising.py --train_batch 16 --noise_level 50 --device_id 0 --model_name 1024_CD_group_50_16 &

python3 train_gray_denoising.py --train_batch 16 --noise_level 25 --device_id 1 --model_name 1024_GD_group_25_16
