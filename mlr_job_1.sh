#!/bin/bash

#SBATCH -J MLR_1
#SBATCH -p p100
#SBATCH -o ./logs/112_CD_g_510_16_o.%j
#SBATCH -e ./logs/112_CD_g_510_16_e.%j
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 24:00:00
#SBATCH --mail-user=jkarki@patriots.uttyler.edu
#SBATCH -A COSC4381Spring2020
#SBATCH --mail-type=all

source ~/.bash_profile

python3 train_color_denoising.py --train_batch 16 --noise_level 10 --model_name 1102_CD_group_10_16 --device_id 0 &

python3 train_color_denoising.py --train_batch 16 --noise_level 5 --model_name 1102_CD_group_5_16 --device_id 1 