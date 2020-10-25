#!/bin/bash

#SBATCH -J MLR_3
#SBATCH -p p100
#SBATCH -o ./logs/1025_GD_group_2550_32_o.%j
#SBATCH -e ./logs/1025_GD_group_2550_32_e.%j
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 24:00:00
#SBATCH --mail-user=jkarki@patriots.uttyler.edu
#SBATCH -A COSC4381Spring2020
#SBATCH --mail-type=all

module load python3/3.7.0

python3 train_color_denoising.py --train_batch 16 --noise_level 25 --device_id 0 --num_epochs 600 --out_dir /trained_600 &

python3 train_color_denoising.py --train_batch 16 --noise_level 50 --device_id 1 --num_epochs 600 --out_dir /trained_600
