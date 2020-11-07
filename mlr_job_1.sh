#!/bin/bash

#SBATCH -J MLR_1
#SBATCH -p p100
#SBATCH -o ./logs/115_CD_sc_510_16_o.%j
#SBATCH -e ./logs/115_CD_sc_510_16_e.%j
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 24:00:00
#SBATCH --mail-user=jkarki@patriots.uttyler.edu
#SBATCH -A COSC4381Spring2020
#SBATCH --mail-type=all

source ~/.bash_profile

python3 train_mosaic.py --train_batch 16 --noise_level 25 --model_name 1027_D_group_25_16 --device_id 0 