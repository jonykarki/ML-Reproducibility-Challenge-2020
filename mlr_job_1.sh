#!/bin/bash

#SBATCH -J MLR_1
#SBATCH -p p100
#SBATCH -o ./logs/1027_D_cdsc_25_16_o.%j
#SBATCH -e ./logs/1027_D_cdsc_25_16_e.%j
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 24:00:00
#SBATCH --mail-user=jkarki@patriots.uttyler.edu
#SBATCH -A COSC4381Spring2020
#SBATCH --mail-type=all

module load python3/3.7.0

python3 train_mosaic.py --train_batch 16 --device_id 0 &

python3 train_mosaic.py --train_batch 16 --mode sc --device_id 1 