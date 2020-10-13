#!/bin/bash



#SBATCH -J gan_training
#SBATCH -o ./logs/output_CD_GROUP_BS_8_o.%j
#SBATCH -e ./logs/output_CD_GROUP_BS_8_e.%j
#SBATCH -p gtx
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 24:00:00
#SBATCH --mail-user=asedhain@patriots.uttyler.edu
#SBATCH -A COSC4381Spring2020
#SBATCH --mail-type=all 

echo "Setting env for Maverick2"
source ../maverick_init.sh

python3 train_color_denoising.py --train_batch 8

echo "Complete"
