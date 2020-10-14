#!/bin/bash



#SBATCH -J 1014_GD_group_25_2
#SBATCH -o ./logs/output_GD_GROUP_BS_2_o.%j
#SBATCH -e ./logs/output_GD_GROUP_BS_2_e.%j
#SBATCH -p gtx
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 24:00:00
#SBATCH --mail-user=asedhain@patriots.uttyler.edu
#SBATCH -A COSC4381Spring2020
#SBATCH --mail-type=all 

echo "Setting env for Maverick2"
source $WORK/mlreproduce/maverick_init.sh

python3 ./train_gray_denoising.py --train_batch 2 --model_name 1014_GD_group_25_2

echo "Complete"
