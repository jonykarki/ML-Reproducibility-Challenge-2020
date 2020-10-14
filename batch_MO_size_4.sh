#!/bin/bash



#SBATCH -J 1014_MO_group_25_4
#SBATCH -o ./logs/output_MO_GROUP_BS_4_o.%j
#SBATCH -e ./logs/output_MO_GROUP_BS_4_e.%j
#SBATCH -p gtx
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 24:00:00
#SBATCH --mail-user=asedhain@patriots.uttyler.edu
#SBATCH -A COSC4381Spring2020
#SBATCH --mail-type=all 

echo "Setting env for Maverick2"
source $WORK/mlreproduce/maverick_init.sh

python3 ./train_mosaic.py --train_batch 4 --model_name 1014_MO_group_25_4

echo "Complete"
