#!/bin/bash

#SBATCH -J MLR_4
#SBATCH -p gtx
#SBATCH -o ./logs/1027_Testlog_o.%j
#SBATCH -e ./logs/1027_Testlog_e.%j
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 24:00:00
#SBATCH --mail-user=jkarki@patriots.uttyler.edu
#SBATCH -A COSC4381Spring2020
#SBATCH --mail-type=all

module load python3/3.7.0

python3 MLR_TEST_LOG.py
