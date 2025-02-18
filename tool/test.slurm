#!/bin/bash
#SBATCH -o spsm.out 
#SBATCH -J spsmTest
#SBATCH --nodes=1

module load cuda/12.0
cd /data/home/gcx/alphasparse_muxi
bash tool/run_florida.sh
