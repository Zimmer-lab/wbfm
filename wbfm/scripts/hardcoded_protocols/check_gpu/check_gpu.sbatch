#!/usr/bin/env bash

#SBATCH --job-name=gpu_check
#SBATCH --mem=1G
#SBATCH --cpus-per-task=1
#SBATCH --output=log_gpu_test-%j.out    # Standard output and error log
#SBATCH --time=0:10:00
#SBATCH --gres=gpu:1
#SBATCH --constraint="l40s|a30|t4|v100|l4"

pwd; hostname; date

CMD_DIR="/lisc/data/scratch/neurobiology/zimmer/wbfm/code/wbfm/wbfm/scripts"
CMD="${CMD_DIR}/hardcoded_protocols/check_gpu/check_gpu.py"

python $CMD
