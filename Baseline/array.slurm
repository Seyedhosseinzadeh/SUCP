#!/bin/bash
#SBATCH --job-name=recsys
#SBATCH --time=48:00:00
#SBATCH --array=1-22
#SBATCH --mem=16g

# Set-up the environment.

source ~/anaconda3/etc/profile.d/conda.sh
conda activate py27

python --version

HPARAMS_FILE=hyperparams

python recommendation_overlap.py $(head -$SLURM_ARRAY_TASK_ID $HPARAMS_FILE | tail -1)

