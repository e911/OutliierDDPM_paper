#!/bin/bash -l

#SBATCH --account=examm-rnn
#SBATCH --partition=tier3

#SBATCH --mail-type=ALL
#SBATCH --mail-user=pt6757@rit.edu

#SBATCH --time 1-22:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=10
#SBATCH --cpus-per-task=1
#SBATCH --mem=20g

#SBATCH --gres=gpu:a100:1

#SBATCH --job-name=testJob

# Activate the conda environment
conda activate diffuse  # or `conda activate diffuse` depending on your conda setup

# Run the Python script with command-line arguments
python -m diffuseNew.main --mode train --epochs 10 --channels 1 --batch_size 64 --learning_rate 0.0008 --ddpm_timesteps 1000 --n 2 --steps 50