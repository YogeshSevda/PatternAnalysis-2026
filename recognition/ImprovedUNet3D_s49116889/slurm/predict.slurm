#!/bin/bash
# Test-set inference + README figures for both trained models.
# Submit from recognition/ImprovedUNet3D_s49116889 with the torch env active:
#   conda activate torch
#   sbatch slurm/predict.slurm
#SBATCH --job-name=predict
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%x-%j.out

python predict.py --models 3d 2d
