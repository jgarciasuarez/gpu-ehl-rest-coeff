#!/bin/bash
#SBATCH --job-name=gpu-ehl-sweep
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --output=gpu-ehl-sweep_%j.out
#SBATCH --error=gpu-ehl-sweep_%j.err

set -euo pipefail

# Adjust the module and environment activation to your cluster setup.
# Example using conda:
# module load cuda
# source /path/to/conda/etc/profile.d/conda.sh
# conda activate gpu-ehl

# Example using uv:
# uv run --python .venv python -m gpu_ehl.cli sweep "$@"

export XLA_FLAGS="--xla_force_host_platform_device_count=$SLURM_CPUS_PER_TASK"

jax-ehl sweep "$@"
