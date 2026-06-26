#!/bin/bash
#SBATCH --job-name=qwen_lab4
#SBATCH --output=resultado_%j.log
#SBATCH --error=error_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --nodelist=g001

module load python3/3.13.2
source .venv/bin/activate

python main.py