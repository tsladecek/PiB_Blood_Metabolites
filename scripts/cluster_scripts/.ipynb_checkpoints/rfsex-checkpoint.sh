#!/bin/bash
#SBATCH --partition normal
#SBATCH --mem-per-cpu 1G
#SBATCH -c 1

python rf_sex.py
