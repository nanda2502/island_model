#!/bin/bash
#SBATCH -p genoa
#SBATCH -N 1
#SBATCH --cpus-per-task=192
#SBATCH -t 24:00:00

cd build

./island_model 0 0