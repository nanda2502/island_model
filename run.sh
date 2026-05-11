#!/bin/bash
#SBATCH -p fat_genoa
#SBATCH -N 1
#SBATCH --cpus-per-task=192
#SBATCH -t 24:00:00

export ISLAND_MODEL_DENSE_PAYOFF_CACHE=0
export ISLAND_MODEL_MEMORY_GIB=384
export ISLAND_MODEL_MEMORY_FRACTION=0.90

cd build

./island_model 0 0