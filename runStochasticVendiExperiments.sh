#!/bin/bash
#PBS -l walltime=48:00:00
#PBS -l select=1:ncpus=10:mem=50gb

# bash script to run generalisation experiments on HPC
cd ${PBS_O_WORKDIR}

# Launch virtual environment
module load anaconda3/personal

# Define a list of integer values
ds_size=(1000 2000 5000 10000)
sample_size=(10 50 100 200)

# Loop through each number and call the Python script
for N in "${ds_size[@]}"; do
    for n in "${sample_size[@]}"; do
        python vendi_random_sampling.py -dd "/rds/general/user/kc2322/home/data" -N "$N" -n "$n"
    done
done