#!/bin/bash
#PBS -l walltime=12:00:00
#PBS -l select=1:ncpus=15:mem=80gb:ngpus=1:gpu_type=RTX6000

# bash script to run generalisation experiments on HPC
cd ${PBS_O_WORKDIR}

# Launch virtual environment
module load anaconda3/personal

# install requirements
#pip install -r requirements.txt

# run experiments
python computationalComplexity.py
