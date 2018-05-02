#!/bin/bash

. pymu 1
export PYTHONUNBUFFERED=1

# Use python as shell
#$ -S /bin/bash

# Preserve environment variables
#$ -V

# Execute from current working directory
#$ -cwd

# Merge standard output and standard error into one file
#$ -j yes

# Standard name of the job (if none is given on the command line):
#$ -N cross_pred_avg_esn_10000_bocf_uv_no_noise_0025

# Path for the output files
#$ -o /home/roland/paper_correction/new_us

# Limit memory usage
#$ -hard -l h_vmem=62G

# array range
#$ -t 1-7

# parallel
#$ -pe mvapich2-grannus05 16

#$ -q mvapich2-grannus05.q

python cross_prediction_avg_p.py ESN bocf_uv --use_no_noise
