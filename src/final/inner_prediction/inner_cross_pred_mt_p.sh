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
#$ -N inner_cross_pred_nn_v

# Path for the output files
#$ -o /home/roland/q-out_inner_cross/

# Limit memory usage
#$ -hard -l h_vmem=62G

# array range
#$ -t 1-138

# parallel
#$ -pe mvapich2-grannus06 16

#$ -q mvapich2-grannus06.q

python inner_cross_pred_mt_p.py NN v
