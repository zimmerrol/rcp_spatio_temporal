#!/bin/bash

. pymu 1

# Use python as shell
#$ -S /bin/bash

# Preserve environment variables
#$ -V

# Execute from current working directory
#$ -cwd

# Standard name of the job (if none is given on the command line):
#$ -N vh_esn_gs

# Path for the output files
#$ -o /home/roland/q-out/

# Limit memory usage
#$ -hard -l h_vmem=62G

# array range
#$ -t 1-12

# parallel
#$ -pe mvapich2-grannus04 16

#$ -q mvapich2-grannus04.q

python vh_cross_pred_esn_gs_p.py -u
