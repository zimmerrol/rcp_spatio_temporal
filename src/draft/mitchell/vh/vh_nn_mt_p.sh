#!/bin/bash

. pymu 1

# Use python as shell
#$ -S /bin/bash

# Preserve environment variables
#$ -V

# Execute from current working directory
#$ -cwd

# Merge standard output and standard error into one file
#$ -j yes

# Standard name of the job (if none is given on the command line):
#$ -N vh_nn_htov

# Path for the output files
#$ -o /home/roland/q-out/

# Limit memory usage
#$ -hard -l h_vmem=62G

# array range
#$ -t 1-72

# parallel
#$ -pe mvapich2-grannus05 16

#$ -q mvapich2-grannus05.q

python vh_nn_mt_p.py -u
