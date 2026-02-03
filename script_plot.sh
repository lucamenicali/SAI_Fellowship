#!/bin/bash

#$ -pe smp 8
#$ -q long
#$ -N vit_cae_figs
#$ -t 6

module load python3

export OMP_NUM_THREADS=${NSLOTS}

python3 plot_results.py
