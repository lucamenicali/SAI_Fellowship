#!/bin/bash

#$ -pe smp 8
#$ -q gpu@@crippa # crippa
#$ -l gpu_card=1
#$ -N vit_cae_results
#$ -t 6

module load tensorflow

export OMP_NUM_THREADS=${NSLOTS}

python3 get_results.py
