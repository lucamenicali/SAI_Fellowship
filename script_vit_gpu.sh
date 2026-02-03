#!/bin/bash

#$ -pe smp 8
#$ -q gpu@@crc_gpu # acms_a40 # #crippa 
#$ -l gpu_card=1
#$ -N vit_cae_compression
#$ -t 9-11

module load tensorflow

export OMP_NUM_THREADS=${NSLOTS}

python3 train_model.py
