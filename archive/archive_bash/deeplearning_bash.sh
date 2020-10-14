#!/bin/bash

#$ -l tmem=15G
#$ -l h_vmem=15G
#$ -l gpu=true
#$ -pe gpu 2
#$ -R y
#$ -l h_rt=12:00:00
#$ -j y
#$ -N 'ln_classificaton_bin_1' 

source /share/apps/source_files/python/python-3.7.0.source
source /share/apps/source_files/cuda/cuda-10.0.source
export LD_LIBRARY_PATH=/share/apps/TensorRT-5.1.5.0/lib:$LD_LIBRARY_PATH
python3 ~/scripts/python/ln_classification.py

