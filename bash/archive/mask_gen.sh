#/bin/bash

#$ -l tmem=10G
#$ -l h_vmem=10G
#$ -l h_rt=5:00:00
#$ -j y
#$ -N 'maskgen' 

source /share/apps/source_files/python/python-3.7.0.source
python3 ~/scripts/python/masks_generator.py
