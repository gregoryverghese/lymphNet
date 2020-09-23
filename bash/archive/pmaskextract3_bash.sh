#!/bin/bash

#$ -l tmem=50G
#$ -l h_vmem=50G
#$ -l h_rt=24:00:00
#$ -j y
#$ -N 'patchExtraction' 

source /share/apps/source_files/python/python-3.7.0.source
export LD_LIBRARY_PATH=/share/apps/openslide-3.4.1/lib:$LD_LIBRARY_PATH
python3 ~/scripts/python/patchmask_extraction_bi.py
