#!/bin/bash

#$ -l tmem=10G
#$ -l h_vmem=10G
#$ -l h_rt=12:00:00
#$ -j y
#$ -N 'convert' 

source /share/apps/source_files/python/python-3.7.0.source
python3 ~/convert.py
