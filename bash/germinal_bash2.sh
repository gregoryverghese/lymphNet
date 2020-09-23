#!/bin/bash

#$ -l tmem=40G
#$ -l gpu=true
#$ -R y
#$ -l h_rt=20:00:00
#$ -j y
#$ -N 'germinal' 
 
source /share/apps/source_files/python/python-3.7.0.source
python3 germinal.py
