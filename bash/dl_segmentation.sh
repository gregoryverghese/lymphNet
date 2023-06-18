#!/bin/bash

#$ -l tmem=28G
#$ -l gpu=true
#$ -pe gpu 2
#$ -R y
#$ -l h_rt=45:00:00
#$ -j y
#$ -N 'dlv3+base-100e'

hostname
date

source /share/apps/source_files/python/python-3.7.0.source
source /share/apps/source_files/cuda/cuda-11.2.source
export LD_LIBRARY_PATH=/share/apps/TensorRT-6.0.1.8/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH
python3 /home/hrafique/lymphnode-keras/src/tuning.py -rp /SAN/colcc/WSI_LymphNodes_BreastCancer/HollyR/data/patches/10x/10x-1024-512/tfrecords -rd baseline -cf /home/hrafique/lymphnode-keras/config/config_germinal.yaml -cp checkpoints -op /SAN/colcc/WSI_LymphNodes_BreastCancer/HollyR/output -tp /SAN/colcc/WSI_LymphNodes_BreastCancer/HollyR/data/patches/10x/testing/baseline -p True -mn deeplabv3plus

date
