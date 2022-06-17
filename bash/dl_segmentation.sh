#!/bin/bash

#$ -l tmem=6G
#$ -l gpu=true
#$ -pe gpu 4
#$ -R y
#$ -l h_rt=16:00:00
#$ -j y
#$ -N 'attn-40e-1'
hostname
date
source /share/apps/source_files/python/python-3.7.0.source
source /share/apps/source_files/cuda/cuda-11.2.source
export LD_LIBRARY_PATH=/share/apps/TensorRT-6.0.1.8/lib:$LD_LIBRARY_PATH
python3 /home/hrafique/lymphnode-keras/src/tuning.py -rp /SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/lymphnode-keras/data/patches/segmentation/2.5x/one/data_4/tfrecords -rd tfrecords_g_256_32 -cf /home/hrafique/lymphnode-keras/config/config_germinal.yaml -cp checkpoints -op /SAN/colcc/WSI_LymphNodes_BreastCancer/HollyR/output -tp /SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/lymphnode-keras/data/patches/segmentation/2.5x/one/testing -p True -mn attention
date
