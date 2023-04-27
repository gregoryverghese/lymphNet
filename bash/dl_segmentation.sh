#!/bin/bash

#$ -l tmem=85G
#$ -l gpu=true
#$ -pe gpu 1
#$ -R y
#$ -l h_rt=24:00:00
#$ -j y
#$ -N 'multiscale_g_10x'

source /share/apps/source_files/python/python-3.7.0.source
source /share/apps/source_files/cuda/cuda-11.2.source
export LD_LIBRARY_PATH=/share/apps/TensorRT-5.1.5.0/lib:$LD_LIBRARY_PATH
#python3 /home/verghese/lymphnode-keras/lymphnode/src/tuning.py -rp /SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/lymphnode-keras/data/patches/segmentation/2.5x/one/data_4/tfrecords -rd tfrecords_g_256_32 -cf /home/verghese/lymphnode-keras/lymphnode/config/config_germinal_template.json -cp checkpoints -op /SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/lymphnode-keras/output -mn attention

python3 /home/verghese/lymphSeg-keras/lymphnode/src/tuning.py -rp /SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/lymphnode-keras/data/tfrecords/v2/10x-1024-512 -rd baseline -cf /home/verghese/lymphSeg-keras/lymphnode/config/config_germinal.yaml -cp checkpoints -op /SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/lymphnode-keras/output -mn multiscale -tp /SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/lymphnode-keras/data/testing/vah-norm-4
