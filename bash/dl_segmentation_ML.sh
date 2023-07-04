#!/bin/bash

#$ -l tmem=10G
#$ -l gpu=true
#$ -pe gpu 2
#$ -R y
#$ -l h_rt=45:00:00
#$ -j y
#$ -N 'dlv3+base-100e'
#$ -cwd

hostname
date

source /share/apps/source_files/python/python-3.9.5.source
source /share/apps/source_files/cuda/cuda-11.2.source
export LD_LIBRARY_PATH=/share/apps/TensorRT-6.0.1.8/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH
#python3 /home/hrafique/lymphnode-keras/src/tuning.py -rp /SAN/colcc/WSI_LymphNodes_BreastCancer/HollyR/data/patches/10x/10x-1024-512/tfrecords -rd baseline -cf /home/hrafique/lymphnode-keras/config/config_germinal.yaml -cp checkpoints -op /SAN/colcc/WSI_LymphNodes_BreastCancer/HollyR/output -tp /SAN/colcc/WSI_LymphNodes_BreastCancer/HollyR/data/patches/10x/testing/baseline -p True -mn deeplabv3plus

python3 /home/mengyuli/projects/BreastCancer/script_mod_Mengyuan/20230620-Holly/lymphnode-keras/src/tuning.py -rp /SAN/colcc/WSI_LymphNodes_BreastCancer/Mengyuan/Zooniverse/v6_2.5x_patch_256_slide_128/1fore/ -rd tfrecords -cf config_germinal_2.5_yaml -cp checkpoints -op /SAN/colcc/WSI_LymphNodes_BreastCancer/Mengyuan/Zooniverse/v6_2.5x_patch_256_slide_128/1fore/output -tp /SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/lymphnode-keras/data/patches/v1/10x/testing -p True -mn attention

date
