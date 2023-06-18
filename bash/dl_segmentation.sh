#!/bin/bash

#$ -l tmem=12G
#$ -l gpu=true
#$ -pe gpu 4
#$ -R y
#$ -l h_rt=32:00:00
#$ -j y
#$ -N 'multiscale_g_10x_4gpu_vahadane'

source /share/apps/source_files/python/python-3.7.2.source
source /share/apps/source_files/cuda/cuda-11.2.source
export LD_LIBRARY_PATH=/share/apps/TensorRT-5.1.5.0/lib:$LD_LIBRARY_PATH

python3 /home/verghese/lymphSeg-keras/lymphnode/src/tuning.py \
	-rp  /SAN/colcc/WSI_LymphNodes_BreastCancer/HollyR/data/patches/tfrecords/10x/10x-1024-512 \
	-rd vahadane \
	-cf /home/verghese/lymphSeg-keras/lymphnode/config/config_germinal.yaml \
	-cp checkpoints \
	-op /SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/lymphnode-keras/output \
	-mn multiscale \
	-tp /SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/lymphnode-keras/data/testing/baseline

#python3 /home/verghese/lymphSeg-keras/lymphnode/src/tuning.py -rp /SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/lymphnode-keras/data/tfrecords/v2/10x-1024-512 -rd baseline -cf /home/verghese/lymphSeg-keras/lymphnode/config/config_germinal.yaml -cp checkpoints -op /SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/lymphnode-keras/output -mn multiscale -tp /SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/lymphnode-keras/data/testing/vah-norm-4
