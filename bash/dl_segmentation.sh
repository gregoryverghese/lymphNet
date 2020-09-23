bin/bash

#$ -l tmem=6G
#$ -l gpu=true
#$ -pe gpu 4
#$ -R y
#$ -l h_rt=16:00:00
#$ -j y
#$ -N 'unet_models_g_2.5x_multiscale_sparse

 
source /share/apps/source_files/python/python-3.7.0.source
source /share/apps/source_files/cuda/cuda-10.1.source
export LD_LIBRARY_PATH=/share/apps/TensorRT-5.1.5.0/lib:$LD_LIBRARY_PATH
python3 /home/verghese/scripts/python/tuning.py -rp /SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/data/patches/segmentation/2.5x/one/tfrecords -rd tfrecords_g_1536 -cf config/config_germinal_template.json -cp checkpoints -op output/ -mn multiscale
