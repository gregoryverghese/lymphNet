!/bin/bash

#$ -l tmem=12.5G
#$ -l gpu=true
#$ -pe gpu 2 
#$ -R y
#$ -l h_rt=20:00:00
#$ -j y
#$ -N 'unet_models_s_2.5x_64' 
 
source /share/apps/source_files/python/python-3.7.0.source
source /share/apps/source_files/cuda/cuda-10.0.source
export LD_LIBRARY_PATH=/share/apps/TensorRT-5.1.5.0/lib:$LD_LIBRARY_PATH
python3 /home/verghese/scripts/python/tuning.py -rp /SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/data/patches/segmentation/2.5x/one/tfrecords_s_64 -cf config/config_sinus_template.json -cp checkpoints -op predictions
