#!/bin/bash

#$ -l tmem=40G
#$ -l h_vmem=40G
#$ -l h_rt=24:00:00
#$ -j y
#$ -N 'patchmask_2.5x_s_256_64_data4' 

source /share/apps/source_files/python/python-3.7.0.source
export LD_LIBRARY_PATH=/share/apps/openslide-3.4.1/lib:$LD_LIBRARY_PATH
python3 ~/breastcancer_ln_deeplearning/scripts/python/preprocessing/patchmask_extraction.py -np /SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/data/wsi/Guys/sum_swap_toms/wsi -xp /SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/data/wsi/Guys/sum_swap_toms/annotations -op /SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/data/patches/segmentation/2.5x/one/data_4 -cp /home/verghese/breastcancer_ln_deeplearning/scripts/config/config_pextract_s.json -id images/images_s_256_64 -md masks/masks_s_256_64

