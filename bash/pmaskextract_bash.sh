#!/bin/bash

#$ -l tmem=40G
#$ -l h_vmem=40G
#$ -l h_rt=24:00:00
#$ -j y
#$ -N 'patchmask_2.5x_g_256' 
#$ -o /home/verghese/breastcancer_ln_deeplearning/status_files/

source /share/apps/source_files/python/python-3.7.0.source
export LD_LIBRARY_PATH=/share/apps/openslide-3.4.1/lib:$LD_LIBRARY_PATH
python3 ~/breastcancer_ln_deeplearning/scripts/python/preprocessing/patching.py -np /SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/data/wsi/Guys/all/wsi -xp /SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/data/wsi/Guys/sum_swap_toms/annotations -op /SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/data/patches/segmentation/2.5x/one/data_4 -cp /home/verghese/breastcancer_ln_deeplearning/scripts/config/config_pextract_g.json -id images/images_g_256_32 -md masks/masks_g_256_32


