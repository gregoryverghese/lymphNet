#!/bin/bash

#$ -l tmem=30G
#$ -l h_vmem=30G
#$ -l h_rt=72:00:00
#$ -j y
#$ -N 'patch_extract_40x_s' 

source /share/apps/source_files/python/python-3.7.0.source
export LD_LIBRARY_PATH=/share/apps/openslide-3.4.1/lib:$LD_LIBRARY_PATH
python3 ~/scripts/python/dev/patchmask_extraction.py -np /SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/data/wsi/Guys/all/wsi -xp /SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/data/wsi/Guys/all/annotations -op /SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/data/patches/classification/binary -cp /home/verghese/config/config_pextract_classification.json

