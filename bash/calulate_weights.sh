

#$ -l tmem=15G
#$ -l h_vmem=15G
#$ -l h_rt=200000
#$ -j y
#$ -N 'calculate_weights_gs_data2' 

source /share/apps/source_files/python/python-3.7.0.source
python3 /home/verghese/breastcancer_ln_deeplearning/scripts/python/preprocessing/calculate_classweights_2.py -mp='/SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/data/patches/segmentation/2.5x/two/data_2/masks/masks_gs_256_32' -op='~/weights' -sn='weights_gs_256_32_data2' -nc=3


