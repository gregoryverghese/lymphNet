

#$ -l tmem=15G
#$ -l h_vmem=15G
#$ -l h_rt=200000
#$ -j y
#$ -N 'calculate_weights_g_sparse_1024' 

source /share/apps/source_files/python/python-3.7.0.source
python3 /home/verghese/breastcancer_ln_deeplearning/scripts/python/preprocessing/calculate_classweights.py -mp='/SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/data/patches/segmentation/2.5x/one/data_2/masks/masks_g_1024_256' -op='~/weights' -sn='weights_g_1024_256' -nc=2


