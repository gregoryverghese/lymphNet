

#$ -l tmem=15G
#$ -l h_vmem=15G
#$ -l h_rt=200000
#$ -j y
#$ -N 'calculate_weights_g_10x_sparse' 

source /share/apps/source_files/python/python-3.7.0.source
python3 /home/verghese/scripts/python/preprocessing/calculate_classweights.py -mp='/SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/data/patches/segmentation/10x/one/data/masks/masks_g_1536' -op='~/weights' -sn='weights_g_10x_sparse' -nc=2


