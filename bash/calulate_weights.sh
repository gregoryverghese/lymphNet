

#$ -l tmem=15G
#$ -l h_vmem=15G
#$ -l h_rt=200000
#$ -j y
#$ -N 'calculate_weights_g_1024_10x' 

source /share/apps/source_files/python/python-3.7.0.source
python3 /home/verghese/lymphSeg-keras/lymphnode/src/preprocessing/calculate_classweights.py -mp='/SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/data/patches/segmentation/10x/one/data_4/masks/masks_g_1536_512' -op='~/weights' -sn='weights_1536_data4' -nc=2


