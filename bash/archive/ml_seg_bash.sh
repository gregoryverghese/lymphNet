

#$ -l tmem=300G
#$ -l h_vmem=300G
#$ -l h_rt=200000
#$ -j y
#$ -N 'ln_ml_seg' 

source /share/apps/source_files/python/python-3.7.0.source
python3 /home/verghese/scripts/python/ln_ml_segmentation_multi.py

