

#$ -l tmem=16G
#$ -l h_vmem=16G
#$ -l h_rt=200000
#$ -j y
#$ -N 'preprocessing' 

source /share/apps/source_files/python/python-3.7.0.source
python3 /home/verghese/scripts/python/seg_preprocessing.py

