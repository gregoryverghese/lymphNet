

#$ -l tmem=15G
#$ -l h_vmem=15G
#$ -l h_rt=200000
#$ -j y
#$ -N 'calculate_weights_bi' 

source /share/apps/source_files/python/python-3.7.0.source
python3 /home/verghese/scripts/python/calculate_classweights_bi.py

