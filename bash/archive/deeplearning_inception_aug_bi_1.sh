
#$ -l tmem=12.5G
#$ -l gpu=true
#$ -pe gpu 3
#$ -R y
#$ -l h_rt=36:00:00
#$ -j y
#$ -N 'ln_classificaton_inception_bi' 

source /share/apps/source_files/python/python-3.7.0.source
source /share/apps/source_files/cuda/cuda-10.0.source
export LD_LIBRARY_PATH=/share/apps/TensorRT-5.1.5.0/lib:$LD_LIBRARY_PATH
python3 /home/verghese/scripts/python/ln_classification_Inception_aug_bi.py
