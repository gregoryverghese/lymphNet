
#$ -l tmem=12.5G
#$ -l gpu=true
#$ -pe gpu 2
#$ -R y
#$ -l h_rt=36:00:00
#$ -j y
#$ -N 'ln_classificaton_bin__test2' 

source /share/apps/source_files/python/python-3.7.0.source
source /share/apps/source_files/cuda/cuda-10.0.source
export LD_LIBRARY_PATH=/share/apps/TensorRT-5.1.5.0/lib:$LD_LIBRARY_PATH
python3 /home/verghese/scripts/python/ln_classification_VGG16_aug_1.py
