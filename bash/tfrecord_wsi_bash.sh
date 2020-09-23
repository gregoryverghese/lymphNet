
#$ -l tmem=20G
#$ -l h_vmem=20G
#$ -l h_rt=24:00:00
#$ -j y
#$ -N 'tfrecords_write' 


source /share/apps/source_files/python/python-3.7.0.source
source /share/apps/source_files/cuda/cuda-10.0.source
export LD_LIBRARY_PATH=/share/apps/TensorRT-5.1.5.0/lib:$LD_LIBRARY_PATH

python3 scripts/python/dev/tfrecord_wsi_write.py
