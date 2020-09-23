
#$ -l tmem=20G
#$ -l h_vmem=20G
#$ -l h_rt=24:00:00
#$ -j y
#$ -N 'tfrecords_write' 

feature="$1"
zoom="$2"
step="$3"
echo "$feature"
echo "$zoom"
echo "$step"

source /share/apps/source_files/python/python-3.7.0.source
source /share/apps/source_files/cuda/cuda-10.0.source
export LD_LIBRARY_PATH=/share/apps/TensorRT-5.1.5.0/lib:$LD_LIBRARY_PATH
python3 ~/scripts/python/data/tfrecord_write.py -fp=/SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/data/patches/segmentation/"$zoom"/one/data/images/images_"$feature"_"$step" -mp=/SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/data/patches/segmentation/"$zoom"/one/data/masks/masks_"$feature"_"$step" -op=/SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/data/patches/segmentation/"$zoom"/one/tfrecords/tfrecords_"$feature"_"$step" -cf=/home/verghese/config/config_tf.json
