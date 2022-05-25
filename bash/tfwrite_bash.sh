
#$ -l tmem=20G
#$ -l h_vmem=20G
#$ -l h_rt=24:00:00
#$ -j y
#$ -N 'tfrecords_write' 

feature="$1"
zoom="$2"
size="$3"
step="$4"
echo "$feature"
echo "$zoom"
echo "$size"
echo "$step"

source /share/apps/source_files/python/python-3.7.0.source
source /share/apps/source_files/cuda/cuda-10.0.source
export LD_LIBRARY_PATH=/share/apps/TensorRT-5.1.5.0/lib:$LD_LIBRARY_PATH
python3 ~/breastcancer_ln_deeplearning/scripts/python/data/tfrecord_write.py -fp=/SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/data/patches/segmentation/"$zoom"/one/data_4/images/images_"$feature"_"$size"_"$step" -mp=/SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/data/patches/segmentation/"$zoom"/one/data_4/masks/masks_"$feature"_"$size"_"$step" -op=/SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/data/patches/segmentation/"$zoom"/one/data_4/tfrecords/tfrecords_"$feature"_"$size"_"$step" -cf=/home/verghese/breastcancer_ln_deeplearning/scripts/config/config_tf.json
